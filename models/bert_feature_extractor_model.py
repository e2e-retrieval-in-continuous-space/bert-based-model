import enum
import torch
import uuid
from pathlib import Path
from torch import nn, Tensor
from typing import List, Callable
from transformers import AutoTokenizer, AutoModel, AutoConfig
from loggers import getLogger
import pickle

logger = getLogger(__name__)

CACHE_PATH = Path(__file__).parents[0] / "embeddings-cache"
CACHE_PATH.mkdir(parents=True, exist_ok=True)


class BERTVersion(enum.Enum):
    """
    Enum representing supported BERT versions
    """
    BASE_UNCASED = 'bert-base-uncased'


def average_layers_and_tokens(tensor: Tensor) -> Tensor:
    return average_axis(0, average_axis(len(tensor.shape) - 2, tensor))


def average_axis(axis: int, tensor: Tensor) -> Tensor:
    return tensor.sum(axis) / tensor.shape[axis]


class BERTAsFeatureExtractorEncoder(nn.Module):
    def __init__(
            self,
            bert_version: BERTVersion,
            hidden_size: int = None,
            bert_reducer: Callable[[Tensor], Tensor] = average_layers_and_tokens,
            device=None
    ):
        super().__init__()
        self.device = device
        if not self.device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.bert_version = bert_version.value
        self.bert_reducer = bert_reducer
        self.config = AutoConfig.from_pretrained(self.bert_version, output_hidden_states=True, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_version, config=self.config)
        self.bert = AutoModel.from_pretrained(self.bert_version, config=self.config)
        self.bert.to('cuda')
        self.embeddings_dim = self.config.hidden_size
        self.hidden_size = hidden_size or self.embeddings_dim * 2

        self.linear = nn.Linear(self.embeddings_dim, self.hidden_size)
        self.cache = EmbeddingsCache(bert_version.value)

    def forward(self, documents: List[str]) -> Tensor:
        embeddings = self.compute_embeddings(documents) + 0.1
        retval = self.linear(embeddings)
        return retval

    def compute_embeddings(self, documents: List[str]) -> Tensor:
        """
        Encodes a list of strings (documents) into vectors.

        Caches results between runs.

        >>> m = BERTAsFeatureExtractorEncoder(BERTVersion.BASE_UNCASED)
        >>> out = m.compute_embeddings(["I like ice cream."])

        Let's confirm the embedding is what we pre-computed earlier:
        >>> torch.mean(out)
        tensor(-0.0199)

        Compute again and confirm it's the same:
        >>> torch.mean(m.compute_embeddings(["I like ice cream."]))
        tensor(-0.0199)

        Add a cache miss into the mix and confirm both cache misses and hits make sense:
        >>> out2 = m.compute_embeddings(["I like ice cream.", "It's a blue bird.", "I like ice cream."])
        >>> torch.mean(out2[0]) == torch.mean(out2[2])
        tensor(True)
        >>> torch.mean(out2[0])
        tensor(-0.0199)
        >>> torch.mean(out2[1])
        tensor(-0.0187)
        """
        results, misses = self.cache.get_many(documents)

        nb_misses = len(misses)
        if nb_misses:
            logger.debug("Computing BERT embeddings for %d cache misses (%d hits)", nb_misses, len(results) - nb_misses)
            embeddings = self.bert_reducer(self._run_bert(documents))
            idx = 0
            for i, doc in enumerate(results):
                if doc is None:
                    results[i] = embeddings[idx]
                    self.cache[misses[idx]] = embeddings[idx].tolist()
                    idx += 1

        retval = torch.stack([Tensor(hit) for hit in results])
        return retval

    def _run_bert(self, documents):
        with torch.no_grad():
            encoded_chunks = []
            for chunk in chunks(documents, 1000):
                inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to('cuda')
                hidden_states = self.bert(**inputs).hidden_states
                encoded_chunks.append(torch.stack([s.to('cpu') for s in hidden_states]))
            return torch.stack(encoded_chunks)


class EmbeddingsCache(dict):

    def __init__(self, name, *args, **kwargs):
        super(EmbeddingsCache, self).__init__(*args, **kwargs)
        self.cache_path = CACHE_PATH / name
        self.unsaved_keys = []
        logger.info("Loading cached embeddings from %s", self.cache_path)
        try:
            for i, path in enumerate(self.cache_path.glob('*.pickle')):
                with path.open('rb') as cachefile:
                    self.update(pickle.load(cachefile))
            logger.info("%d cached embeddings loaded from %d files", len(self), i + 1)
        except Exception as e:
            logger.warn("Cached BERT embeddings from %s cannot be loaded (%s)", self.cache_path, e)

    def get_many(self, documents):
        hits = []
        misses = []
        for doc in documents:
            if doc in self:
                hits.append(self[doc])
            else:
                hits.append(None)
                misses.append(doc)
        return hits, misses

    def __setitem__(self, key, value):
        if key not in self:
            self.unsaved_keys.append(key)
        super(EmbeddingsCache, self).__setitem__(key, value)
        if len(self.unsaved_keys) > 10000:
            self.save()

    def save(self):
        nb_items = len(self.unsaved_keys)
        logger.debug("Saving %d embeddings", nb_items)
        new_path = self.cache_path / "{0}.pickle".format(uuid.uuid4())
        with new_path.open('wb') as cachefile:
            tmp_dict = {k: self[k] for k in self.unsaved_keys}
            pickle.dump(tmp_dict, cachefile)
            self.unsaved_keys = []
        logger.debug("Saved %d embeddings", nb_items)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
