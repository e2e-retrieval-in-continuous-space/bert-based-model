import enum
import torch
from torch import nn, Tensor
from typing import List, Callable
from transformers import AutoTokenizer, AutoModel, AutoConfig
from loggers import getLogger
from models.utils import EmbeddingsCache

logger = getLogger(__name__)


class BERTVersion(enum.Enum):
    """
    Enum representing supported BERT versions
    """
    BASE_UNCASED = 'bert-base-uncased'
    LARGE_UNCASED = 'bert-large-uncased'


def reducer_try_vertical_tokens(tensor: Tensor) -> Tensor:
    r = average_layers(average_axis(-1, tensor))
    shape = list(r.shape)
    shape[-1] = 768
    target = torch.zeros(*shape)
    target[:, :r.shape[-1]] = r
    return target


def reducer_all_layers(tensor: Tensor) -> Tensor:
    return average_layers(average_tokens(tensor))


def reducer_last_layer(tensor: Tensor) -> Tensor:
    return average_layers(average_tokens(tensor[:, -1, :, :]))


def reducer_2nd_last_layer(tensor: Tensor) -> Tensor:
    return average_layers(average_tokens(tensor[:, -2, :, :]))


def reducer_last_4_layers(tensor: Tensor) -> Tensor:
    return average_layers(average_tokens(tensor[:, -4:, :, :]))


def average_layers(tensor: Tensor) -> Tensor:
    return torch.mean(tensor, 1)


def average_tokens(tensor: Tensor) -> Tensor:
    return torch.mean(tensor, -2)


def average_axis(axis: int, tensor: Tensor) -> Tensor:
    return torch.mean(tensor, axis)


class BERTAsFeatureExtractorEncoder(nn.Module):
    def __init__(
            self,
            bert_version: BERTVersion,
            hidden_size: int = None,
            bert_reducer: Callable[[Tensor], Tensor] = reducer_all_layers,
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
        self.bert.to(self.device)
        self.embeddings_dim = self.config.hidden_size
        self.hidden_size = hidden_size or self.embeddings_dim * 2

        self.linear = nn.Linear(self.embeddings_dim, self.hidden_size).to(self.device)
        self.cache = EmbeddingsCache(bert_version.value)

    def forward(self, documents: List[str]) -> Tensor:
        embeddings = self.compute_sentence_embeddings(documents)
        return self.linear(embeddings)

    def compute_sentence_embeddings(self, documents: List[str]) -> Tensor:
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
            embeddings = self.bert_reducer(self._run_bert(misses))
            idx = 0
            for i, doc in enumerate(results):
                if doc is None:
                    results[i] = embeddings[idx]
                    self.cache[misses[idx]] = embeddings[idx].tolist()
                    idx += 1

        retval = torch.stack([Tensor(hit) for hit in results]).to(self.device)
        return retval

    def _run_bert(self, documents):
        with torch.no_grad():
            encoded_chunks = []
            for chunk in chunks(documents, 1000):
                inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(self.device)
                hidden_states = self.bert(**inputs).hidden_states
                stacked = torch.stack([s.to('cpu') for s in hidden_states]).transpose(0, 1)
                coords = (inputs.input_ids == 0).nonzero()  # irrelevant_tokens_coords
                stacked[coords[:, 0], :, coords[:, 1], :] = 0
                encoded_chunks.append(stacked)
            return torch.cat(encoded_chunks)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
