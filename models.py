import enum
import torch
from torch import nn, Tensor
from typing import List, Callable
from transformers import AutoTokenizer, AutoModel, AutoConfig
from loggers import getLogger
logger = getLogger(__name__)

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
        bert_reducer: Callable[[Tensor], Tensor] = average_layers_and_tokens
    ):
        super().__init__()

        self.bert_version = bert_version.value
        self.bert_reducer = bert_reducer
        self.config = AutoConfig.from_pretrained(self.bert_version, output_hidden_states=True, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_version, config=self.config)
        self.bert = AutoModel.from_pretrained(self.bert_version, config=self.config)
        self.embeddings_dim = self.config.hidden_size
        self.hidden_size = hidden_size or self.embeddings_dim * 2

        self.linear = nn.Linear(self.embeddings_dim, self.hidden_size)
        self.cached_embeddings = {}

    def forward(self, documents: List[str]) -> Tensor:
        embeddings = self.compute_embeddings(documents)
        return self.linear(embeddings)

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
        hits = []
        misses = []
        for doc in documents:
            if doc in self.cached_embeddings:
                hits.append(self.cached_embeddings[doc])
            else:
                hits.append(None)
                misses.append(doc)

        nb_misses = len(misses)
        if nb_misses:
            logger.debug("Computing BERT embeddings for %d cache misses (%d hits)", nb_misses, len(hits) - nb_misses)
            with torch.no_grad():
                inputs = self.tokenizer(misses, return_tensors="pt", padding=True, truncation=True)
                hidden_states = self.bert(**inputs).hidden_states
                embeddings = self.bert_reducer(torch.stack(hidden_states))
            idx = 0
            for i, doc in enumerate(hits):
                if doc is None:
                    array = embeddings[idx].numpy()
                    hits[i] = self.cached_embeddings[misses[idx]] = array
                    idx += 1

        return torch.stack([torch.from_numpy(hit) for hit in hits])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
