import enum
import torch
from torch import nn, Tensor
from typing import List, Callable
from transformers import AutoTokenizer, AutoModel, AutoConfig


class BERTVersion(enum.Enum):
    """
    Enum representing supported BERT versions
    """
    BERT_BASE_UNCASED = 'bert-base-uncased'


def average_layers_and_tokens(tensor: Tensor) -> Tensor:
    return average_axis(0, average_axis(len(tensor.shape) - 2, tensor))


def average_axis(axis: int, tensor: Tensor) -> Tensor:
    return tensor.sum(axis) / tensor.shape[axis]


class BertAsFeatureExtractorEncoder(nn.Module):
    def __init__(
        self,
        compute_embeddings: Callable[[List[str]], Tensor],
        embeddings_size: int,
        hidden_size: int,
        bert_version: BERTVersion,
        bert_reducer: Callable[[Tensor], Tensor] = average_layers_and_tokens
    ):
        super().__init__()

        self.compute_embeddings = compute_embeddings
        self.embeddings_dim = embeddings_size
        self.linear = nn.Linear(self.embeddings_dim, hidden_size)

        self.bert_version = bert_version
        self.bert_reducer = bert_reducer
        self.config = AutoConfig.from_pretrained(self.bert_version, output_hidden_states=True, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_version, config=self.config)
        self.bert = AutoModel.from_pretrained(self.bert_version, config=self.config)

    def forward(self, documents: List[str]):
        inputs = self.tokenizer(documents, return_tensors="pt", padding=True, truncation=True)
        hidden_states = self.bert(**inputs).hidden_states
        embeddings = self.bert_reducer(torch.stack(hidden_states))

        return self.linear(embeddings)
