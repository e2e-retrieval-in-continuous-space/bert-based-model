import enum
import torch
from torch import nn, Tensor
from typing import List, Callable
from transformers import AutoTokenizer, AutoModel, AutoConfig


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

    def forward(self, documents: List[str]):
        inputs = self.tokenizer(documents, return_tensors="pt", padding=True, truncation=True)
        hidden_states = self.bert(**inputs).hidden_states
        embeddings = self.bert_reducer(torch.stack(hidden_states))

        return self.linear(embeddings)
