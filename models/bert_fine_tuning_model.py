import torch
from torch import nn, Tensor
from typing import List, Callable
from transformers import AutoTokenizer, AutoModel, AutoConfig
from loggers import getLogger
from models.bert_feature_extractor_model import BERTVersion, reducer_all_layers

logger = getLogger(__name__)

class BERTFineTuningModel(nn.Module):
    def __init__(
            self,
            bert_version: BERTVersion,
            hidden_size: int = None,
            bert_reducer: Callable[[Tensor], Tensor] = reducer_all_layers,
            device=None,
            bert_chunk_size=100
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
        self.bert_chunk_size = bert_chunk_size

        self.linear = nn.Linear(self.embeddings_dim, self.hidden_size).to(self.device)

    def forward(self, documents: List[str]) -> Tensor:
        embeddings = self._run_bert(documents)
        return self.linear(embeddings)

    def _run_bert(self, documents):
        """
        Args:
            documents:
                List of sentences
        Returns:
            Tensor with shape (batch_size, num_layers, max_sentence_len, embedding_size)
        """
        inputs = self.tokenizer(documents, return_tensors="pt", padding=True, truncation=True).to(self.device)
        hidden_states = self.bert(**inputs).hidden_states
        stacked = torch.stack([s.to('cpu') for s in hidden_states]).transpose(0, 1)
        coords = (inputs.input_ids == 0).nonzero(as_tuple=False)  # irrelevant_tokens_coords
        stacked[coords[:, 0], :, coords[:, 1], :] = 0
        return stacked


if __name__ == "__main__":
    import doctest

    doctest.testmod()
