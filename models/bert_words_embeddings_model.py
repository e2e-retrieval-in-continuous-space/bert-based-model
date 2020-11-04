import enum
import torch
from torch import nn, Tensor
from typing import List
from transformers import AutoTokenizer, AutoModel, AutoConfig
from loggers import getLogger
from models.utils import DictCache

logger = getLogger(__name__)


class BERTVersion(enum.Enum):
    """
    Enum representing supported BERT versions
    """
    BASE_UNCASED = 'bert-base-uncased'


def average_axis(axis: int, tensor: Tensor) -> Tensor:
    return tensor.sum(axis) / tensor.shape[axis]


class BERTWordsEmbeddingsModel(nn.Module):
    def __init__(
            self,
            bert_version: BERTVersion,
            hidden_size: int = None,
            device=None
    ):
        super().__init__()
        self.device = device
        if not self.device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.bert_version = bert_version.value
        self.config = AutoConfig.from_pretrained(self.bert_version, output_hidden_states=True, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_version, config=self.config)
        self.bert = None
        self.embeddings_dim = self.config.hidden_size
        self.hidden_size = hidden_size or self.embeddings_dim * 2

        self.computed_tokens_embeddings = None

        self.cache = DictCache('{0}-tokens'.format(bert_version.value), filename='embeddings')
        precomputed = list(self.cache.get('embeddings', {0: []}).values())

        self.embeddings = nn.Embedding.from_pretrained(
            Tensor(precomputed).to(self.device),
            False,
            self.tokenizer.pad_token_id).to(self.device)

    def forward(self, inputs: List[str]) -> Tensor:
        encoded_input = self.tokenizer(inputs,
                                       add_special_tokens=False,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt")

        # encoded_input.input_ids: (batch_size, max_word_count)
        # embedding: (batch_size, max_word_count, embedding_dim)
        embedding = self.embeddings(encoded_input.input_ids.to(self.device))
        return torch.mean(embedding, dim=1)

    def compute_word_embeddings_batch(self, documents: List[str]):
        if self.bert is None:
            self.bert = AutoModel.from_pretrained(self.bert_version, config=self.config)
            self.bert.to(self.device)

        if self.computed_tokens_embeddings is None:
            self.computed_tokens_embeddings = {i: [0, torch.zeros(768)] for i in range(self.tokenizer.vocab_size)}

        with torch.no_grad():
            logger.debug("Computing BERT embeddings for %d documents", len(documents))

            inputs = self.tokenizer(documents, return_tensors="pt", padding=True, truncation=True).to(self.device)
            hidden_states = self.bert(**inputs).hidden_states

            last_4_layers = torch.stack([s.to('cpu') for s in hidden_states[:-4]]).transpose(0, 1)
            tokens_avg_across_layers = average_axis(1, last_4_layers)

            for i, doc_tokens in enumerate(inputs.input_ids):
                for j, token in enumerate(doc_tokens):
                    token_id = token.item()
                    self.computed_tokens_embeddings[token_id][0] += 1
                    self.computed_tokens_embeddings[token_id][1] += tokens_avg_across_layers[i][j]

    def tokens_embeddings_to_vocab_embeddings(self):
        self.cache.clear()
        retval = {k: ((v[1] / v[0]).tolist() if v[0] else [0]*768) for (k, v) in self.computed_tokens_embeddings.items()}
        self.cache['embeddings'] = retval
        self.cache.save()
        return retval
