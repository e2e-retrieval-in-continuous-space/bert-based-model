from typing import List
import torch
from torch import nn
from loggers import getLogger
from models.utils import BagOfWordsTokenizer

logger = getLogger(__name__)


class BagOfWordsEmbeddingModel(nn.Module):

    def __init__(self, embedding_dim, train_data, test_data, qid2text, device=None):
        super(BagOfWordsEmbeddingModel, self).__init__()
        self.device = device
        if not self.device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BagOfWordsTokenizer(
            train_data,
            test_data,
            qid2text
        )
        self.embeddings = nn.Embedding(self.tokenizer.vocab_size,
                                       embedding_dim,
                                       self.tokenizer.pad_token_id).to(self.device)


    def forward(self, inputs: List[str]):
        """
        Average embedding over words of sentences

        Args:
            inputs:
                a batch of sentences.
        Returns:
            Tensor with shape (batch_size, embedding_dim)
        """
        tokens = self.tokenizer(inputs)
        embedding = self.embeddings(tokens.to(self.device))
        return torch.mean(embedding, dim=1)

