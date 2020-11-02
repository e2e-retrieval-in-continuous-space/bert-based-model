import numpy as np
from pathlib import Path
from zipfile import ZipFile

import torch
from torch import nn, Tensor
from typing import List
from transformers import cached_path
from loggers import getLogger
from models.utils import BagOfWordsTokenizer

logger = getLogger(__name__)


class GloveEmbeddingsModel(nn.Module):
    def __init__(
            self,
            train_data,
            test_data,
            qid2text,
            hidden_size: int = None,
            device=None,
            glove_base_filename=None
    ):
        super().__init__()
        self.device = device
        if not self.device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if glove_base_filename is None:
            zip_path = cached_path("http://nlp.stanford.edu/data/glove.6B.zip")
            glove_base_filename = Path(zip_path).parent / 'glove.6B.300d.txt'
            if not glove_base_filename.exists():
                with ZipFile(zip_path, 'r') as zipObj:
                    zipObj.extractall(Path(zip_path).parent)
            if not glove_base_filename.exists():
                raise Exception('Glove file not found')
        self.glove_base_filename = glove_base_filename

        self.tokenizer = BagOfWordsTokenizer(
            train_data,
            test_data,
            qid2text
        )
        embedding_matrix = self.create_glove_embedding_matrix(300, glove_base_filename,
                                                              self.tokenizer.vocab,
                                                              self.tokenizer.vocab_size)
        self.embeddings_dim = embedding_matrix.shape[1]
        self.hidden_size = hidden_size or self.embeddings_dim * 2

        self.embeddings = nn.Embedding.from_pretrained(
            Tensor(embedding_matrix).to(self.device),
            padding_idx=self.tokenizer.pad_token_id
        ).to(self.device)
        self.linear = nn.Linear(self.embeddings_dim, self.hidden_size).to(self.device)

    def forward(self, documents: List[str]) -> Tensor:
        tokens = self.tokenizer(documents).to(self.device)
        embeddings = self.embeddings(tokens)
        return torch.mean(self.linear(embeddings), dim=1)

    def create_glove_embedding_matrix(self, embeddings_dim, embeddings_file_name, word_to_index, max_idx, sep=' '):
        matrix = np.zeros((max_idx + 1, embeddings_dim))
        with open(embeddings_file_name) as infile:
            for idx, line in enumerate(infile):
                word, *rest = line.split(sep)

                if word not in word_to_index:
                    continue

                word_idx = word_to_index[word]
                if word_idx <= max_idx:
                    matrix[word_idx] = np.asarray(rest, dtype='float32')

        return matrix


if __name__ == "__main__":
    import doctest

    doctest.testmod()
