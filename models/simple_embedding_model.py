from typing import List
import torch
from torch import nn
from transformers import AutoTokenizer

from data_utils import flatmap
from loggers import getLogger

logger = getLogger(__name__)


class SimpleEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim, pretrained_tokenizer_name='bert-base-uncased', device=None, examples=None):
        super(SimpleEmbeddingModel, self).__init__()
        self.device = device
        if not self.device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)

        vocab = set()
        for e in examples:
            vocab.update(self.tokenize(e.q1_text))
            vocab.update(self.tokenize(e.q2_text))
        vocab = list(vocab)
        vocab.append("<UNK>")

        self.max_seq_length = max([len(self.tokenize(text)) for text in flatmap([(e.q1_text, e.q2_text) for e in examples])])
        self.word2idx = {word: i for i, word in enumerate(vocab)}

        self.embeddings = nn.Embedding(len(vocab),
                                       embedding_dim,
                                       self.word2idx["<UNK>"]).to(self.device)
    def encode(self, sentences):
        batch_tokens = [self.add_padding(self.tokenize(s)) for s in sentences]
        return [[self.word_to_id(t) for t in tokens] for tokens in batch_tokens]

    def word_to_id(self, word):
        return self.word2idx[word] if word in self.word2idx else self.word2idx["<UNK>"]

    def add_padding(self, s):
        return s + ["<UNK"] * (self.max_seq_length - len(s))

    def tokenize(self, s):
        return s.lower().replace("?", "").split(" ")

    def forward(self, inputs: List[str]):
        """
        Average embedding over words of sentences

        Args:
            inputs:
                a batch of sentences.
        Returns:
            Tensor with shape (batch_size, embedding_dim)
        """

        """
        encoded_input = self.tokenizer(inputs,
                                       add_special_tokens=False,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt")
        """
        encoded_input = torch.LongTensor(self.encode(inputs)).to(self.device)
        embedding = self.embeddings(encoded_input)

        # encoded_input.input_ids: (batch_size, max_word_count)
        # embedding: (batch_size, max_word_count, embedding_dim)
        #embedding = self.embeddings(encoded_input.input_ids.to(self.device))
        return torch.mean(embedding, dim=1)



if __name__ == "__main__":
    m = SimpleEmbeddingModel(300)
    res = m(["How old are you?", "this is a test"])
    print(res.shape)
