from typing import List
import torch
from torch import nn
from transformers import AutoTokenizer


class SimpleEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim, pretrained_tokenizer_name='bert-base-uncased'):
        super(SimpleEmbeddingModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
        self.embeddings = nn.Embedding(self.tokenizer.vocab_size,
                                       embedding_dim,
                                       self.tokenizer.pad_token_id)


    def forward(self, inputs: List[str]):
        """
        Average embedding over words of sentences

        Args:
            inputs:
                a batch of sentences.
        Returns:
            Tensor with shape (batch_size, embedding_dim)
        """
        encoded_input = self.tokenizer(inputs,
                                       add_special_tokens=False,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt")

        # encoded_input.input_ids: (batch_size, max_word_count)
        # embedding: (batch_size, max_word_count, embedding_dim)
        embedding = self.embeddings(encoded_input.input_ids)
        return torch.mean(embedding, dim=1)



if __name__ == "__main__":
    m = SimpleEmbeddingModel(300)
    res = m(["How old are you?", "this is a test"])
    print(res.shape)