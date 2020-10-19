import torch
from typing import List


def compute_random_embeddings(sentences: List[str], length: int = 768) -> torch.Tensor:
    """
    Returns random embeddings of length `length` for each input sentence. Each random
    value is a float between 0 and 1.
    :return: `Tensor` with all the embeddings.
    """
    return torch.rand(len(sentences), length)
