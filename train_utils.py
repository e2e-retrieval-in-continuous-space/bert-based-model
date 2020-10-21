import torch
from torch import nn
from torch import optim
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F


def pairwise_cosine_similarity(q1_batch_embedding, q2_batch_embedding):
    """
    Squared Euclidean is proportional to the cosine distance.
    TODO:  Perhaps we can do away the squaring if that doesn't change the ranking of docs.

    https://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance/146279#146279
    https://en.wikipedia.org/wiki/Cosine_similarity#Properties

    Args:
         q1_batch_embedding:
            Shape (batch_size, embedding_size)
         q2_batch_embedding:
            Shape (batch_size, embedding_size)
    Returns:
         all pairs simliarity matrix (batch_size, batch_size)
    """
    q1_norm = F.normalize(q1_batch_embedding, dim=1, p=2)
    q2_norm = F.normalize(q2_batch_embedding, dim=1, p=2)
    pairiwse_l2_dist = torch.cdist(q1_norm, q2_norm, p=2)
    return torch.square(pairiwse_l2_dist)


def in_batch_sampled_softmax(q1_batch_embedding, q2_batch_embedding, pairwise_similarity_func):
    """
    Apply similarity_function to all pairs of questions between q1_batch_embedding and q2_batch_embedding
    to form similarity matrix M where the diagonal contains positive examples and the off-diagonal contains
    random negative examples.

    Then calculate softmax loss term for each row of M and then average them.

    Args:
         q1_batch_embedding:
            Shape (batch_size, embedding_size)
         q2_batch_embedding:
            Shape (batch_size, embedding_size)
    """
    similarity_matrix = pairwise_similarity_func(q1_batch_embedding, q2_batch_embedding)
    neg_log_softmax = torch.neg((F.log_softmax(similarity_matrix, dim=1)))
    loss_terms = torch.diagonal(neg_log_softmax, 0)
    return np.mean(loss_terms)


def loss_batch(model: nn.Module, loss_func, q1_batch, q2_batch, similarity_func, opt=None):
    loss = loss_func(model(q1_batch), model(q2_batch), similarity_func)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(q1_batch)


def iterate_batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]


def fit(epochs,
        model: nn.Module,
        loss_func,
        opt: optim.Optimizer,
        train_data: List[Tuple[str, str]],
        val_data: List[Tuple[str, str]],
        pairwise_similarity_func=pairwise_cosine_similarity,
        batch_size = 1000):
    """
    Args:
        model:
            A model for encoding questions

        train_data:
            A list of tuples of positive pair of questions (question_text1, question_text2)

        val_data:
            A list of tuples of positive pair of questions (question_text1, question_text2)
    """
    for epoch in range(epochs):
        model.train()

        for q1_batch, q2_batch in iterate_batch(train_data, batch_size):
            loss_batch(model, loss_func, q1_batch, q2_batch, pairwise_similarity_func, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb, pairwise_similarity_func)
                  for xb, yb in iterate_batch(val_data, batch_size)]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

    print("Epoch {}: val_loss={}".format(epoch, val_loss))


if __name__ == "__main__":
    pass

