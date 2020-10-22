import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from data_utils import flatmap
import numpy as np
from typing import List, Tuple


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


def average_precision_k(actual_count, labels, k):
    """
    Average precision at k candidates.

    precision@k is the number of relevant results among the top k elements divided by k


    Args:
        actual_count
            The number of known relevant candidates for the query that correspond to the labels
        labels:
            relevance labels for top k candidates
    """
    assert len(labels) >= k

    score = 0
    for j in range(1, k + 1):
        if not labels[j]:
            # jth candidate is not relevant
            continue

        # precision@j is the number of relevant results among the top j elements divided by j
        score += sum(labels[:j])/j
    return score/actual_count


def mean_average_precision(actual_count_set: List[List[int]],
                           label_set: List[List[bool]],
                           k):
    """
    MAP@K implementation based on the definition in Gillick et al. 2018
    """
    return sum(
        [average_precision_k(actual_count, labels, k)
         for actual_count, labels in zip(actual_count_set, label_set)]
    )


def top_candidates(encoded_batch_q, encoded_candidates, k):
    """
    For each query, find indices of the top k relevant candidates
    For example, for k = 2 and the top k candidates are in index position 2 and 4
    in encoded_candidates then return [2, 4]
    """
    pass


def search(model: nn.Module, batch_query, candidate_id, candidate_text, k):
    # (batch_size, hidden_size)
    encoded_batch_q = model(batch_query)
    encoded_candidates = model(candidate_text)

    top_k_predict_indices = top_candidates(encoded_batch_q, encoded_candidates, k)
    result = []
    for indices in top_k_predict_indices:
        result.append([candidate_id[i] for i in indices])
    return result


def get_labels(actual, predict):
    """
    Returns:
        List[bool]
    """
    return [p in actual for p in predict]


def evaluate(
    model: nn.Module,
    test_data: List[Tuple[str, str]],
    dataset,
    candidates,
    batch_size,
    k,
    epoch):

    queries = flatmap(test_data)
    candidate_text = [c[1] for c in candidates]
    candidate_id = [c[0] for c in candidates]

    map_score = 0
    for batch_query in iterate_batch(queries, batch_size):
        # for each query, the actual relevant document ids:  [set(1,2,3), set(5), ...]
        actual_id_set = dataset.get_relevant_result(batch_query)
        actual_count_set = [len(ids) for ids in actual_id_set]

        predict_id_set = search(model, batch_query, candidate_id, candidate_text, k)

        label_set = [get_labels(actual, predict) for actual, predict in zip(actual_id_set, predict_id_set)]

        map_score += mean_average_precision(actual_count_set, label_set, k)

    print("Epoch {}: MAP score is {}".format(epoch, map_score))



def fit(epochs,
        model: nn.Module,
        loss_func,
        opt: optim.Optimizer,
        train_data: List[Tuple[str, str]],
        test_data: List[Tuple[str, str]],
        candidates: Tuple[str, str],
        pairwise_similarity_func=pairwise_cosine_similarity,
        top_k=100,
        batch_size = 1000):
    """
    Args:
        model:
            A model for encoding questions

        train_data:
            A list of tuples of positive pair of questions (question_text1, question_text2)

        test_data:
            A list of tuples of positive pair of questions (question_text1, question_text2)

        candidates:
            A list of tuples of (id, text)

        top_k:
            Use top K candidates for computing mean average precision

    """
    for epoch in range(epochs):
        model.train()

        for q1_batch, q2_batch in iterate_batch(train_data, batch_size):
            loss_batch(model, loss_func, q1_batch, q2_batch, pairwise_similarity_func, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb, pairwise_similarity_func)
                  for xb, yb in iterate_batch(test_data, batch_size)]
            )

            evaluate(model, test_data, candidates, top_k, epoch)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

    print("Epoch {}: val_loss={}".format(epoch, val_loss))


if __name__ == "__main__":
    pass
