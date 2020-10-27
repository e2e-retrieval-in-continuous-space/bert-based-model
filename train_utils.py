from math import ceil

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from data_utils import flatmap
import numpy as np
from typing import List, Tuple, Generator
from loggers import getLogger
from datetime import datetime
import os.path
import time
logger = getLogger(__name__)

def pairwise_cosine_similarity(q1_batch_embedding, q2_batch_embedding):
    """
    Squared Euclidean is proportional to the cosine distance.

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
    squared = torch.square(pairiwse_l2_dist)
    return 1 - 0.5 * squared

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

    >>> from torch import Tensor
    >>> similarity = lambda a, b: Tensor([[1, 0.5], [0.5, 1]])
    >>> in_batch_sampled_softmax([(1, 2),(3, 4)], [(1, 2),(3, 4)], similarity)
    tensor(0.4741)
    """
    similarity_matrix = pairwise_similarity_func(q1_batch_embedding, q2_batch_embedding)
    neg_log_softmax = torch.neg((F.log_softmax(similarity_matrix, dim=1)))
    loss_terms = torch.diagonal(neg_log_softmax, 0)
    return torch.mean(loss_terms)


def loss_batch(model: nn.Module, loss_func, q1_batch, q2_batch, similarity_func, opt=None):
    q1_batch_text = [q.text for q in q1_batch]
    q2_batch_text = [q.text for q in q2_batch]
    q1_batch_encoded = model(q1_batch_text)
    q2_batch_encoded = model(q2_batch_text)
    loss = loss_func(q1_batch_encoded, q2_batch_encoded, similarity_func)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(q1_batch)


def iterate_batch(data: List[Tuple[str, str]], batch_size: int) -> Generator[Tuple[List[str], List[str]], None, None]:
    """
    Given batch of [(q1, q2), (q1,q2), ... ] tuples, yields pairs of mini-batches
    like [(q1, q1, ...), (q2, q2, ...)].

    Args:
        data
            List of two-tuples of duplicate questions.
        batch_size:
            Size of yielded mini-batches.

    >>> list(iterate_batch([("a", "b"), ("a", "b")], 1))
    [[('a',), ('b',)], [('a',), ('b',)]]
    >>> list(iterate_batch([("a", "b"), ("a", "b")], 2))
    [[('a', 'a'), ('b', 'b')]]
    """
    for i in range(0, len(data), batch_size):
        yield list(zip(*data[i:i + batch_size]))


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
    for j in range(k):
        if not labels[j]:
            # jth candidate is not relevant
            continue

        # precision@j is the number of relevant results among the top j elements divided by j
        score += sum(labels[:j+1]) / (j+1)
    return score / actual_count


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


def top_candidates(encoded_batch_q, encoded_candidates, k, pairwise_similarity_func):
    """
    For each query, find indices of the top k relevant candidates measured by the pairwise_similarity_func.

    Args:
        encoded_batch_q:
            Encoded batch of queries.  Shape (batch_size, hidden_size)
        encoded_candidates:
            Encoded candidates.  Shape (candidate_size, hidden_size)

    Returns:
            A list of lists of indices of top k relevant candidates as measured by the similarity function
            [ [idx1, idx2, ...], [idx9, idx10, ...], ... ]
    """
    # Shape: (batch_size, candidate_size)
    similarity = pairwise_similarity_func(encoded_batch_q, encoded_candidates)
    return torch.topk(similarity, k).indices.tolist()


def search(model: nn.Module, batch_query, candidate_id, candidate_text, k, pairwise_similarity_func):
    # (batch_size, hidden_size)
    batch_query_text = [q.text for q in batch_query]
    encoded_batch_q = model(batch_query_text)
    encoded_candidates = model(candidate_text)

    top_k_predict_indices = top_candidates(encoded_batch_q, encoded_candidates, k, pairwise_similarity_func)
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
        epoch,
        pairwise_similarity_func):
    """
    Returns:
        MAP@K for the given test_data
    """
    queries = list(set(flatmap(test_data)))
    candidate_text = [c[1] for c in candidates]
    candidate_id = [c[0] for c in candidates]
    query_batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]

    map_score = 0
    for batch_query in query_batches:
        # for each query, the actual relevant document ids:  [set(1,2,3), set(5), ...]
        query_ids = [q.qid for q in batch_query]
        actual_id_set = dataset.get_relevant_result(query_ids)
        actual_count_set = [len(ids) for ids in actual_id_set]

        predict_id_set = search(model, batch_query, candidate_id, candidate_text, k, pairwise_similarity_func)

        label_set = [get_labels(actual, predict) for actual, predict in zip(actual_id_set, predict_id_set)]

        map_score += mean_average_precision(actual_count_set, label_set, k)
        #logger.debug("Epoch {}: MAP score is {}".format(epoch, map_score))

    return map_score


def fit(epochs,
        model: nn.Module,
        opt: optim.Optimizer,
        train_data: List[Tuple[str, str, str, str]],
        test_data: List[Tuple[str, str, str, str]],
        dataset,
        candidates: List[Tuple[str, str]],
        loss_func=in_batch_sampled_softmax,
        pairwise_similarity_func=pairwise_cosine_similarity,
        top_k=100,
        batch_size=1000,
        save_model_dir=None):
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
        logger.debug("Running train...")
        model.train()

        logger.debug("Running loss_batch, %s...", {"batch_size": batch_size, "batches": ceil(len(train_data)/batch_size)})
        for i, (q1_batch, q2_batch) in enumerate(iterate_batch(train_data, batch_size)):
            start_time = time.perf_counter()
            loss_val, batch_count = loss_batch(model, loss_func, q1_batch, q2_batch, pairwise_similarity_func, opt)
            duration = time.perf_counter() - start_time

            logger.debug("Finished loss_batch %s at %.2f example/second, loss=%f", i, batch_count/duration, loss_val)

        logger.debug("Running model.eval()...")
        model.eval()
        with torch.no_grad():
            logger.debug("Computing losses per batch on evaluation data...")
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb, pairwise_similarity_func)
                  for xb, yb in iterate_batch(test_data, batch_size)]
            )

            logger.debug("Evaluating...")
            map_score = evaluate(model, train_data, dataset, candidates, batch_size, top_k, epoch, pairwise_similarity_func)
            logger.info("Epoch {}: train data MAP score is {}".format(epoch, map_score))

            map_score = evaluate(model, test_data, dataset, candidates, batch_size, top_k, epoch, pairwise_similarity_func)
            logger.info("Epoch {}: test data MAP score is {}".format(epoch, map_score))


        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        logger.info("Epoch %d: val_loss=%f", epoch, val_loss)

    if save_model_dir:
        file_name = "{}.{}.state_dict".format(model.__class__.__name__, datetime.now().strftime("%Y-%m-%d"))
        full_file_name = os.path.join(save_model_dir, file_name)
        logger.info("Saving model to %s", full_file_name)
        torch.save(model.state_dict(), full_file_name)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
