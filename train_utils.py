from math import ceil

import torch
import tensorflow as tf
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils import flatmap, Question
import numpy as np
from typing import List, Tuple, Generator, Dict
from loggers import getLogger
from datetime import datetime
import os.path
import time
import json

from quora_dataset import QuoraDataset, RetrievalDataset

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
    normalize_a = tf.nn.l2_normalize(q1_batch_embedding, 1)
    normalize_b = tf.nn.l2_normalize(q2_batch_embedding, 1)
    distance = 1 - tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return distance


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
    neg_log_softmax = tf.math.negative(tf.nn.log_softmax(similarity_matrix, axis=1))
    loss_terms = tf.linalg.diag(neg_log_softmax, 0)
    return tf.math.reduce_mean(loss_terms)


def loss_batch(model: nn.Module, loss_func, q1_batch, q2_batch, similarity_func, minimize=None):
    q1_batch_text = [q.text for q in q1_batch]
    q2_batch_text = [q.text for q in q2_batch]
    q1_batch_encoded = model(q1_batch_text)
    q2_batch_encoded = model(q2_batch_text)
    loss = loss_func(q1_batch_encoded, q2_batch_encoded, similarity_func)
    if minimize is not None:
        minimize(loss)

    return loss.numpy()[0], len(q1_batch)


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


def average_precision_k_for_batch(actual_count_set: List[List[int]],
                                      label_set: List[List[bool]],
                                      k):
    """
    Sum of the average precision at k for the given query batch
    """
    return [average_precision_k(actual_count, labels, k)
         for actual_count, labels in zip(actual_count_set, label_set)]


def top_candidates(encoded_batch_q, encoded_candidates, k, pairwise_similarity_func):
    """
    For each query, find indices of the top k relevant candidates measured by the pairwise_similarity_func.

    Args:
        encoded_batch_q:
            Encoded batch of queries.  Shape (batch_size, hidden_size)
        encoded_candidates:
            Encoded candidates.  Shape (candidate_size, hidden_size)

    Returns:
        topk:
            (batch_size, k) - A list of lists of indices of top k relevant candidates as measured by the similarity function
            [ [idx1, idx2, ...], [idx9, idx10, ...], ... ]
        topk_similarity:
            (batch_size, k) - Similiar structure as topk but with similarity score as values
    """
    # Shape: (batch_size, candidate_size)
    similarity = pairwise_similarity_func(encoded_batch_q, encoded_candidates)
    # (batch_size, k)
    topk = tf.math.top_k(similarity, k=k).indices.tolist()

    # TODO: cleanup needed.  This is needed when training on GPU to make sure
    # select_index would be the same devie as similarity tensor.
    # Otherwise, torch will throw exception
    with tf.device('/CPU:0'):
        select_index = tf.Tensor(topk)
    topk_similarity = tf.gather(similarity, 1, select_index)
    return topk, topk_similarity


def search(model: nn.Module, batch_query_text, candidate_id, candidate_text, k, pairwise_similarity_func):
    # (batch_size, hidden_size)
    encoded_batch_q = model(batch_query_text)
    encoded_candidates = model(candidate_text)

    top_k_predict_indices, top_k_similarity = top_candidates(encoded_batch_q, encoded_candidates, k, pairwise_similarity_func)
    result = []
    for indices in top_k_predict_indices:
        result.append([candidate_id[i] for i in indices])
    return result, top_k_similarity


def get_labels(actual, predict):
    """
    Returns:
        List[bool]
    """
    return [p in actual for p in predict]


def collate_fn(batch):
    """
    Given batch of [(q1, q2), (q1,q2), ... ] tuples, yields pairs of mini-batches
    like [(q1, q1, ...), (q2, q2, ...)].

    Args:
        batch
            List of 2-tuples of (Question, Question) or (Query, ResultSet)
    """
    # batch is list of tuples (Question, Question) or (Query, Results)
    return tuple(zip(*batch))


def evaluate(
        model: nn.Module,
        retrieval_data: RetrievalDataset,
        candidate_ids: List[str],
        qid2text: Dict[str, str],
        retrieval_batch_size,
        k,
        epoch,
        pairwise_similarity_func,
        save_model_dir=None):
    """
    Returns:
        MAP@K for the given test_data
    """
    data_loader = DataLoader(retrieval_data, batch_size=retrieval_batch_size, collate_fn=collate_fn)
    candidate_text = [qid2text[qid] for qid in candidate_ids]
    candidate_set = set(candidate_ids)

    map_score = 0
    eval_records = []
    for query_ids, actual_id_set in data_loader:
        # query_ids is a list of qids.  e.g. [qid1, qid2, ...]
        # actual_id_set is a list of sets of relevant for the query_ids
        actual_count_set = [len(ids) for ids in actual_id_set]

        for qid in flatmap(actual_id_set):
            assert qid in candidate_set, "qid {} does not exist in candidate set".format(qid)

        batch_query_text = [qid2text[qid] for qid in query_ids]
        predict_id_set, similarity_set = search(model, batch_query_text, candidate_ids, candidate_text, k + 1, pairwise_similarity_func)
        for i, predictions in enumerate(predict_id_set):
            predict_id_set[i] = [pid for pid in predictions if pid != query_ids[i]][:k]

        label_set = [get_labels(actual, predict) for actual, predict in zip(actual_id_set, predict_id_set)]

        recalls = [sum(labels)/actual_count for labels, actual_count in zip(label_set, actual_count_set)]

        avg_precision_k = average_precision_k_for_batch(actual_count_set, label_set, k)
        map_score += sum(avg_precision_k)

        records = create_eval_records(epoch, query_ids, actual_id_set, predict_id_set, similarity_set, label_set, recalls, avg_precision_k)
        eval_records.extend(records)

    if save_model_dir:
        file_name = "{}.epoch{}.eval.{}.json".format(model.__class__.__name__,
                                                     epoch,
                                                  datetime.now().strftime("%Y-%m-%d"))
        full_file_name = os.path.join(save_model_dir, file_name)
        logger.info("Saving evaluation result to %s", full_file_name)
        with open(full_file_name, "w") as f:
            f.write(json.dumps(eval_records))

    return map_score / len(retrieval_data)


def create_eval_records(epoch, query_ids, actual_id_set, predict_id_set, similarity_set, label_set, recalls, avg_precision_k):
    n = len(query_ids)
    result = []
    # replace sets with lists to avoid JSON serialization error
    actual_id_set = [list(ids) for ids in actual_id_set]
    for i in range(n):
        result.append({
            "epoch": epoch,
            "query_id": query_ids[i],
            "actual_id": actual_id_set[i],
            "predict_id": predict_id_set[i],
            "similarity": similarity_set[i].tolist(),
            "predict_labels": label_set[i],
            "recalls": recalls[i],
            "avg_precision_k": avg_precision_k[i]
        })
    return result


class DummyWriter():
    def __init__(self): pass
    def add_scalar(self, *args, **kwargs): pass
    def flush(self): pass
    def close(self): pass


def fit(epochs,
        model: nn.Module,
        opt: optim.Optimizer,
        train_data: QuoraDataset,
        test_data: QuoraDataset,
        retrieval_data: RetrievalDataset,
        candidate_ids: List[str],
        qid2text: Dict[str, str],
        loss_func=in_batch_sampled_softmax,
        pairwise_similarity_func=pairwise_cosine_similarity,
        top_k=100,
        batch_size=1000,
        retrieval_batch_size=1000,
        save_model_dir=None):
    """
    Args:
        model:
            A model for encoding questions

        train_data:
            A QuoraDataset of positive examples for training

        test_data:
            A QuoraDataset of positive examples for testing

        retrieval_data:
            A RetrievalDataset of query to relevant result for evaluation using MAP@K

        candidate_ids:
            A list of qids

        top_k:
            Use top K candidates for computing mean average precision

    """
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
    writer = SummaryWriter(save_model_dir, flush_secs=30) if save_model_dir else DummyWriter()

    for epoch in range(epochs):
        logger.debug("Running train...")
        model.train()

        logger.debug("Running loss_batch, %s...", {"batch_size": batch_size, "batches": ceil(len(train_data)/batch_size)})
        for i, (q1_batch, q2_batch) in enumerate(train_loader):
            start_time = time.perf_counter()
            loss_val, batch_count = loss_batch(model, loss_func, q1_batch, q2_batch, pairwise_similarity_func, opt)
            duration = time.perf_counter() - start_time
            logger.debug("Finished loss_batch %s at %.2f example/second, loss=%f", i, batch_count/duration, loss_val)
            writer.add_scalar("Loss/train", loss_val, epoch * len(train_loader) + i)

        logger.debug("Running model.eval()...")
        model.eval()
        with torch.no_grad():
            logger.debug("Computing losses per batch on evaluation data...")
            losses = []
            nums = []
            for i, (xb, yb) in enumerate(test_loader):
                loss_val, batch_count = loss_batch(model, loss_func, xb, yb, pairwise_similarity_func)
                losses.append(loss_val)
                nums.append(batch_count)
                writer.add_scalar("Loss/test", loss_val, epoch * len(test_loader) + i)

            logger.debug("Evaluating...")
            map_score = evaluate(model,
                                 retrieval_data,
                                 candidate_ids,
                                 qid2text,
                                 retrieval_batch_size,
                                 top_k,
                                 epoch,
                                 pairwise_similarity_func,
                                 save_model_dir)
            logger.info("Epoch {}: test data MAP score is {}".format(epoch, map_score))

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        logger.info("Epoch %d: val_loss=%f", epoch, val_loss)

    if save_model_dir:
        writer.flush()
        writer.close()

        file_name = "{}.{}.state_dict".format(model.__class__.__name__, datetime.now().strftime("%Y-%m-%d"))
        full_file_name = os.path.join(save_model_dir, file_name)
        logger.info("Saving model to %s", full_file_name)
        torch.save(model.state_dict(), full_file_name)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
