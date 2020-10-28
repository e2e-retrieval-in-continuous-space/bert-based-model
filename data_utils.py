from collections import defaultdict, namedtuple
from itertools import chain
import random
from typing import List
import numpy as np

Example = namedtuple("Example",
                     ["id", "qid1", "qid2", "q1_text", "q2_text", "is_duplicate"])

Question = namedtuple("Question",
                      ["qid", "text"])

def generate_data(examples: List[Example], reachable):
    """
    Use the transitive closure of the graph to generate all relevant
    pairs of qids.

    Args:
        examples:
            A list of positive examples
        reachable:
            A dict of qid to its reachable nodes

    Returns:
        A set of tuples of (qid1, qid2) which are relevant
    """
    result = []
    for e in examples:
        for neighbor in reachable[e.qid1]:
            result.append((e.qid1, neighbor))
        for neighbor in reachable[e.qid2]:
            result.append((e.qid2, neighbor))
    return set(result)


def build_graph(examples: List[Example]):
    """
    Build an adjacent lists based on the examples.  Each node
    is represented by the qid of a question.

    Args:
        examples: A list of Examples to build adjacent lists on.

    Returns:
        Returns a dict of qid to its adjacent nodes
    """
    # graph is an adjacency list
    graph = defaultdict(list)
    for ex in examples:
        graph[ex.qid1].append(ex.qid2)
        graph[ex.qid2].append(ex.qid1)
    return graph


def split_examples(examples: List[Example], split_fracs: List[float], seed=None):
    """
    Shuffle the examples and then split the examples according to split fractions.

    Args:
        examples:
            A list of Examples to split
        split_fracs:
            A list of fractions denoting proportions that must add up to 1
        seed:
            Seed for random shuffling

    Returns:
        A list of lists of Examples that correspond to the split fractions.
    """

    if seed:
        random.seed(seed)
    random.shuffle(examples)

    split_points = [0] + [int(round(frac * len(examples)))
                          for frac in np.cumsum(split_fracs)]

    return [examples[split_points[i]:split_points[i + 1]]
            for i in range(len(split_fracs))]


def build_transitive_closure(graph):
    """
    Build a transitive closure based on adjacent lists

    Args:
        graph:
            A dict of qid to its adjacent nodes in the graph

    Returns:
        A dict of qid to a set of nodes reachable from it
    """
    reachable = defaultdict(set)
    for u in graph.keys():
        dfs(graph, u, u, reachable)
    return reachable


def dfs(graph, u, v, reachable):
    """
    Depth-first search traversal of graph to build the transitive
    closure

    Args:
        u:
            Current node
        v:
            A node reachable from u
        reachable:
            A dict of qid to a set of nodes reachable from it

    """
    if u != v:
        # exclude itself
        reachable[u].add(v)
    for w in graph[v]:
        if w not in reachable[u]:
            dfs(graph, u, w, reachable)


def flatmap(iterable):
    """
    Short-hand for [(a,b), (c, d, e)] -> [a, b, c, d, e]

    Args:
        iterable:
            An object that is iterable
    Returns:
        A list that is flattened by one-level
    """
    return list(chain.from_iterable(iterable))


def chunks(data, n):
    for i in range(0, len(data), n):
        yield data[i:i + n]
