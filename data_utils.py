from collections import defaultdict, namedtuple
from itertools import chain
import random
from typing import List, Tuple, Set
import numpy as np

Example = namedtuple("Example",
                     ["id", "qid1", "qid2", "q1_text", "q2_text", "is_duplicate"])

Question = namedtuple("Question",
                      ["qid", "text"])


class UnionFind:
    def __init__(self, all_qids: Set[str]):
        self.rank = {qid: 1 for qid in all_qids}
        self.parent = {qid: qid for qid in all_qids}

    def find(self, qid):
        if self.parent[qid] != qid:
            self.parent[qid] = self.find(self.parent[qid])
        return self.parent[qid]

    def union(self, qid1, qid2):
        # find set representative for qid1
        p1 = self.find(qid1)
        # find set representative for qid2
        p2 = self.find(qid2)

        # qid1 and qid2 are already in the same set
        if p1 == p2:
            return

        if self.rank[p1] < self.rank[p2]:
            self.parent[p1] = p2
        elif self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
        else:
            self.parent[p2] = p1
            self.rank[p1] += 1

def generate_all_examples(qid_examples: List[Tuple[str, str]]):
    """
    Use the transitive closure of the graph to generate all
    positive examples

    Args:
        qid_examples:
            A list of positive qid pairs (qid1, qid2)

    Returns:
        A set of (qid1, qid2) and the transitive closure graph
    """
    # Build adjacency list
    graph = build_graph(qid_examples)

    # Build reachability map
    reachable = build_transitive_closure(graph)

    result = []
    for qid1, qid2 in qid_examples:
        for neighbor in reachable[qid1]:
            result.append((qid1, neighbor))

        for neighbor in reachable[qid2]:
            result.append((qid2, neighbor))
    return set(result), reachable


def build_graph(qid_examples: List[Tuple[str, str]]):
    """
    Build an adjacent lists based on the examples.  Each node
    is represented by the qid of a question.

    Args:
        qid_examples: A list of (qid1, qid2) to build adjacent lists on.

    Returns:
        Returns a dict of qid to a list adjacent nodes
    """
    # graph is an adjacency list
    graph = defaultdict(list)
    for qid1, qid2 in qid_examples:
        graph[qid1].append(qid2)
        graph[qid2].append(qid1)
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
