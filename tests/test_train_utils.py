import unittest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import cosine_similarity
from train_utils import pairwise_cosine_similarity, top_candidates


class TestTrainUtils(unittest.TestCase):

    def test_pairwise_similarity(self):
        a = torch.randn(3, 4)
        a = F.normalize(a, dim=1, p=2)

        b = torch.randn(3, 4)
        b = F.normalize(b, dim=1, p=2)

        batch_size, _ = a.shape
        expected = [[0] * batch_size for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(batch_size):
                expected[i][j] = torch.dist(a[i], b[j])

        actual = pairwise_cosine_similarity(a, b)
        expected = torch.square(torch.Tensor(expected))

        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected))

    def test_top_candidates(self):
        query = Tensor([[1, 1]])
        candidates = Tensor([[0, 2], [3, 3], [0, 0], [-3, -3]])
        actual = top_candidates(query, candidates, 2, pairwise_cosine_similarity)[0]
        expected = torch.topk(cosine_similarity(query, candidates), 2, largest=True).indices.tolist()[:2]

        self.assertListEqual(actual, expected)


if __name__ == '__main__':
    """
    python -m unittest tests/test_train_utils.py
    """
    unittest.main()
