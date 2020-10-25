import unittest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import cosine_similarity
from train_utils import pairwise_cosine_similarity, top_candidates


class TestTrainUtils(unittest.TestCase):

    def test_pairwise_similarity(self):
        a = torch.randn(3, 4)
        b = torch.randn(3, 4)

        batch_size, _ = a.shape
        expected = [F.cosine_similarity(a[i].unsqueeze(0), b) for i in range(batch_size)]
        expected = torch.stack(expected)
        actual = pairwise_cosine_similarity(a, b)

        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected))

    def test_top_candidates(self):
        k = 2
        query = Tensor([[1, 1], [-2, 4]])
        candidates = Tensor([[0, 2], [3, 3], [1, 0.5], [-3, -3]])
        actual = top_candidates(query, candidates, k, pairwise_cosine_similarity)

        batch_size = query.shape[0]
        expected = [cosine_similarity(query[i].unsqueeze(0), candidates) for i in range(batch_size)]
        expected_similarity = torch.stack(expected)
        expected = torch.topk(expected_similarity, k, largest=True).indices.tolist()

        self.assertListEqual(actual, expected)


if __name__ == '__main__':
    """
    python -m unittest tests/test_train_utils.py
    """
    unittest.main()
