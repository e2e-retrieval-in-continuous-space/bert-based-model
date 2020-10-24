import unittest
from train_utils import pairwise_cosine_similarity
import torch.nn.functional as F
import torch

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

if __name__ == '__main__':
    """
    python -m unittest tests/test_trail_utils.py
    """
    unittest.main()