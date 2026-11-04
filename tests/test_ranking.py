import unittest
import numpy as np
import torch
from wsprec.cli import FM
from wsprec.ranking import evaluate_ranking


class RankingTests(unittest.TestCase):
    def model(self):
        m = FM(2, 4, 2)
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
            m.item_bias.weight[:, 0] = torch.tensor([10., 3., 2., 1.])
        return m

    def test_masking_and_multi_positive_ndcg(self):
        rows = np.array([[0, 2, 5, 0], [0, 3, 4, 0], [1, 1, 2, 0]])
        seen = np.array([[0, 0, 1, 0]])
        r = evaluate_ranking(self.model(), rows, seen, (1, 2, 3))
        self.assertEqual(r['ranking_users'], 1)
        self.assertEqual(r['recall@1'], 0)
        self.assertEqual(r['recall@2'], .5)
        self.assertEqual(r['recall@3'], 1)
        self.assertEqual(r['hit_rate@2'], 1)
        self.assertAlmostEqual(r['ndcg@2'], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))

    def test_empty_positive_set(self):
        r = evaluate_ranking(self.model(), np.array([[0, 1, 2, 0]]), [], (10,))
        self.assertEqual(r['ranking_users'], 0)
        self.assertIsNone(r['ndcg@10'])

    def test_ties_and_large_k(self):
        m = self.model()
        with torch.no_grad():
            m.item_bias.weight.zero_()
        r = evaluate_ranking(m, np.array([[0, 0, 5, 0]]), [], (1, 10))
        self.assertEqual(r['ndcg@1'], 1)
        self.assertEqual(r['recall@10'], 1)
