import unittest
from unittest.mock import patch
import numpy as np
import torch
from wsprec.cli import FM
from wsprec.study import scores, quick_metric, apply_recipe
from wsprec.study_metrics import rank_metrics


class StudyTests(unittest.TestCase):
    def test_vectorized_ranking_matches_forward_and_detailed_metrics(self):
        m = FM(2, 4, 2)
        with torch.no_grad():
            s = scores(m.state_dict())
            for u in range(2):
                expected = m(torch.full((4,), u), torch.arange(4)).numpy()
                # Ranking omits a user-constant term only.
                np.testing.assert_allclose(s[u]-s[u, 0], expected-expected[0], atol=3e-7)
        train = np.array([[0, 0, 4, 0], [1, 1, 3, 0]])
        valid = np.array([[0, 2, 5, 1], [0, 1, 4, 1], [1, 3, 5, 1]])
        self.assertAlmostEqual(quick_metric(s, valid, train, 2), rank_metrics(s, valid, train, train, 2)['summary']['ndcg@2'])

    @patch('wsprec.study.counts', return_value=(np.array([2, 5]), np.array([0, 1, 2, 3])))
    def test_recipe_relative_edits_do_not_mutate_original_and_enforce_bounds(self, _):
        base = FM(2, 4, 2).state_dict()
        original = base['item_bias.weight'].clone()
        recipe = {'ops': [{'kind': 'popularity_shift', 'coefficient': .1}]}
        state = apply_recipe(base, recipe)
        torch.testing.assert_close(base['item_bias.weight'], original)
        self.assertTrue((state['item_bias.weight'] != original).any())
        for bad in ({'kind':'scale','parameter':'item.weight','factor':4},
                    {'kind':'scale','parameter':'user_bias.weight','factor':.5},
                    {'kind':'norm_shrink','parameter':'item.weight','target':-1,'strength':1},
                    {'kind':'support_shrink','parameter':'item.weight','tau':float('inf')},
                    {'kind':'neighbor_smooth','parameter':'item.weight','source':'text','strength':.2}):
            with self.assertRaises(ValueError):
                apply_recipe(base, {'ops': [bad]})

    def test_quick_metric_small_catalogue_and_empty_positives(self):
        s = np.array([[2., 1.]])
        rows = np.array([[0, 0, 5, 0]])
        empty = np.empty((0, 4), dtype=int)
        self.assertEqual(quick_metric(s, rows, empty), 1.)
        self.assertIsNone(quick_metric(s, np.array([[0, 1, 2, 0]]), empty))


if __name__ == '__main__':
    unittest.main()
