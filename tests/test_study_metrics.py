import unittest
import numpy as np
from wsprec.study_metrics import rank_metrics, paired_bootstrap


class StudyMetricTests(unittest.TestCase):
    def train(self):
        return np.array([[0, 0, 1, 0], [1, 1, 1, 0], [1, 1, 2, 1],
                         [2, 2, 1, 0], [2, 2, 2, 1], [2, 2, 3, 2]])

    def test_mask_group_denominators_and_global_discounts(self):
        scores = np.tile([100., 1., 3., 2.], (3, 1))
        result = rank_metrics(scores, [[0, 0, 5, 0], [0, 1, 5, 0],
                                       [0, 3, 4, 0], [0, 3, 5, 1]],
                              [[0, 0, 1, 0]], self.train(), k=2)
        d = 1 / np.log2(3)
        self.assertEqual(result['summary']['ranking_users'], 1)
        self.assertEqual(result['summary']['positives'], 2)
        self.assertAlmostEqual(result['summary']['ndcg@2'], d / (1 + d))
        self.assertEqual(result['summary']['recall@2'], .5)
        self.assertEqual(result['summary']['hit_rate@2'], 1.)
        zero = result['item_groups']['zero']
        self.assertAlmostEqual(zero['ndcg@2'], d)
        self.assertEqual(zero['recall@2'], 1.)
        self.assertEqual(zero['positives'], 1)
        self.assertEqual(result['item_groups']['mid']['recall@2'], 0.)
        self.assertEqual(result['item_groups']['high']['ranking_users'], 0)
        self.assertIsNone(result['item_groups']['high']['ndcg@2'])
        self.assertEqual(result['user_item_groups']['low']['zero'], zero)
        self.assertEqual(result['user_item_groups']['high']['zero']['ranking_users'], 0)
        self.assertEqual(result['exposure'], {'zero': .5, 'low': 0., 'mid': 0., 'high': .5})
        self.assertEqual(result['catalogue_coverage'], .5)
        self.assertEqual(result['recommendation_slots'], 2)

    def test_train_only_groups_all_ratings_and_zero_items(self):
        result = rank_metrics(np.zeros((3, 4)), [[1, 3, 5, 0]], [], self.train())
        grouping = result['grouping']
        np.testing.assert_array_equal(grouping['user_counts'], [1, 2, 3])
        np.testing.assert_array_equal(grouping['item_counts'], [1, 2, 3, 0])
        np.testing.assert_array_equal(grouping['user_groups'], [0, 1, 2])
        np.testing.assert_array_equal(grouping['item_groups'], [1, 2, 3, 0])
        np.testing.assert_allclose(grouping['item_thresholds'], [5 / 3, 7 / 3])
        self.assertEqual(result['user_item_groups']['mid']['zero']['ranking_users'], 1)
        other = rank_metrics(np.ones((3, 4)), [[2, 0, 5, 0]] * 20, [], self.train())
        for key in ('user_groups', 'item_groups', 'user_thresholds', 'item_thresholds'):
            np.testing.assert_array_equal(grouping[key], other['grouping'][key])

    def test_ties_use_item_id_and_k_above_candidate_count(self):
        result = rank_metrics(np.ones((1, 4)), [[0, 2, 5, 0]], [[0, 0, 5, 0]], [], k=1)
        self.assertEqual(result['summary']['recall@1'], 0.)
        result = rank_metrics(np.ones((1, 4)), [[0, 2, 5, 0]], [[0, 0, 5, 0]], [], k=10)
        self.assertAlmostEqual(result['summary']['ndcg@10'], 1 / np.log2(3))
        self.assertEqual(result['recommendation_slots'], 3)
        self.assertEqual(result['exposure']['zero'], 1.)
        self.assertEqual(result['catalogue_coverage'], .75)

    def test_empty_eligible_positives_and_equal_count_quantiles(self):
        result = rank_metrics(np.zeros((2, 3)), [[0, 1, 5, 0], [1, 0, 2, 0]],
                              [[0, 1, 1, 0]], [[0, 0, 1, 0], [1, 1, 1, 0]])
        self.assertEqual(result['summary']['ranking_users'], 0)
        self.assertIsNone(result['summary']['ndcg@10'])
        self.assertEqual(result['catalogue_coverage'], 0.)
        self.assertEqual(result['per_user']['user_id'].shape, (0,))
        np.testing.assert_array_equal(result['grouping']['user_groups'], [0, 0])
        np.testing.assert_array_equal(result['grouping']['item_groups'], [1, 1, 0])

    def test_bootstrap_alignment_and_determinism(self):
        base = {'user_id': [0, 1, 2], 'ndcg': [.25, .5, .75]}
        edited = {'user_id': [2, 0, 1], 'ndcg': [.75, .75, .25]}
        result = paired_bootstrap(base, edited)
        self.assertAlmostEqual(result['mean_delta'], 1 / 12)
        self.assertEqual(result['improved_fraction'], 1 / 3)
        self.assertEqual(result['harmed_fraction'], 1 / 3)
        self.assertEqual(result['unchanged_fraction'], 1 / 3)
        # Three independent draws: all negative and all positive each occur
        # with probability 1/27, so the 95% percentile interval spans both.
        np.testing.assert_allclose(result['ci95'], [-.25, .5])
        self.assertEqual(result, paired_bootstrap(base, edited))
        identical = paired_bootstrap(base, base, repetitions=20)
        self.assertEqual(identical['ci95'], [0., 0.])
        self.assertEqual(identical['unchanged_fraction'], 1.)

    def test_bootstrap_rejects_missing_duplicate_or_empty_users(self):
        base = {'user_id': [0, 1], 'ndcg': [.2, .3]}
        for edited in ({'user_id': [0], 'ndcg': [.2]},
                       {'user_id': [0, 2], 'ndcg': [.2, .3]},
                       {'user_id': [0, 0], 'ndcg': [.2, .3]},
                       {'user_id': [0, 1], 'ndcg': [.2]},
                       {'user_id': [], 'ndcg': []}):
            with self.assertRaises(ValueError):
                paired_bootstrap(base, edited)

    def test_input_validation(self):
        for k in (0, 1.5, True):
            with self.assertRaises(ValueError):
                rank_metrics(np.zeros((1, 2)), [], [], [], k=k)
        with self.assertRaises(ValueError):
            rank_metrics(np.zeros((1, 2)), [[0, 2, 5, 0]], [], [])
        with self.assertRaises(ValueError):
            rank_metrics(np.array([[np.nan]]), [], [], [])


if __name__ == '__main__':
    unittest.main()
