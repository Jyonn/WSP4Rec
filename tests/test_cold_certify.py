"""Synthetic compiler/incorrect-weight fixtures only; no research checkpoints."""
import unittest
import numpy as np
from wsprec.cold_benchmark import compiled_network, fold_known_status
from wsprec.cold_certify import certify_written, fold_certificate

try:
    import z3
except ImportError:
    z3 = None


class FoldCertificateTests(unittest.TestCase):
    def test_exact_and_rounded_fold(self):
        source = compiled_network(0)
        source['b1'][0] = np.float32(1e8)
        source['W1'][20,0] = 1
        source['W1'][21,0] = 2
        result = fold_certificate(source, fold_known_status(source))
        self.assertTrue(result['recognized_fold'])
        self.assertEqual(result['output_error_bound_exact'], '9')
        self.assertTrue(result['bias_rounding_included'])
        self.assertFalse(result['float32_execution_certified'])

    def test_other_change_unrecognized(self):
        source = compiled_network(0)
        edited = fold_known_status(source)
        edited['W2'][0,0] += 1
        self.assertEqual(fold_certificate(source, edited)['status'], 'not_recognized')


@unittest.skipIf(z3 is None, 'optional z3-solver unavailable')
class SymbolicCertificateTests(unittest.TestCase):
    def test_compiled_policy_all_tasks(self):
        for task in (0,1,2):
            with self.subTest(task=task):
                result = certify_written(compiled_network(task), task)
                self.assertEqual(result['status'], 'unsat', result)
                self.assertTrue(result['verified'])

    def test_wrong_matrix_counterexample(self):
        s = compiled_network(2)
        s['W3'][0,0] += 1
        result = certify_written(s, 2)
        self.assertEqual(result['status'], 'sat', result)
        self.assertGreater(abs(result['counterexample']['error']), 1e-6)
        self.assertEqual(len(result['counterexample']['features']), 22)

    def test_permutation(self):
        p = np.roll(np.arange(22), 3)
        source = compiled_network(1)
        source['W1'] = source['W1'][p]
        self.assertTrue(certify_written(source, 1, permutation=p)['verified'])
        wrong = certify_written(source, 1)
        self.assertEqual(wrong['status'], 'sat', wrong)

    def test_tiny_nonzero_not_dropped(self):
        s = compiled_network(0)
        s['W1'][20,31] = np.float32(1e-10)
        s['W2'][31,31] = 1
        s['W3'][31,0] = np.float32(1e6)
        result = certify_written(s, 0)
        self.assertEqual(result['status'], 'sat', result)


if __name__ == '__main__':
    unittest.main()
