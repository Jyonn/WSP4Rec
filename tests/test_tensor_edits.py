import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from wsprec.tensor_edits import compile_patch


class TensorEditsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "assets"
        self.root.mkdir()
        self.state = {"matrix": torch.tensor([[2., 4., 3.], [5., 6., 7.]], requires_grad=True),
                      "other": torch.tensor(10.)}

    def run_ops(self, *ops, state=None):
        return compile_patch(self.state if state is None else state,
                             {"id": "test", "operations": list(ops)}, self.root)

    def op(self, op, value, name="matrix"):
        return {"parameter": name, "op": op, "value": value}

    def asset(self, name):
        return {"parameter": "matrix", "op": "add", "asset": name}

    def test_sequential_broadcast_and_stats(self):
        original = self.state["matrix"].detach().clone()
        result, stats = self.run_ops(self.op("add", [1, 0, -1]), self.op("multiply", [[1], [0.75]]))
        expected = (original + torch.tensor([1, 0, -1])) * torch.tensor([[1], [0.75]])
        torch.testing.assert_close(result["matrix"], expected)
        delta = (original - expected).norm().item()
        self.assertEqual(stats["changed_parameters"], 5)
        self.assertEqual(stats["total_parameters"], 7)
        self.assertAlmostEqual(stats["delta_l2"], delta, places=6)
        self.assertAlmostEqual(stats["relative_delta_l2"], delta / math.sqrt(float(original.square().sum()) + 100))
        torch.testing.assert_close(self.state["matrix"], original)
        self.assertFalse(result["matrix"].requires_grad)
        result["other"].zero_()
        self.assertEqual(self.state["other"].item(), 10)
        independent, _ = self.run_ops()
        torch.testing.assert_close(independent["matrix"], original)

    def test_matrix_operations(self):
        left = [[1, 0.25], [0, 1]]
        right = [[1, 0, 0], [0.2, 1, 0], [0, 0, 1]]
        result, _ = self.run_ops(self.op("left_multiply", left), self.op("right_multiply", right))
        expected = torch.tensor(left) @ self.state["matrix"] @ torch.tensor(right)
        torch.testing.assert_close(result["matrix"], expected)

    def test_replacements(self):
        result, _ = self.run_ops(self.op("replace", 4), self.op("replace", 9, "other"))
        torch.testing.assert_close(result["matrix"], torch.full((2, 3), 4.))
        self.assertEqual(result["other"].item(), 9)
        matrix = [[2, 3, 4], [5, 6, 7]]
        result, _ = self.run_ops(self.op("replace", matrix))
        torch.testing.assert_close(result["matrix"], torch.tensor(matrix).float())

    def test_invalid_shapes_and_budget(self):
        for kind, operand in [("add", [[[1]]]), ("multiply", [1, 2]),
                              ("left_multiply", [[1, 0]]), ("right_multiply", [1]),
                              ("replace", [3]), ("add", 100), ("multiply", 2)]:
            with self.subTest(kind=kind, operand=operand), self.assertRaises(ValueError):
                self.run_ops(self.op(kind, operand))
        state = {"a": torch.tensor(2.)}
        _, stats = self.run_ops(self.op("add", 1, "a"), state=state)
        self.assertEqual(stats["relative_delta_l2"], 0.5)
        with self.assertRaises(ValueError):
            self.run_ops(self.op("add", 1.01, "a"), state=state)
        with self.assertRaises(ValueError):
            self.run_ops(self.op("add", 2, "a"), state={"a": torch.tensor(99.)})
        zero = {"a": torch.tensor(0.)}
        self.assertEqual(self.run_ops(state=zero)[1]["relative_delta_l2"], 0.)
        with self.assertRaises(ValueError):
            self.run_ops(self.op("add", 0.01, "a"), state=zero)

    def test_assets(self):
        for dtype in (np.float32, np.float64, np.int32, ">i2"):
            np.save(self.root / "v.npy", np.array([1, -1, 0], dtype=dtype))
            result, _ = self.run_ops(self.asset("v.npy"))
            torch.testing.assert_close(result["matrix"], self.state["matrix"] + torch.tensor([1, -1, 0]))
        outside = self.root.parent / "outside.npy"
        np.save(outside, [0])
        (self.root / "link.npy").symlink_to(outside)
        for name in ("../outside.npy", str(outside), "link.npy", "missing.npy"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_ops(self.asset(name))

    def test_reject_invalid_operands(self):
        for value in (True, [False, 0], float("nan"), float("inf"), 1j, "expression"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_ops(self.op("add", value))
        for value in ([np.nan], [np.inf], [1j], [True], np.array([{}], dtype=object)):
            np.save(self.root / "v.npy", value)
            with self.assertRaises(ValueError):
                self.run_ops(self.asset("v.npy"))
        with (self.root / "v.npy").open("wb") as stream:
            stream.truncate(20 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError):
            self.run_ops(self.asset("v.npy"))

    def test_invalid_operation(self):
        op = self.op("add", 0)
        for ops in ([op] * 7, [dict(op, asset="x.npy")], [dict(op, op="eval")],
                    [dict(op, parameter="missing")], [dict(op, code="pass")]):
            with self.subTest(ops=ops), self.assertRaises(ValueError):
                self.run_ops(*ops)
