import unittest
import torch
from wsprec.cli import FM, evaluate


class FMTests(unittest.TestCase):
    def test_two_field_formula(self):
        m = FM(2, 3, 2)
        with torch.no_grad():
            m.bias.fill_(1)
            m.user_bias.weight.fill_(2)
            m.item_bias.weight.fill_(3)
            m.user.weight.fill_(2)
            m.item.weight.fill_(4)
        result = m(torch.tensor([0, 1]), torch.tensor([1, 2]))
        torch.testing.assert_close(result, torch.tensor([22., 22.]))

    def test_metrics(self):
        import numpy as np
        m = FM(1, 1, 2)
        with torch.no_grad():
            for parameter in m.parameters():
                parameter.zero_()
        result = evaluate(m, np.array([[0, 0, 3, 0]]))
        self.assertEqual(result['rmse'], 3.)
        self.assertEqual(result['mae'], 3.)


if __name__ == '__main__':
    unittest.main()
