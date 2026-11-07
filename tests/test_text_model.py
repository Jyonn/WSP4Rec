import unittest
import numpy as np
import torch
from wsprec.text_model import TextMF, scores, effective_items, fit_pca, simple_fusion_delta


class TextModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = TextMF(3, 5, 2, 4)
        self.features = np.random.default_rng(7).normal(size=(5, 4)).astype(np.float32)

    def test_forward_and_full_scores_match(self):
        u = torch.tensor([0, 2, 1, 0])
        i = torch.tensor([1, 4, 2, 3])
        direct = self.model(u, i, torch.from_numpy(self.features)).detach().numpy()
        full = scores(self.model.state_dict(), self.features)
        np.testing.assert_allclose(direct, full[u, i], atol=5e-7)
        s = self.model.state_dict()
        expected = s['item.weight'] + torch.from_numpy(self.features) @ s['text_projection']
        torch.testing.assert_close(effective_items(s, self.features), expected)

    def test_without_text_matches_ordinary_fm(self):
        from wsprec.cli import FM
        torch.manual_seed(9)
        model = FM(3, 5, 2)
        u = torch.arange(3).repeat_interleave(5)
        i = torch.arange(5).repeat(3)
        expected = model(u, i).detach().numpy().reshape(3, 5)
        np.testing.assert_allclose(scores(model.state_dict()), expected, atol=5e-7)
        torch.testing.assert_close(effective_items(model.state_dict()), model.item.weight)

    def test_bpr_updates_text_projection(self):
        prior = self.model.text_projection.detach().clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
        u = torch.tensor([0, 1, 2])
        positive = self.model(u, torch.tensor([0, 1, 2]), torch.from_numpy(self.features))
        negative = self.model(u, torch.tensor([3, 4, 0]), torch.from_numpy(self.features))
        loss = torch.nn.functional.softplus(-(positive-negative)).mean()
        optimizer.zero_grad(); loss.backward()
        self.assertGreater(float(self.model.text_projection.grad.norm()), 0)
        optimizer.step()
        self.assertGreater(float((self.model.text_projection.detach()-prior).norm()), 0)

    def test_pca_reconstructs_features_without_labels(self):
        raw = np.random.default_rng(4).normal(size=(8, 7))
        x, p = fit_pca(raw, 4)
        projected = (raw-p['mean']) @ p['components'].T
        projected /= np.linalg.norm(projected, axis=1, keepdims=True)
        np.testing.assert_allclose(x, projected, atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-7)
        largest = np.argmax(np.abs(p['components']), axis=1)
        self.assertTrue(np.all(p['components'][np.arange(4), largest] >= 0))

    def test_fusion_preserves_text_branch_and_obeys_actual_budget(self):
        state = self.model.state_dict()
        U = state['user.weight'].numpy().astype(float)
        V = effective_items(state, self.features).numpy().astype(float)
        F = U + .001
        G = V - .001
        pop = np.arange(5).astype(float) - 2
        delta = simple_fusion_delta(state, self.features, None, .5, 0., target=(F,G,pop))
        self.assertNotIn('text_projection', delta)
        np.testing.assert_allclose(delta['item.weight'], .5*(G-V), atol=1e-9)
        actual_item = V + delta['item.weight']
        np.testing.assert_allclose(actual_item, .5*V+.5*G, atol=1e-9)
        delta = simple_fusion_delta(state, self.features, None, 100., 100., target=(F*100,G*100,pop))
        norm = np.sqrt(sum(np.square(v.numpy().astype(float)).sum() for v in state.values()))
        dn = np.sqrt(sum(np.square(v.astype(float)).sum() for v in delta.values()))
        self.assertLessEqual(dn/norm, .495+1e-7)


if __name__ == '__main__': unittest.main()
