import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from wsprec import cold_benchmark as b


class ColdBenchmarkTests(unittest.TestCase):
 def test_compiler_agrees_with_independent_oracle(self):
  rng=np.random.default_rng(9)
  # Exhaust all relevant veto/exception/language binary conditions, with random genres/times.
  x=rng.integers(0,2,size=(2**10,22)).astype(np.float32)
  for j,channel in enumerate((0,4,8,9,10,11,12,13,18,19)):
   x[:,channel]=(np.arange(len(x))>>j)&1
  x[:,14:16]=rng.random((len(x),2))
  for task in (0,1,2):
   np.testing.assert_allclose(b.forward(b.compiled_network(task),x),b.oracle(x,task),atol=1e-5,rtol=1e-6)

 def test_status_folding_preserves_warm_and_matches_warm_counterfactual(self):
  rng=np.random.default_rng(4)
  state={k:rng.normal(size=shape).astype(np.float32) for k,shape in b.SHAPES.items()}
  x=rng.random((64,22),dtype=np.float32);x[:,20:22]=1
  repaired=b.fold_known_status(state)
  np.testing.assert_allclose(b.forward(repaired,x),b.forward(state,x),atol=1e-4,rtol=2e-6)
  x[:,20:22]=0
  warm=x.copy();warm[:,20:22]=1
  np.testing.assert_allclose(b.forward(repaired,x),b.forward(state,warm),atol=1e-4,rtol=2e-6)
  self.assertTrue(np.all(repaired['W1'][20:22]==0))

 def test_new_identity_pools_are_disjoint_and_repeatable(self):
  w,ids,order=b.make_world(11,warm_users=7,warm_items=9,practice_users=3,practice_items=4,cold_users=5,cold_items=6)
  w2,ids2,order2=b.make_world(11,warm_users=7,warm_items=9,practice_users=3,practice_items=4,cold_users=5,cold_items=6)
  self.assertEqual(ids,ids2);self.assertEqual(len(set(sum(ids.values(),[]))),34)
  np.testing.assert_array_equal(order,order2)
  for name in w:np.testing.assert_array_equal(w[name],w2[name])
  pairs=b.pair_features(w['cold_users'],w['cold_items'],0,0)
  self.assertEqual(pairs.shape,(5,6,22));self.assertTrue(np.all(pairs[...,20:22]==0))
  np.testing.assert_array_equal(pairs[0,0,list(b.USER_COLUMNS)],w['cold_users'][0])
  np.testing.assert_array_equal(pairs[0,0,list(b.ITEM_COLUMNS)],w['cold_items'][0])

 def test_oracle_status_invariance_and_documentary_exception(self):
  x=np.zeros((2,22),dtype=np.float32);x[:,10:14]=1
  self.assertEqual(b.oracle(x,0).tolist(),[-40,-40])
  self.assertEqual(b.oracle(x,1).tolist(),[0,0])
  x[1,20:22]=1
  np.testing.assert_array_equal(b.oracle(x,2),b.oracle(x[::-1],2))

 def test_rank_metrics_reward_correct_order_and_penalize_veto(self):
  utility=np.array([[3.,2.,1.,-40.],[1.,4.,2.,-40.]])
  metrics=b.ranking_metrics(utility,utility,veto=utility<0,k=2)
  self.assertAlmostEqual(metrics['ndcg@2'],1.)
  self.assertEqual(metrics['veto_exposure@2'],0.)
  self.assertEqual(metrics['top1_normalized_regret'],0.)
  reverse=b.ranking_metrics(-utility,utility,veto=utility<0,k=2)
  self.assertLess(reverse['ndcg@2'],1.)
  self.assertGreater(reverse['veto_exposure@2'],0.)

 def test_prepare_requires_protocol(self):
  with tempfile.TemporaryDirectory() as tmp:
   with patch.object(b,'OUT',Path(tmp)):
    with self.assertRaises(FileNotFoundError):b.prepare(Path(tmp)/'missing.json')


if __name__=='__main__':unittest.main()
