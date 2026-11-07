import unittest
import numpy as np
from wsprec.cold_study import public_contract,DSL
from wsprec.cold_benchmark import compiled_network,forward,pair_features,make_world

class ColdStudyTests(unittest.TestCase):
 def test_contract_has_complete_permuted_legend(self):
  perm=np.arange(22)[::-1];text=public_contract(2,perm,True)
  self.assertIn('Input 0: symbol q2_0: item_known_to_training',text)
  self.assertIn('Add5 additionally',text)
  self.assertIn('except when BOTH',text)
  self.assertEqual(text.count('Input '),22)
 def test_canonicalization_is_algebraic_input_reindexing(self):
  state=compiled_network(1);perm=np.random.default_rng(4).permutation(22)
  encoded={k:v.copy() for k,v in state.items()};encoded['W1']=state['W1'][perm]
  canonical={k:v.copy() for k,v in encoded.items()};canonical['W1']=encoded['W1'][np.argsort(perm)]
  for k in state:np.testing.assert_array_equal(state[k],canonical[k])
  world,_,_=make_world(3,warm_users=3,warm_items=4)
  x=pair_features(world['warm_users'],world['warm_items'])
  np.testing.assert_allclose(forward(encoded,x[...,perm]),forward(canonical,x),atol=1e-5)
 def test_public_prompt_excludes_implementation_helpers(self):
  text=public_contract(0,np.arange(22))+DSL
  for word in ('compiled_network','fold_known_status','warm_utility.npy','entities.npz'):
   self.assertNotIn(word,text)
  self.assertIn('No tools, training, fitting',text)

if __name__=='__main__':unittest.main()
