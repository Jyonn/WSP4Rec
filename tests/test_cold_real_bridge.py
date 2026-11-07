import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import torch
from wsprec import cold_real_bridge as bridge
from wsprec.cold_benchmark import compiled_network,forward,RankMLP

class RealColdBridgeTests(unittest.TestCase):
 def test_profile_uses_only_supplied_support_and_zero_means_no_evidence(self):
  items=np.zeros((3,10),dtype=np.float32);items[0,0]=1;items[1,1]=1;items[2,0]=1
  support=np.array([[0,0,5,1],[0,1,1,2]],dtype=np.int64)
  user,continuous,counts=bridge.profile(support,items)
  np.testing.assert_array_equal(user[:4],[1,0,0,0]);self.assertEqual(user[7],1.)
  self.assertGreater(continuous[0],0);self.assertLess(continuous[1],0)
  self.assertEqual(counts['genre_observations'],[1,1,0,0])
  empty,soft,_=bridge.profile([],items)
  np.testing.assert_array_equal(empty[:4],np.zeros(4));np.testing.assert_array_equal(soft,np.zeros(4))

 def test_common_onboarding_cutoff_is_after_maximum_support_and_strict(self):
  warm=np.array([[7,i,4,i+1] for i in range(10)]+[[8,2,5,3]],dtype=np.int64)
  cold=np.array([[7,90,5,5],[7,91,5,10],[7,92,5,11],[8,92,5,20]],dtype=np.int64)
  supports,targets,cutoffs,excluded=bridge.cold_onboarding(warm,cold,[7,8])
  self.assertEqual(set(supports),{7});self.assertEqual(cutoffs,{7:10});self.assertEqual(excluded,[8])
  np.testing.assert_array_equal(targets,np.array([[7,92,5,11]]))
  self.assertEqual(len(supports[7][:0]),0);self.assertEqual(len(supports[7][:5]),5)

 def test_archive_genre_indices_are_used_without_movie_name_inference(self):
  labels=['unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)
   with zipfile.ZipFile(root/'ml-100k.zip','w') as z:z.writestr('ml-100k/u.genre',''.join(f'{g}|{i}\n' for i,g in enumerate(labels)))
   flags=np.zeros(19,dtype=int);flags[[15,5,7]]=1
   (root/'u.item').write_text('1|Unseen Fiction|||'+'url|'+'|'.join(map(str,flags))+'\n',encoding='latin-1')
   attributes=bridge.item_attributes(root,{'item_new_to_old':[0]})
   np.testing.assert_array_equal(attributes[0,:4],[1,1,0,0]);self.assertEqual(attributes[0,6],1)
   self.assertEqual(attributes[0,7],0);self.assertEqual(attributes[0,5],0)

 def test_source_split_is_chronological_and_disjoint(self):
  rows=np.array([[0,i,4,20-i] for i in range(20)],dtype=np.int64)
  train,valid,test=bridge.chronological_source_split(rows)
  self.assertEqual((len(train),len(valid),len(test)),(16,2,2))
  self.assertLess(train[:,3].max(),valid[:,3].min());self.assertLess(valid[:,3].max(),test[:,3].min())
  self.assertEqual(len(set(train[:,1])&set(valid[:,1])),0)

 def test_ranking_uses_shared_candidates_and_excludes_seen(self):
  users=np.array([5]);items=np.array([2,3,4]);targets=np.array([[5,2,5,10],[5,4,5,20]])
  seen=np.array([[5,2,3,1]])
  result=bridge.ranking(np.array([[100.,0.,1.]]),targets,users,items,seen)
  self.assertEqual(result['ranking_users'],1);self.assertEqual(result['positives'],1)
  self.assertEqual(result['ndcg@10'],1.);self.assertEqual(result['recall@10'],1.)

 def test_frozen_compiler_receives_profiles_attributes_without_ids(self):
  users=np.zeros((10,10),dtype=np.float32);users[8,0]=1;users[:,7]=1
  items=np.zeros((3,10),dtype=np.float32);items[2,0]=1
  features=bridge.feature_rows(users,items,np.array([8]),np.array([2]),0,0)
  self.assertEqual(features.shape,(1,22));np.testing.assert_array_equal(features[0,20:],[0,0])
  state=compiled_network(0);before={k:v.copy() for k,v in state.items()}
  np.testing.assert_allclose(forward(state,features),[3])
  for k in state:np.testing.assert_array_equal(state[k],before[k])

 def test_cold_aware_feature_batches_support_real_bpr_gradients(self):
  torch.manual_seed(42);model=RankMLP()
  users=torch.zeros((2,10));users[:,0]=1;users[:,7]=1
  items=torch.zeros((2,10));items[0,0]=1;items[1,1]=1
  u=torch.tensor([0,1]);s=torch.zeros(2,dtype=torch.long)
  positive=bridge._torch_features(users,items,u,torch.tensor([0,0]),s,s)
  negative=bridge._torch_features(users,items,u,torch.tensor([1,1]),s,s)
  loss=torch.nn.functional.softplus(-(model(positive)-model(negative))).mean();loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()),0.)

if __name__=='__main__':unittest.main()
