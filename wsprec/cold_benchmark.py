"""Post-lock fictional recommendation worlds for semantic weight programming.

Only the evaluator imports this module. Native editors receive the public input
contract and policy text, never this oracle, training labels, or held-out worlds.
All entities are generated for this experiment; their meanings are the supplied
attributes/preferences, not facts recoverable from a pretraining corpus.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/v8_mit'
SHAPES={'W1':(22,32),'b1':(32,),'W2':(32,32),'b2':(32,),'W3':(32,1),'b3':(1,)}
USER_COLUMNS=(0,1,2,3,8,10,12,14,16,18)
ITEM_COLUMNS=(4,5,6,7,9,11,13,15,17,19)
FEATURE_NAMES=(
 'user_likes_genre_0','user_likes_genre_1','user_likes_genre_2','user_likes_genre_3',
 'item_genre_0','item_genre_1','item_genre_2','item_genre_3',
 'user_wants_language_A','item_language_A','user_avoids_graphic','item_graphic',
 'user_allows_documentary_exception','item_documentary','user_time_budget',
 'item_runtime','user_wants_new','item_new','user_avoids_strobe','item_strobe',
 'user_known_to_training','item_known_to_training')
SEEDS=(42,43,44)
TRAIN_SIZES=(128,2048,16384)


def load(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def policy_spec(task):
 if task not in (0,1,2):raise ValueError('task must be0,1,2')
 return {'task':task,'genre_names':['Sci-Fi','Comedy','Drama','Action'],
         'genre_weights':([3,2,4,1],[1,4,2,3],[4,1,3,2])[task],
         'language_bonus':2.,'fresh_bonus':1.,'runtime_excess_penalty':4.,
         'veto_penalty':40.,'documentary_exception':task>=1,
         'language_mismatch_veto':task>=1,'genre0_language_joint_bonus':5. if task==2 else 0.,
         'feature_names':list(FEATURE_NAMES),
         'known_status_semantics':'Whether an entity identity belonged to the recommender training catalogue; desired utility depends on its supplied preferences/attributes, not identity status.'}


def oracle(features,task):
 """Independent explicit utility definition, not implemented by the neural model."""
 x=np.asarray(features,dtype=np.float64)
 if x.shape[-1]!=22:raise ValueError('22 input channels required')
 p=policy_spec(task)
 matched=x[...,:4]*x[...,4:8]
 result=matched@np.asarray(p['genre_weights'],dtype=float)
 result+=2*x[...,8]*x[...,9]+x[...,16]*x[...,17]
 result-=4*np.maximum(x[...,15]-x[...,14],0)
 graphic=x[...,10]*x[...,11]
 if task>=1:graphic*=1-x[...,12]*x[...,13]
 result-=40*(graphic+x[...,18]*x[...,19])
 if task>=1:result-=40*x[...,8]*(1-x[...,9])
 if task==2:result+=5*x[...,0]*x[...,4]*x[...,8]*x[...,9]
 return result


def veto_mask(features,task):
 x=np.asarray(features);graphic=(x[...,10]>0)&(x[...,11]>0)
 if task>=1:graphic &=~((x[...,12]>0)&(x[...,13]>0))
 veto=graphic|((x[...,18]>0)&(x[...,19]>0))
 if task>=1:veto|=(x[...,8]>0)&(x[...,9]<1)
 return veto


def empty_network():return {k:np.zeros(shape,dtype=np.float32) for k,shape in SHAPES.items()}


def validate_network(state):
 if set(state)!=set(SHAPES):raise ValueError('exactly W1,b1,W2,b2,W3,b3 required')
 result={}
 for name,shape in SHAPES.items():
  a=np.asarray(state[name],dtype=np.float32)
  if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'invalid{name}')
  result[name]=a.copy()
 return result


def forward(state,features,return_hidden=False):
 s=validate_network(state);x=np.asarray(features,dtype=np.float32)
 h1=np.maximum(x@s['W1']+s['b1'],0)
 h2=np.maximum(h1@s['W2']+s['b2'],0)
 y=(h2@s['W3']+s['b3'])[...,0]
 return (y,h1,h2) if return_hidden else y


def compiled_network(task):
 """Fixed symbolic policy-to-ReLU compiler, an exact same-architecture baseline."""
 s=empty_network();w=s['W1'];b=s['b1']
 for genre in range(4):w[genre,genre]=w[genre+4,genre]=1;b[genre]=-1
 for unit,indices in ((4,(8,9)),(5,(16,17)),(7,(10,11)),(8,(12,13)),(9,(18,19))):
  w[list(indices),unit]=1;b[unit]=-1
 w[15,6]=1;w[14,6]=-1
 w[8,10]=1;w[9,10]=-1
 w[[0,4,8,9],11]=1;b[11]=-3
 for unit in range(7):s['W2'][unit,unit]=1
 s['W2'][7,7]=1
 if task>=1:s['W2'][8,7]=-1
 s['W2'][9,8]=1
 if task>=1:s['W2'][10,9]=1
 if task==2:s['W2'][11,10]=1
 s['W3'][:4,0]=policy_spec(task)['genre_weights']
 s['W3'][4:10,0]=[2,1,-4,-40,-40,-40]
 s['W3'][10,0]=5 if task==2 else 0
 return s


def fold_known_status(state):
 """Exact warm-preserving fixed repair: fold constant training-status channels."""
 s=validate_network(state)
 s['b1']=(s['b1']+s['W1'][20]+s['W1'][21]).astype(np.float32)
 s['W1'][20:22]=0
 return s


def pair_features(users,items,user_known=1,item_known=1):
 u=np.asarray(users,dtype=np.float32);i=np.asarray(items,dtype=np.float32)
 if u.ndim!=2 or i.ndim!=2 or u.shape[1]!=10 or i.shape[1]!=10:raise ValueError('10D user and item rows required')
 result=np.zeros((len(u),len(i),22),dtype=np.float32)
 result[...,USER_COLUMNS]=u[:,None,:]
 result[...,ITEM_COLUMNS]=i[None,:,:]
 result[...,20]=user_known;result[...,21]=item_known
 return result


def _entities(rng,count,kind):
 rows=np.zeros((count,10),dtype=np.float32)
 for row in rows:
  row[rng.choice(4,size=int(rng.integers(1,3)),replace=False)]=1
 rows[:,4:]=rng.integers(0,2,size=(count,6))
 if kind=='user':rows[:,7]=rng.uniform(.2,1,size=count)
 else:
  rows[:,5]=(rng.random(count)<.35)
  rows[:,6]=(rng.random(count)<.35)
  rows[:,7]=rng.uniform(.1,1,size=count)
  rows[:,9]=(rng.random(count)<.25)
 ids=[str(uuid.UUID(bytes=rng.bytes(16),version=4)) for _ in range(count)]
 return rows,ids


def make_world(seed,warm_users=128,warm_items=256,practice_users=32,practice_items=64,cold_users=128,cold_items=256):
 """Pure generator used by tests; no model or label selection occurs here."""
 rng=np.random.default_rng(seed);world={};identities={}
 for name,count,kind in (('warm_users',warm_users,'user'),('warm_items',warm_items,'item'),
                         ('practice_users',practice_users,'user'),('practice_items',practice_items,'item'),
                         ('cold_users',cold_users,'user'),('cold_items',cold_items,'item')):
  world[name],identities[name]=_entities(rng,count,kind)
 total=warm_users*warm_items;order=rng.permutation(total)
 return world,identities,order


def prepare(protocol,seed_base=202611080):
 """Materialize a new world only after a local experiment protocol exists."""
 protocol=Path(protocol)
 if not protocol.exists():raise FileNotFoundError('protocol must be written before generation')
 directory=OUT/'data'
 if (directory/'manifest.json').exists():raise FileExistsError('worlds already generated')
 directory.mkdir(parents=True,exist_ok=True);start=time.time();manifests={}
 for task in (0,1,2):
  seed=seed_base+task;world,ids,order=make_world(seed);folder=directory/f'task{task}';folder.mkdir()
  np.savez(folder/'entities.npz',**world)
  np.savez(folder/'warm_split_indices.npz',train_pool=order[:16384],valid=order[16384:20480],test=order[20480:])
  warm=pair_features(world['warm_users'],world['warm_items']).reshape(-1,22)
  np.save(folder/'warm_features.npy',warm);np.save(folder/'warm_utility.npy',oracle(warm,task).astype(np.float32))
  practice=pair_features(world['practice_users'],world['practice_items'],0,0).reshape(-1,22)
  # Practice examples are disjoint entities from both source and final cold pools.
  np.save(folder/'practice_features.npy',practice);np.save(folder/'practice_utility.npy',oracle(practice,task).astype(np.float32))
  save(folder/'identities.json',ids);save(folder/'policy.json',policy_spec(task))
  np.savez(folder/'compiled_oracle.npz',**compiled_network(task))
  manifests[str(task)]={'seed':seed,'identity_counts':{k:len(v) for k,v in ids.items()},
                       'warm_pairs':len(warm),'train_sizes':list(TRAIN_SIZES),'valid_pairs':4096,'warm_test_pairs':12288,
                       'generation_time':time.time(),'identity_sha256':digest(folder/'identities.json'),
                       'all_identity_pools_disjoint':len(set(v for pool in ids.values() for v in pool))==sum(map(len,ids.values()))}
 save(directory/'manifest.json',{'protocol_sha256':digest(protocol),'generated_after_protocol_at':start,'tasks':manifests,
      'identity_scope':'Fictional users/items with new random UUID identities and sampled preferences/attributes. This does not imply the concepts or policy rules were absent from LLM pretraining.',
      'source_split':'Only warm pair training/validation labels used for BP. Cold entities/labels reserved for final evaluation; practice entity pool is separate.'})
 print('Prepared fictional worlds',flush=True)


def world(task):
 with np.load(OUT/'data'/f'task{task}'/'entities.npz') as z:return {k:z[k].copy() for k in z.files}


def evaluation_slice(task,name):
 """Evaluator-only. Never called during BP training or editor discovery."""
 data=world(task)
 pools={'warm_warm':('warm_users','warm_items',1,1),'cold_user':('cold_users','warm_items',0,1),
        'cold_item':('warm_users','cold_items',1,0),'both_cold':('cold_users','cold_items',0,0)}
 if name not in pools:raise ValueError(name)
 u,i,uk,ik=pools[name];features=pair_features(data[u],data[i],uk,ik)
 candidate=np.ones(features.shape[:2],dtype=bool)
 if name=='warm_warm':
  candidate[:]=False
  with np.load(OUT/'data'/f'task{task}'/'warm_split_indices.npz') as z:candidate.reshape(-1)[z['test']]=True
 return features,oracle(features,task),candidate


def ranking_metrics(prediction,utility,candidates=None,veto=None,k=10):
 """Full eligible catalogue graded NDCG, normalized utility regret, veto exposure."""
 pred=np.asarray(prediction,dtype=float);truth=np.asarray(utility,dtype=float)
 if pred.shape!=truth.shape or pred.ndim!=2 or not np.isfinite(pred).all():raise ValueError('finite user/item matrices required')
 allowed=np.ones_like(pred,dtype=bool) if candidates is None else np.asarray(candidates,dtype=bool)
 values=[];discount=1/np.log2(np.arange(k)+2);all_items=np.arange(pred.shape[1])
 for user in range(len(pred)):
  ids=all_items[allowed[user]]
  if not len(ids):continue
  order=ids[np.lexsort((ids,-pred[user,ids]))][:k]
  rel=np.maximum(truth[user],0.)
  ideal=np.sort(rel[ids])[::-1][:k]@discount[:min(k,len(ids))]
  if ideal<=0:continue
  ndcg=float(rel[order]@discount[:len(order)]/ideal)
  best=float(truth[user,ids].max());worst=float(truth[user,ids].min())
  regret=float((best-truth[user,order[0]])/max(best-worst,1e-12))
  violations=float(np.asarray(veto)[user,order].mean()) if veto is not None else None
  values.append((user,ndcg,regret,violations))
 return {f'ndcg@{k}':float(np.mean([v[1] for v in values])) if values else None,
         'top1_normalized_regret':float(np.mean([v[2] for v in values])) if values else None,
         f'veto_exposure@{k}':float(np.mean([v[3] for v in values])) if values and veto is not None else None,
         'ranking_users':len(values),'per_user':{'user_id':[v[0] for v in values],'ndcg':[v[1] for v in values],
                                               'regret':[v[2] for v in values],'veto_exposure':[v[3] for v in values]}}


class RankMLP(nn.Module):
 def __init__(self):
  super().__init__()
  for name,shape in SHAPES.items():
   value=torch.empty(shape)
   if name.startswith('W'):nn.init.kaiming_uniform_(value.T,a=math.sqrt(5))
   else:value.zero_()
   setattr(self,name,nn.Parameter(value))
 def forward(self,x):
  h=torch.relu(x@self.W1+self.b1);h=torch.relu(h@self.W2+self.b2)
  return (h@self.W3+self.b3).squeeze(-1)


def train_one(task,size,seed,device='cpu',augment_status=False,epochs_override=None):
 condition='cold_aware' if augment_status else 'warm_only'
 folder=OUT/'bp'/f'task{task}'/f'n{size}'/f's{seed}'/condition;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'history.json').exists():return load(folder/'history.json')
 torch.manual_seed(seed);np.random.seed(seed);started=time.monotonic()
 X=np.load(OUT/'data'/f'task{task}'/'warm_features.npy');y=np.load(OUT/'data'/f'task{task}'/'warm_utility.npy')/40.
 with np.load(OUT/'data'/f'task{task}'/'warm_split_indices.npz') as split:
  train_ids=split['train_pool'][:size];valid_ids=split['valid']
 x=torch.as_tensor(X[train_ids],device=device);target=torch.as_tensor(y[train_ids],device=device)
 vx=torch.as_tensor(X[valid_ids],device=device);vy=torch.as_tensor(y[valid_ids],device=device)
 model=RankMLP().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-6)
 best=np.inf;best_epoch=0;history=[];epochs=epochs_override or 400;steps=0
 for epoch in range(1,epochs+1):
  model.train()
  for indices in torch.randperm(len(x),device=device).split(1024):
   batch=x[indices]
   if augment_status:
    batch=batch.clone();batch[:,20:22]=torch.randint(2,(len(batch),2),device=device).to(batch.dtype)
   loss=(model(batch)-target[indices]).square().mean();optimizer.zero_grad();loss.backward();optimizer.step();steps+=1
  model.eval()
  with torch.no_grad():value=float((model(vx)-vy).square().mean())
  history.append({'epoch':epoch,'normalized_valid_mse':value,'training_batch_loss':float(loss.detach())})
  if value<best:
   best,best_epoch=value,epoch
   arrays={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
   arrays['W3']*=40.;arrays['b3']*=40.
   np.savez(folder/'best.npz',**arrays)
  if epochs_override is None and epoch-best_epoch>=40:break
  if time.monotonic()-started>3500:raise TimeoutError('single BP trial time budget')
 result={'task':task,'train_pairs':size,'seed':seed,'condition':condition,'device':device,'lr':.003,'weight_decay':1e-6,
         'batch_size':1024,'max_epochs':epochs,'epochs_run':epoch,'best_epoch':best_epoch,'updates':steps,
         'selection':'Minimum normalized MSE on4096disjoint warm validationpairs; no cold labels or rankings inspected.',
         'normalized_valid_mse':best,'raw_valid_rmse':float(40*np.sqrt(best)),'wall_seconds':time.monotonic()-started,
         'checkpoint':str((folder/'best.npz').relative_to(ROOT)),
         'output_units':'Training predicts utility/40. Exported W3,b3 multiplied40, so saved network outputs raw utility.',
         'status_augmentation':'Independently Bernoulli(.5) known-user/item flags, same original utility labels' if augment_status else None,
         'epochs_fixed_to_source_budget':epochs_override,'history':history}
 save(folder/'history.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
 return result


def train_all(device='cpu',tasks=(0,1,2)):
 trials=[]
 for task in tasks:
  for size in TRAIN_SIZES:
   for seed in SEEDS:trials.append(train_one(task,size,seed,device))
 if 1 in tasks:
  for seed in SEEDS:
   baseline=next(t for t in trials if t['task']==1 and t['train_pairs']==16384 and t['seed']==seed)
   trials.append(train_one(1,16384,seed,device,True,baseline['epochs_run']))
 summary_path=OUT/'bp'/(f'task{tasks[0]}' if len(tasks)==1 else '')/'training_summary.json'
 save(summary_path,{'trials':trials,'architecture':'22->32ReLU->32ReLU->1','torch':torch.__version__,'numpy':np.__version__,
      'repair_sources':{str(t['seed']):t['checkpoint'] for t in trials if t['task']==1 and t['train_pairs']==16384 and t['condition']=='warm_only'},
      'all_cold_labels_unread_during_training':True})
 return trials


if __name__=='__main__':
 torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','train'));p.add_argument('--protocol',type=Path,default=ROOT/'cold_protocol_v8.json');p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--task',type=int,choices=(0,1,2));args=p.parse_args()
 prepare(args.protocol) if args.command=='prepare' else train_all(args.device,tasks=(args.task,) if args.task is not None else (0,1,2))
