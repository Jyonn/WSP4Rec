"""Supplementary frozen-policy transfer to entity-disjoint MovieLens cold start.

Synthetic worlds establish fresh-entity provenance. This real bridge instead
blocks identity lookup by its numerical interface; it cannot establish that
MovieLens movies were absent from the LLM's pretraining corpus. No raw title or
identity is supplied to the policy writer, and frozen written weights are never
adapted to this data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from .cold_benchmark import (RankMLP, USER_COLUMNS, ITEM_COLUMNS, compiled_network,
                             forward, validate_network)
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/v8_mit/real_bridge'
RAW=ROOT/'data/ml-100k'
COHORT=ROOT/'data/ml-100k-description-complete-v3'
GENRES=('Sci-Fi','Comedy','Drama','Action')
SEEDS=(42,43,44)
SUPPORTS=(0,5,10)

def load(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def read_genres(raw=RAW):
 """Use GroupLens's own genre index file, including the archived copy."""
 raw=Path(raw);path=raw/'u.genre'
 if path.exists():text=path.read_text(encoding='latin-1')
 else:
  with zipfile.ZipFile(raw/'ml-100k.zip') as z:text=z.read('ml-100k/u.genre').decode('latin-1')
 return {line.split('|')[0]:int(line.split('|')[1]) for line in text.splitlines() if line.strip()}

def item_attributes(raw,mapping):
 genres=read_genres(raw);rows=Path(raw,'u.item').read_text(encoding='latin-1').splitlines()
 indexed={int(row.split('|')[0])-1:row.split('|') for row in rows}
 result=np.zeros((len(mapping['item_new_to_old']),10),dtype=np.float32)
 for new,old in enumerate(mapping['item_new_to_old']):
  flags=np.asarray(indexed[int(old)][5:],dtype=np.float32)
  result[new,:4]=[flags[genres[name]] for name in GENRES]
  result[new,6]=flags[genres['Documentary']]
 # Runtime, graphic content, strobe, language and novelty are unavailable;
 # zeros mean they are unused in this bridge, not verified facts about movies.
 return result

def profile(support_rows,items):
 """Evidence-only profile: Beta(1,1) positive rate and binary > .5 preference."""
 rows=np.asarray(support_rows,dtype=np.int64).reshape(-1,4)
 count=np.zeros(4);positive=np.zeros(4)
 if len(rows):
  flags=items[rows[:,1],:4].astype(float)
  count=flags.sum(0);positive=flags[rows[:,2]>=4].sum(0)
 rate=(positive+1.)/(count+2.)
 user=np.zeros(10,dtype=np.float32);user[:4]=rate>.5;user[7]=1.
 continuous=(2*positive-count)/(count+2.)
 return user,continuous.astype(np.float32),{'genre_observations':count.astype(int).tolist(),'genre_positive_observations':positive.astype(int).tolist()}

def chronological_source_split(rows):
 rows=np.asarray(rows,dtype=np.int64);parts=[[],[],[]]
 for u in sorted(set(rows[:,0].tolist())):
  group=rows[rows[:,0]==u];group=group[np.lexsort((group[:,1],group[:,3]))];n=len(group)
  if n<3:continue
  a=min(max(1,8*n//10),n-2);b=min(max(a+1,9*n//10),n-1)
  for target,chunk in zip(parts,(group[:a],group[a:b],group[b:])):target.append(chunk)
 return tuple(np.concatenate(p) if p else np.empty((0,4),dtype=np.int64) for p in parts)

def cold_onboarding(warm_rows,cold_rows,cold_users,max_support=10):
 """Common future targets after the maximum support window, shared by all K."""
 warm_rows=np.asarray(warm_rows,dtype=np.int64);cold_rows=np.asarray(cold_rows,dtype=np.int64)
 supports={};targets=[];cutoffs={};excluded=[]
 for u in cold_users:
  history=warm_rows[warm_rows[:,0]==u];history=history[np.lexsort((history[:,1],history[:,3]))]
  if len(history)<max_support:excluded.append(int(u));continue
  selected=history[:max_support];cutoff=int(selected[-1,3])
  supports[int(u)]=selected;cutoffs[int(u)]=cutoff
  # Strictly later timestamps; a same-time target is not assumed to follow support.
  target=cold_rows[(cold_rows[:,0]==u)&(cold_rows[:,3]>cutoff)]
  if len(target):targets.append(target)
 return supports,np.concatenate(targets) if targets else np.empty((0,4),dtype=np.int64),cutoffs,excluded

def prepare(protocol):
 protocol=Path(protocol)
 if not protocol.exists():raise FileNotFoundError('lock the parent protocol before preparation')
 if (OUT/'data/manifest.json').exists():raise FileExistsError('real bridge data already prepared')
 started=time.monotonic();mapping=load(COHORT/'mapping.json');items=item_attributes(RAW,mapping)
 raw=np.loadtxt(RAW/'u.data',dtype=np.int64);raw[:,:2]-=1
 users=mapping['user_new_to_old'];original_items=mapping['item_new_to_old']
 um={old:new for new,old in enumerate(users)};im={old:new for new,old in enumerate(original_items)}
 raw=raw[np.isin(raw[:,0],users)&np.isin(raw[:,1],original_items)].copy()
 raw[:,0]=[um[int(v)] for v in raw[:,0]];raw[:,1]=[im[int(v)] for v in raw[:,1]]
 rng=np.random.default_rng(2026110819);up=rng.permutation(len(users));ip=rng.permutation(len(items))
 warm_users=np.sort(up[:int(.8*len(up))]);cold_users=np.sort(up[int(.8*len(up)):])
 warm_items=np.sort(ip[:int(.8*len(ip))]);cold_items=np.sort(ip[int(.8*len(ip)):])
 source=raw[np.isin(raw[:,0],warm_users)&np.isin(raw[:,1],warm_items)]
 train,valid,test=chronological_source_split(source)
 profiles=np.zeros((len(users),10),dtype=np.float32);profiles[:,7]=1
 continuous=np.zeros((len(users),4),dtype=np.float32);supports=[];train_targets=[]
 for u in warm_users:
  history=train[train[:,0]==u];history=history[np.lexsort((history[:,1],history[:,3]))]
  support=history[:10];profiles[u],continuous[u],_=profile(support,items)
  if len(support):supports.append(support)
  target=history[10:];target=target[target[:,2]>=4]
  if len(target):train_targets.append(target)
 onboarding,targets,cutoffs,excluded=cold_onboarding(
  raw[np.isin(raw[:,0],cold_users)&np.isin(raw[:,1],warm_items)],
  raw[np.isin(raw[:,0],cold_users)&np.isin(raw[:,1],cold_items)],cold_users)
 directory=OUT/'data';directory.mkdir(parents=True,exist_ok=True)
 np.save(directory/'items.npy',items);np.save(directory/'warm_profiles.npy',profiles);np.save(directory/'warm_continuous_profiles.npy',continuous)
 np.save(directory/'training_positives.npy',np.concatenate(train_targets) if train_targets else np.empty((0,4),dtype=np.int64))
 np.savez(directory/'source_splits.npz',train=train,valid=valid,test=test,warm_users=warm_users,cold_users=cold_users,warm_items=warm_items,cold_items=cold_items)
 np.savez(directory/'cold_onboarding.npz',**{str(u):h for u,h in onboarding.items()})
 np.save(directory/'cold_targets.npy',targets)
 total_genres=items[train[:,1],:4].sum(0);positive_genres=items[train[train[:,2]>=4,1],:4].sum(0)
 global_rate=(positive_genres+1.)/(total_genres+2.)
 np.save(directory/'global_genre_positive_rate.npy',global_rate.astype(np.float32))
 counts={'users':len(users),'items':len(items),'retained_interactions':len(raw),
         'warm_users':len(warm_users),'cold_users':len(cold_users),'warm_items':len(warm_items),'cold_items':len(cold_items),
         'source_train_rows':len(train),'source_valid_rows':len(valid),'source_test_rows':len(test),
         'training_positives_after_onboarding':sum(len(x) for x in train_targets),
         'onboarding_eligible_cold_users':len(onboarding),'cold_users_excluded_fewer_than_10_warm_supports':len(excluded),
         'cold_target_rows_after_common_cutoff':len(targets)}
 save(directory/'manifest.json',{'counts':counts,'split_rng':2026110819,'genres':list(GENRES),
   'protocol_sha256':digest(protocol),'raw_ratings_sha256':digest(RAW/'u.data'),'raw_items_sha256':digest(RAW/'u.item'),
   'mapping_sha256':digest(COHORT/'mapping.json'),'wall_seconds':time.monotonic()-started,
   'support_budgets':list(SUPPORTS),'support_rule':'Earliest10ratings on warm items establish one common onboarding window. K0/5/10 expose prefixes; all evaluate the same cold-item targets strictly after the10thsupport timestamp.',
   'support_scope':'K>0 means few-shot new-user onboarding, not zero-interaction users. K0 withholds known onboarding evidence on the same eligible cohort.',
   'cutoffs':cutoffs,'excluded_cold_users':excluded,
   'identity_scope':'The LLM sees no real entity names or IDs. This blocks direct identity lookup through this interface, but does not prove these real movies were absent from pretraining.',
   'unavailable_attributes':'Runtime, graphic/strobe/language/novelty indicators are set to unused zeros; they are not imputed factual labels.',
   'constraint_evaluation':'Not evaluated on real content: unavailable runtime/graphic/strobe/language fields are unused zeros, not verified safety labels.',
   'selection':'Warm-source validation only. No target cold labels used for training or selection.'})
 print(json.dumps(counts),flush=True)
 return counts

def feature_rows(users,items,u,i,user_known=1,item_known=1):
 """NumPy pair rows for aligned user/item indices, not a Cartesian product."""
 u=np.asarray(u,dtype=np.int64);i=np.asarray(i,dtype=np.int64)
 result=np.zeros((len(u),22),dtype=np.float32)
 result[:,USER_COLUMNS]=users[u];result[:,ITEM_COLUMNS]=items[i]
 result[:,20]=user_known;result[:,21]=item_known
 return result

def score_catalogue(state,profiles,items,user_ids,item_ids,user_known,item_known):
 result=np.empty((len(user_ids),len(item_ids)),dtype=np.float32)
 for row,u in enumerate(user_ids):
  features=feature_rows(profiles,items,np.full(len(item_ids),u),item_ids,user_known,item_known)
  result[row]=forward(state,features)
 return result

def ranking(scores,targets,user_ids,item_ids,seen=None,k=10):
 """Binary held-out positive ranking under one explicit shared candidate set."""
 scores=np.asarray(scores,dtype=float);targets=np.asarray(targets,dtype=np.int64).reshape(-1,4)
 if scores.shape!=(len(user_ids),len(item_ids)) or not np.isfinite(scores).all():raise ValueError('invalid score matrix')
 item_ids=np.asarray(item_ids);seen=np.empty((0,4),dtype=np.int64) if seen is None else np.asarray(seen)
 values=[];discounts=1/np.log2(np.arange(k)+2);candidate_set=set(item_ids.tolist())
 for row,u in enumerate(user_ids):
  excluded=set(seen[seen[:,0]==u,1].tolist()) if len(seen) else set()
  relevant=set(targets[(targets[:,0]==u)&(targets[:,2]>=4),1].tolist())&candidate_set-excluded
  if not relevant:continue
  mask=~np.isin(item_ids,list(excluded));candidate=item_ids[mask];pred=scores[row,mask]
  order=candidate[np.lexsort((candidate,-pred))][:k]
  hits=np.isin(order,list(relevant)).astype(float);ndcg=float(hits@discounts[:len(order)]/discounts[:min(k,len(relevant))].sum())
  values.append((int(u),ndcg,float(hits.sum()/len(relevant)),len(relevant)))
 return {'ndcg@10':float(np.mean([v[1] for v in values])) if values else None,
         'recall@10':float(np.mean([v[2] for v in values])) if values else None,
         'ranking_users':len(values),'positives':sum(v[3] for v in values),
         'per_user':{'user_id':[v[0] for v in values],'ndcg':[v[1] for v in values],'recall':[v[2] for v in values]}}

def _torch_features(user_profiles,item_features,u,i,user_status,item_status):
 x=torch.zeros((len(u),22),device=u.device,dtype=user_profiles.dtype)
 x[:,list(USER_COLUMNS)]=user_profiles[u];x[:,list(ITEM_COLUMNS)]=item_features[i]
 x[:,20]=user_status;x[:,21]=item_status
 return x

def train_one(seed,device='cpu'):
 directory=OUT/'training'/f's{seed}';directory.mkdir(parents=True,exist_ok=True)
 if (directory/'history.json').exists():return load(directory/'history.json')
 started=time.monotonic();torch.manual_seed(seed)
 items=np.load(OUT/'data/items.npy');profiles=np.load(OUT/'data/warm_profiles.npy');positives=np.load(OUT/'data/training_positives.npy')
 with np.load(OUT/'data/source_splits.npz') as z:train=z['train'];valid=z['valid'];warm_users=z['warm_users'];warm_items=z['warm_items']
 if not len(positives):raise ValueError('no training positives after onboarding')
 users=torch.as_tensor(profiles,device=device);item_features=torch.as_tensor(items,device=device)
 x=torch.as_tensor(positives,dtype=torch.long,device=device);catalogue=torch.as_tensor(warm_items,device=device)
 known=torch.zeros((len(profiles),len(items)),dtype=torch.bool,device=device)
 known[torch.as_tensor(train[:,0],device=device),torch.as_tensor(train[:,1],device=device)]=True
 if (known[x[:,0]][:,catalogue].all(1)).any():raise ValueError('user has no eligible training negative')
 model=RankMLP().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-5)
 history=[];best=-np.inf;best_epoch=0
 for epoch in range(1,121):
  for indices in torch.randperm(len(x),device=device).split(1024):
   batch=x[indices];u=batch[:,0];positive=batch[:,1];negative=catalogue[torch.randint(len(catalogue),(len(batch),),device=device)]
   bad=known[u,negative]
   while bad.any():
    negative[bad]=catalogue[torch.randint(len(catalogue),(int(bad.sum()),),device=device)];bad=known[u,negative]
   user_status=torch.randint(2,(len(batch),),device=device)
   pos_status=torch.randint(2,(len(batch),),device=device);neg_status=torch.randint(2,(len(batch),),device=device)
   pos=_torch_features(users,item_features,u,positive,user_status,pos_status)
   neg=_torch_features(users,item_features,u,negative,user_status,neg_status)
   loss=torch.nn.functional.softplus(-(model(pos)-model(neg))).mean();optimizer.zero_grad();loss.backward();optimizer.step()
  state={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
  metric=ranking(score_catalogue(state,profiles,items,warm_users,warm_items,1,1),valid,warm_users,warm_items,train)
  value=metric['ndcg@10']
  if value is None:raise ValueError('no warm validation positives')
  history.append({'epoch':epoch,'valid_ndcg@10':value,'last_batch_loss':float(loss.detach())})
  if value>best:best,best_epoch=value,epoch;np.savez(directory/'best.npz',**state)
  if epoch-best_epoch>=15:break
  if time.monotonic()-started>3500:raise TimeoutError('single real BP trial budget')
 result={'seed':seed,'device':device,'lr':.003,'weight_decay':1e-5,'max_epochs':120,'patience':15,'batch_size':1024,
         'best_epoch':best_epoch,'epochs_run':epoch,'valid_ndcg@10':best,'wall_seconds':time.monotonic()-started,
         'status_dropout':'User status shared by positive/negative pair; item status independent; all Bernoulli(.5), no new labels.',
         'selection':'Warm source validation NDCG only; no cold target labels accessed.',
         'checkpoint':str((directory/'best.npz').relative_to(ROOT)),'history':history}
 save(directory/'history.json',result);print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
 return result

def train_all(device='cpu'):
 if not (OUT/'data/manifest.json').exists():raise FileNotFoundError('prepare first')
 if (OUT/'training_summary.json').exists():raise FileExistsError('already trained')
 trials=[train_one(seed,device) for seed in SEEDS]
 result={'trials':trials,'selected':{str(t['seed']):t['checkpoint'] for t in trials},'torch':torch.__version__,'numpy':np.__version__,'device':device}
 save(OUT/'training_summary.json',result);return result

def _read_network(path):
 with np.load(path,allow_pickle=False) as z:return validate_network({name:z[name] for name in z.files})

def final(frozen_written_task0_models,freeze_manifest=None):
 """One final pass on common real cold targets, never used to choose a policy."""
 manifest=Path(freeze_manifest) if freeze_manifest else OUT.parent/'frozen_manifest.json'
 if not manifest.exists():raise FileNotFoundError('parent must freeze all policies before real target scoring')
 with (OUT/'test_opened.lock').open('x') as f:f.write(str(time.time()))
 started=time.monotonic();items=np.load(OUT/'data/items.npy');targets=np.load(OUT/'data/cold_targets.npy')
 with np.load(OUT/'data/source_splits.npz') as z:cold_items=z['cold_items'];cold_users=z['cold_users']
 with np.load(OUT/'data/cold_onboarding.npz') as z:onboarding={int(k):z[k] for k in z.files}
 user_ids=np.array(sorted(onboarding),dtype=np.int64)
 baseline_models={f'bp-s{s}':_read_network(ROOT/p) for s,p in load(OUT/'training_summary.json')['selected'].items()}
 baseline_models['fixed_symbolic_task0']=compiled_network(0)
 written_models={};invalid=[]
 for index,entry in enumerate(frozen_written_task0_models):
  entry={'path':str(entry),'id':f'astra-task0-{index}'} if isinstance(entry,(str,Path)) else dict(entry)
  name=entry.get('id',f'astra-task0-{index}')
  if entry.get('valid') is False or not entry.get('path'):
   invalid.append({'id':name,'error':'invalid or absent frozen network'});continue
  path=Path(entry['path']);path=path if path.is_absolute() else ROOT/path
  try:written_models[name]=_read_network(path)
  except (OSError,ValueError,KeyError) as error:invalid.append({'id':name,'error':str(error)})
 global_rate=np.load(OUT/'data/global_genre_positive_rate.npy')
 item_genres=items[cold_items,:4].astype(float)
 global_scores=item_genres@global_rate/np.maximum(item_genres.sum(1),1)
 results={}
 for budget in SUPPORTS:
  profiles=np.zeros((load(OUT/'data/manifest.json')['counts']['users'],10),dtype=np.float32)
  profiles[:,7]=1.;continuous=np.zeros((len(profiles),4),dtype=np.float32)
  for user,history in onboarding.items():profiles[user],continuous[user],_=profile(history[:budget],items)
  models={**baseline_models,**written_models};metrics={}
  for name,state in models.items():metrics[name]=ranking(score_catalogue(state,profiles,items,user_ids,cold_items,0,0),targets,user_ids,cold_items)
  pref=continuous[user_ids].astype(float);denom=np.linalg.norm(pref,axis=1)[:,None]*np.linalg.norm(item_genres,axis=1)[None,:]
  cosine=(pref@item_genres.T)/np.maximum(denom,1e-12)
  metrics['genre_profile_cosine']=ranking(cosine,targets,user_ids,cold_items)
  metrics['global_genre_positive_rate']=ranking(np.broadcast_to(global_scores,(len(user_ids),len(cold_items))),targets,user_ids,cold_items)
  metrics['constant_score_reference']=ranking(np.zeros((len(user_ids),len(cold_items))),targets,user_ids,cold_items)
  counts={m['ranking_users'] for m in metrics.values()};assert len(counts)==1
  results[str(budget)]={'metrics':metrics,'onboarding_users':len(user_ids),'candidate_items':len(cold_items),
                        'positive_target_cohort_shared_across_budgets':True}
 result={'budgets':results,'invalid_frozen_programs':invalid,'frozen_manifest_sha256':digest(manifest),
         'data_manifest_sha256':digest(OUT/'data/manifest.json'),'wall_seconds':time.monotonic()-started,
         'interpretation':'Frozen synthetic-policy compilation does not establish accuracy on private real preferences. Real BP trains source interaction labels; written policies receive no real-data fitting. K>0 is few-shot onboarding, K0 withholds evidence on the same cohort. Real movie pretraining membership unknown; identity lookup is blocked by the input interface.'}
 save(OUT/'final_results.json',result);print('Real bridge final scoring complete',flush=True);return result

if __name__=='__main__':
 torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','train','final'))
 p.add_argument('--protocol',type=Path,default=ROOT/'cold_wsp_protocol_v8.json');p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
 p.add_argument('--models',type=Path,help='JSON list of frozen task0 model paths/records for final')
 p.add_argument('--freeze-manifest',type=Path)
 args=p.parse_args()
 if args.command=='prepare':prepare(args.protocol)
 elif args.command=='train':train_all(args.device)
 else:
  if args.models is None:p.error('--models is required for final')
  final(load(args.models),args.freeze_manifest)
