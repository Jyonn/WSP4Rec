"""Exploratory scalar calibration control fixed before the v7 test pass.

Added after observing validation-stage editor operations; not preregistered.
No description interpretation or new model fitting. Eight fixed pairs per model.
"""
import time
from pathlib import Path
import numpy as np
import torch
from .study import checkpoint,quick_metric,save
from .text_model import scores,effective_items
import json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/v7_text';DATA=ROOT/'data/ml-100k-description-complete-v3'
def load(p):return json.loads(Path(p).read_text())
def generate():
 train=np.load(DATA/'train.npy');valid=np.load(DATA/'valid.npy');start=time.monotonic()
 for condition in ('none','text'):
  X=np.load(OUT/'text/features.npy') if condition=='text' else None
  for seed,path in load(OUT/condition/'training_summary.json')['selected'].items():
   directory=OUT/'calibration'/f'{condition}-s{seed}';directory.mkdir(parents=True,exist_ok=True)
   if (directory/'manifest.json').exists():raise FileExistsError('already generated')
   base=checkpoint(ROOT/path);state=base['state'];target='text_projection' if condition=='text' else 'user.weight'
   norm=np.sqrt(sum(float(v.double().square().sum()) for v in state.values()));records=[]
   for scale in (.8,1.2):
    for bias in (.5,1.,1.5,2.):
     delta={target:(scale-1)*state[target],'item_bias.weight':(bias-1)*state['item_bias.weight']}
     dn=np.sqrt(sum(float(v.double().square().sum()) for v in delta.values()));clip=min(1.,.495*norm/max(dn,1e-30));edited={k:v.clone() for k,v in state.items()}
     for k,v in delta.items():edited[k]+=v*clip
     name=f'scale{scale}-bias{bias}';dest=directory/f'{name}.pt';torch.save({'config':base['config'],'state':edited},dest)
     records.append({'id':name,'scale_parameter':target,'scale':scale,'bias_scale':bias,'clip':clip,'path':str(dest.relative_to(ROOT)),'valid_ndcg@10':quick_metric(scores(edited,X),valid,train),'relative_delta_l2':float(dn*clip/norm)})
   save(directory/'manifest.json',{'records':records,'scope':'Posthoc validation-informed control, fixed eight coefficient pairs before test, no extra editor feedback. Not preregistered or fully matched search space.'})
 save(OUT/'calibration/generation.json',{'wall_seconds':time.monotonic()-start,'candidate_scores':48,'selection':'Validation best of eight plus original, no test selection'})
def choose(condition,seed,baseline):
 records=load(OUT/'calibration'/f'{condition}-s{seed}'/'manifest.json')['records'];best=max(records,key=lambda r:r['valid_ndcg@10']);return best if best['valid_ndcg@10']>baseline else None
if __name__=='__main__':
 torch.set_num_threads(1);generate()
