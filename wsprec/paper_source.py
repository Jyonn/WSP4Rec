"""Rebuild the legacy source split from licensed MovieLens and derived ID maps.

No description text or pretrained text encoder is needed for this paper.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def prepare(raw,dest,mapping):
 if dest.exists():raise FileExistsError(dest)
 maps=json.loads(mapping.read_text());users=maps['user_new_to_old'];items=maps['item_new_to_old'];um={old:j for j,old in enumerate(users)};im={old:j for j,old in enumerate(items)};rows=np.loadtxt(raw,dtype=np.int64);rows[:,:2]-=1;rows=rows[np.isin(rows[:,0],users)&np.isin(rows[:,1],items)];parts=[[],[],[]]
 for old in sorted(users):
  g=rows[rows[:,0]==old].copy();g=g[np.lexsort((g[:,1],g[:,3]))];n=len(g);a=min(max(1,8*n//10),n-2);b=min(max(a+1,9*n//10),n-1);g[:,0]=um[old];g[:,1]=[im[i] for i in g[:,1]]
  for out,part in zip(parts,(g[:a],g[a:b],g[b:])):out.append(part)
 parts=[np.concatenate(sorted(p,key=lambda v:v[0,0])) for p in parts];dest.mkdir(parents=True);np.save(dest/'train.npy',parts[0]);np.save(dest/'valid.npy',parts[1]);np.savez(dest/'splits.npz',train=parts[0],valid=parts[1],test=parts[2]);shutil.copy2(mapping,dest/'mapping.json');(dest/'manifest.json').write_text(json.dumps({'counts':{'after':{'users':len(users),'items':len(items),'interactions':len(rows)}},'sizes':dict(zip(('train','valid','test'),map(len,parts))),'derivation':'Exact legacy anonymous-ID split reconstructed from derived ID mapping; raw ratings downloaded separately; no movie descriptions required.'},indent=2)+'\n')
 reference=ROOT/'reports/v5_paper/evidence/ml100k-dataset.json'
 if reference.exists():
  (dest/'reconstruction_provenance.json').write_text((dest/'manifest.json').read_text())
  shutil.copy2(reference,dest/'manifest.json')
 print('Prepared',dest)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=ROOT/'data/ml-100k/u.data');p.add_argument('--output',type=Path,default=ROOT/'data/ml-100k-description-complete-v3');p.add_argument('--mapping',type=Path,default=ROOT/'reports/v5_paper/source_id_mapping.json');a=p.parse_args();prepare(a.raw,a.output,a.mapping)
