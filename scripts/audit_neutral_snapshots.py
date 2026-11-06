"""Pre-test integrity replay; scores are never fed back to an editor."""
import json
import hashlib
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from wsprec.neutral_study import OUT,ROOT,cp,data,quick_metric,scores,save,digest
from wsprec.tensor_edits import compile_patch


def main():
    torch.set_num_threads(1);started=time.monotonic();train,valid=data();bases={o:cp(o)['state'] for o in ('mse','bpr')}
    inputs={}
    for path in [ROOT/'wsprec/neutral_study.py',ROOT/'wsprec/tensor_edits.py',ROOT/'wsprec/study.py',ROOT/'wsprec/study_metrics.py',ROOT/'wsprec/cli.py',
        OUT/'training_summary.json',OUT/'editor_assignments.json',ROOT/'data/ml-100k-description-complete/train.npy',ROOT/'data/ml-100k-description-complete/valid.npy']:
        inputs[str(path.relative_to(ROOT))]=digest(path)
    training=json.loads((OUT/'training_summary.json').read_text())
    for obj in ('mse','bpr'):
        path=OUT/'training'/training['selected'][obj]['42']/'best.pt';inputs[str(path.relative_to(ROOT))]=digest(path)
    checked=0;mismatches=[];duplicates=[]
    for file in sorted((OUT/'evaluations').glob('*/round*.json')):
        inputs[str(file.relative_to(ROOT))]=digest(file)
        obj=file.parent.name.split('-')[0];seen_ids=set();seen=set()
        for r in json.loads(file.read_text())['results']:
            if r['id'] in seen_ids:duplicates.append({'file':str(file.relative_to(ROOT)),'id':r['id'],'kind':'id'})
            seen_ids.add(r['id'])
            if r['error'] is not None:continue
            folder=ROOT/r['snapshot'];patch=r['patch'];payload=[]
            inputs[str((folder/'patch.json').relative_to(ROOT))]=digest(folder/'patch.json')
            for op in patch['operations']:
                item={k:v for k,v in op.items() if k!='asset'}
                if 'asset' in op:
                    item['asset_sha256']=digest(folder/op['asset'])
                    inputs[str((folder/op['asset']).relative_to(ROOT))]=item['asset_sha256']
                payload.append(item)
            signature=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
            if signature in seen:duplicates.append({'file':str(file.relative_to(ROOT)),'id':r['id'],'kind':'payload'})
            seen.add(signature)
            edited,_=compile_patch(bases[obj],patch,folder);actual=quick_metric(scores(edited),valid,train);checked+=1
            if not np.isclose(actual,r['valid_ndcg@10'],rtol=0,atol=1e-12):
                mismatches.append({'file':str(file.relative_to(ROOT)),'id':r['id'],'recorded':r['valid_ndcg@10'],'replay':actual})
    result={'passed':not mismatches and not duplicates,'replayed_scored_patches':checked,'mismatches':mismatches,'inputs_sha256':inputs,
        'duplicate_records':duplicates,'wall_seconds':time.monotonic()-started,
        'scope':'pre-test replay of exact saved assets, no new candidates, no editor feedback; physical evaluations counted separately'}
    save(OUT/'consistency_audit.json',result);print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit('Integrity audit failed; do not open test')


if __name__=='__main__':main()
