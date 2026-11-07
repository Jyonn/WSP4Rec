"""Audit exact text/ID alignment and paired Astra access conditions."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wsprec.text_study import ROOT,OUT,DATA,load,save,digest
from wsprec.study import checkpoint,quick_metric
from wsprec.text_model import scores,feature_path
assign=load(OUT/'assignments.json');rows=[];train=np.load(DATA/'train.npy');valid=np.load(DATA/'valid.npy')
for key,a in assign.items():
 d=OUT/'inputs'/key;state=checkpoint(ROOT/a['model'])['state'];source=checkpoint(ROOT/a['source_checkpoint'])['state']
 with np.load(d/'parameters.npz') as z:
  assert set(z.files)==set(state)
  for name in state:
   assert np.array_equal(z[name],state[name].numpy())
   assert np.array_equal(z[name],source[name].numpy())
 assert digest(d/'train.npy')==digest(DATA/'train.npy')
 raw_allowed=any(Path(p).name=='movies.json' for p in a['allowlist'])
 assert raw_allowed==(a['access']=='full_text')
 if a['access']=='none':
  assert 'text_projection' not in state and a['projected_features'] is None
  assert not any(Path(p).name in ('features.npy','llama_features.npy') for p in a['allowlist'])
 else:
  assert digest(d/'features.npy')==digest(OUT/'text/features.npy')
  assert digest(d/'llama_features.npy')==digest(DATA/'title_description.npy')
  assert tuple(state['text_projection'].shape)==(64,16)
 if raw_allowed:assert load(d/'movies.json')==load(DATA/'movies.json')
 mapping=load(d/'id_to_embedding_row.json');assert mapping['item_id_to_item_embedding_row']==list(range(1568));assert mapping['user_id_to_user_embedding_row']==list(range(943))
 rows.append({'case':key,'access':a['access'],'seed':a['seed'],'replicate':a['replicate'],'exact_input_model_match':True,'raw_text_permission_correct':True,'exact_feature_alignment':True})
for seed in (42,43,44):
 related=[a for a in assign.values() if a['seed']==seed and a['model_condition']=='text']
 assert len(related)==4
 assert len(set(digest(ROOT/a['model']) for a in related))==1
# Audit selected training validation on the local canonical scorer, without selecting anew.
checks=[]
for condition in ('none','text','shuffled_text'):
 summary=load(OUT/condition/'training_summary.json')
 X=np.load(feature_path(condition)) if condition!='none' else None
 for seed,path in summary['selected'].items():
  record=next(t for t in summary['trials'] if t['checkpoint']==path)
  current=quick_metric(scores(checkpoint(ROOT/path)['state'],X),valid,train)
  checks.append({'condition':condition,'seed':int(seed),'training_validation':record['valid_ndcg@10'],'local_validation':current,'absolute_backend_difference':abs(current-record['valid_ndcg@10'])})
R=ROOT/'reports/v7_text';R.mkdir(exist_ok=True,parents=True)
save(R/'input_audit.json',{'cases':rows,'selected_checkpoint_validation_checks':checks,'same_text_checkpoint_across_numeric_full_and_replicates':True,'limitation':'Instruction allowlists, not OS-enforced file isolation. Hash checks verify supplied inputs, not all editor tool-read behavior.'})
print('Verified',len(rows),'inputcases; maximum backend validation difference',max(r['absolute_backend_difference'] for r in checks))
