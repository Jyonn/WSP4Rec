"""Administrative recovery of completed CLI turns misclassified as failures.

Default: re-audit raw events, preserve original records, and identify cases.
--continue-missing additionally replays the SAME existing answers and permits
only missing turns from the original per-case forward budget. Existing native
requests and existing feedback are never repeated.
"""
import argparse,hashlib,json,shutil,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wsprec import mit_wsp_harness as h
from wsprec.cold_benchmark import ROOT,OUT,forward
from wsprec import cold_study


def load(path):return json.loads(Path(path).read_text())
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def backup(path):
 path=Path(path);dest=OUT/'administrative_audit_original'/path.relative_to(OUT)
 if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
 return dest

def reaudit_all():
 changed=[]
 for path in sorted((OUT/'cases').glob('*/turn*/run.json')):
  previous=load(path);events=[];malformed=0
  for line in path.with_name('events.jsonl').read_text().splitlines():
   try:events.append(json.loads(line))
   except json.JSONDecodeError:malformed+=1
  audit=h.audit_events(events);answer=path.with_name('answer.txt')
  current=dict(previous);current['audit']=audit;current['malformed_event_lines']=malformed
  current['successful']=(current['returncode']==0 and not current['timed_out'] and not malformed
       and audit['tool_free'] and not audit['turn_failed'] and answer.exists() and bool(answer.read_text()))
  original=backup(path)
  current['administrative_reaudit']={'audit_version':2,'original_run_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
    'scope':'Raw event reclassification only; no model request, answer edit or weight evaluation.'}
  if current!=previous:save(path,current);changed.append(str(path.relative_to(OUT)))
 for path in sorted((OUT/'cases').glob('*/trajectory.json')):
  previous=load(path);current=dict(previous)
  current['turns']=[load(path.parent/f'turn{i}/run.json') for i in range(len(previous['turns']))]
  if current!=previous:backup(path);save(path,current)
 return changed


def recover(allow_missing=False,workers=3):
 changed=reaudit_all();assign={**cold_study._load(OUT/'write_assignments.json'),**cold_study._load(OUT/'repair_assignments.json')}
 candidates=[];skipped=[]
 for key,a in assign.items():
  case=ROOT/a['case_directory'];path=case/'trajectory.json'
  if not path.exists():continue
  result=load(path)
  if result.get('successful'):continue
  if result.get('failure')!='model_turn_failed' or not result.get('turns') or not all(t['successful'] for t in result['turns']):
   skipped.append({'case':key,'reason':'not a completed transport-notice audit failure'});continue
  needed=1+a.get('extra_forward_budget',0);available=len(result['turns'])
  if not allow_missing:
   skipped.append({'case':key,'reason':'eligible; use --continue-missing to replay/recover','cached_turns':available,'missing_budgeted_turns':needed-available});continue
  candidates.append((key,a))
 reused=[];new=[];fresh_feedback=[];cached_feedback=[]
 original_turn=h.run_turn
 def cached_turn(prompt,outdir,*args,**kwargs):
  directory=Path(outdir);record=directory/'run.json'
  if record.exists():
   if (directory/'prompt.txt').read_text()!=prompt:raise RuntimeError('cached prompt mismatch: '+str(directory))
   result=load(record)
   if not result['successful']:raise RuntimeError('cannot reuse an unsuccessful native turn')
   reused.append(str(directory.relative_to(OUT)))
   return (directory/'answer.txt').read_text(),result
  new.append(str(directory.relative_to(OUT)))
  return original_turn(prompt,outdir,*args,**kwargs)
 def one(pair):
  key,a=pair;case=ROOT/a['case_directory'];old=load(case/'trajectory.json')
  # Preserve every existing state/feedback too. We never overwrite a raw reply.
  for path in list(case.glob('*.npz'))+list(case.glob('feedback*.txt'))+[case/'trajectory.json']:backup(path)
  (case/'trajectory.json').unlink()
  with np.load(case/'initial_state.npz') as z:initial={k:z[k].copy() for k in z.files}
  counter=[0]
  def feedback(state):
   index=counter[0];counter[0]+=1;existing=case/f'feedback{index+1}.txt'
   if existing.exists():cached_feedback.append(str(existing.relative_to(OUT)));return existing.read_text()
   # This is a previously unspent practice-only batch, never a test forward.
   with np.load(OUT/'repair_inputs/practice_batches.npz') as z:x=z['x'][index].copy();y=z['y'][index].copy()
   prediction,h1,h2=forward(state,x,True)
   data={'inputs':x.tolist(),'policy_target':y.tolist(),'prediction':prediction.tolist(),
      'mean_squared_error':float(np.mean((prediction-y)**2)),
      'hidden_mean':[h1.mean(0).tolist(),h2.mean(0).tolist()],
      'hidden_active_fraction':[(h1>0).mean(0).tolist(),(h2>0).mean(0).tolist()]}
   fresh_feedback.append(str(existing.relative_to(OUT)))
   return 'External practice feedback for the single current candidate. No test data. Submit your next cumulative weight edit; no fitting or unmetered forward calls.\n'+json.dumps(data)
  try:
   result=h.run_trajectory((ROOT/a['prompt']).read_text(),initial,case,
      feedback=feedback if a.get('extra_forward_budget',0) else None,
      extra_forward_budget=a.get('extra_forward_budget',0),effort='high' if key.startswith('r') else 'xhigh',cli=cold_study.CLI)
  except Exception:
   # Preserve the original trajectory on a programming failure; no silent retry.
   if not (case/'trajectory.json').exists():shutil.copy2(OUT/'administrative_audit_original'/case.relative_to(OUT)/'trajectory.json',case/'trajectory.json')
   raise
  for original in (OUT/'administrative_audit_original'/case.relative_to(OUT)).glob('state*.npz'):
   current=case/original.name
   if current.exists():
    with np.load(original) as x,np.load(current) as y:
     for name in x.files:
      if not np.array_equal(x[name],y[name]):raise RuntimeError('cached state replay changed: '+str(current))
  result['administrative_recovery']={'reason':'Completed native replies previously rejected only by transport-notice audit',
      'original_attempt_wall_seconds':old['wall_seconds'],
      'model_turn_wall_seconds':sum(t['wall_seconds'] for t in result['turns']),
      'no_existing_native_request_repeated':True,'no_existing_feedback_forward_repeated':True}
  save(case/'trajectory.json',result)
  print(key,'recovered',result['successful'],flush=True)
  return {'case':key,'successful':result['successful'],'native_turns':len(result['turns'])}
 h.run_turn=cached_turn
 try:
  with ThreadPoolExecutor(max_workers=workers) as pool:outcomes=list(pool.map(one,candidates))
 finally:h.run_turn=original_turn
 record={'recorded_at':time.time(),'changed_run_records':changed,'recovered_cases':outcomes,'skipped':skipped,
   'reused_native_turns':reused,'new_original_budget_turns':new,'cached_feedback_reused':cached_feedback,
   'new_original_budget_practice_batches':fresh_feedback,'no_test_scoring':True,
   'backup_directory':'runs/v8_mit/administrative_audit_original',
   'cost_warning':'Count original case/turn run.json records only; backups are duplicate administrative evidence, not new requests.'}
 dest=OUT/('administrative_recovery.json' if allow_missing else 'administrative_reaudit_preview.json');save(dest,record)
 return record


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--continue-missing',action='store_true');p.add_argument('--workers',type=int,default=3);args=p.parse_args()
 print(json.dumps(recover(args.continue_missing,args.workers),indent=2))
