"""Public semantic contracts for tool-free direct weight writing."""
import argparse,json,hashlib,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .cold_benchmark import OUT,ROOT,policy_spec,FEATURE_NAMES,empty_network,save,digest
from .mit_wsp_harness import run_trajectory
CLI='/private/tmp/wsp-v8-cli/node_modules/.bin/codex'

def public_contract(task,perm,nonce=False):
 p=policy_spec(task);names=list(FEATURE_NAMES)
 for k in range(4):
  names[k]='user likes '+p['genre_names'][k];names[k+4]='item has genre '+p['genre_names'][k]
 legend='\n'.join(f'Input {j}: '+(f"symbol q{task}_{j}: " if nonce else '')+names[k] for j,k in enumerate(perm))
 rules=f"Add genre-match weights {p['genre_weights']} for Sci-Fi, Comedy, Drama, Action respectively, independently whenever the user likes that genre AND the item has it. Add2 when user wants language A AND item is language A. Add1 when user wants new AND item is new. Subtract4 times the amount by which item runtime exceeds user time budget; subtract zero when runtime is within budget. Subtract40 when user avoids strobe AND item has strobe. "
 rules+='Subtract40 when user avoids graphic content AND item has it'
 rules+=', except when BOTH the user permits documentary exceptions AND the item is documentary. ' if task>=1 else '. '
 if task>=1:rules+='Subtract40 when the user wants language A but the item is not language A. '
 if task==2:rules+='Add5 additionally when user likes Sci-Fi AND item has Sci-Fi AND user wants language A AND item is language A. '
 return ('Implement the following personalized ranking policy. Higher scalar output means preferred.\n'+legend+'\nAll fields are binary except user time budget in[.2,1] and item runtime in[.1,1]. Identity-known fields indicate whether that entity belonged to the recommender training catalogue.\nPolicy:\n'+rules+'\nNetwork: h1=ReLU(x@W1+b1); h2=ReLU(h1@W2+b2); output=h2@W3+b3. W1[22,32],b1[32],W2[32,32],b2[32],W3[32,1],b3[1].\n')

DSL='''No tools, training, fitting (including regression or least squares), gradient information, search, or forward evaluations are available. Write actual numeric weights directly. Return a concise functional explanation then exactly one fenced edits block. The block permits only assignment or +=,-=,*=,/= to existing W1,b1,W2,b2,W3,b3, literal integer indices/slices, numeric scalars/lists, and arithmetic referencing existing tensors. No imports, calls, loops, functions, new variables or comprehensions. For example b3[0]=0 is legal syntax (not a suggested policy). Preserve shapes; finite float32 weights, absolute value<=1000000. A comment-only block is legal if no edit is warranted. Do not give an executable algorithm instead of weights.\n'''

def prepare_writes():
 if (OUT/'write_assignments.json').exists():raise FileExistsError('already prepared')
 identities=[(t,e,r) for t in range(3) for e in ('natural','nonce') for r in range(2)]
 np.random.default_rng(2026110891).shuffle(identities);assign={};prompts=OUT/'prompts';prompts.mkdir(exist_ok=True,parents=True)
 for i,(task,encoding,rep) in enumerate(identities,1):
  key=f'w{i:02d}';perm=np.random.default_rng(2026110830+task).permutation(22) if encoding=='nonce' else np.arange(22)
  prompt=public_contract(task,perm,encoding=='nonce')+'\nEvery parameter is initially zero. Supply one reply only. No users, items, data, examples or outcomes are provided. The written network will not run until evaluation is locked.\n'+DSL
  path=prompts/f'{key}.txt';path.write_text(prompt)
  assign[key]={'task':task,'encoding':encoding,'replicate':rep,'permutation':perm.tolist(),'prompt':str(path.relative_to(ROOT)),'case_directory':str((OUT/'cases'/key).relative_to(ROOT))}
 save(OUT/'write_assignments.json',assign)
 snapshot=OUT/'source_snapshots/cold_study_writing_v1.py';snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,snapshot)
 files=[ROOT/'cold_wsp_protocol_v8.json',ROOT/'cold_wsp_amendments_v8.json',snapshot]+list(prompts.glob('w*.txt'))
 save(OUT/'write_input_manifest.json',{'sha256':{str(p.relative_to(ROOT)):digest(p) for p in files},'cases':12,'forward_budget':0})
 print('Prepared12zero-forwardwritingprompts',flush=True)

def run_writes(workers=3):
 assign=json.loads((OUT/'write_assignments.json').read_text())
 def run(pair):
  key,a=pair;result=run_trajectory((ROOT/a['prompt']).read_text(),empty_network(),ROOT/a['case_directory'],extra_forward_budget=0,effort='xhigh',cli=CLI);print(key,result['successful'],flush=True);return key
 with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(run,assign.items()))


def _load(path):return json.loads(Path(path).read_text())
def _state(path):
 with np.load(path) as z:return {k:z[k].copy() for k in z.files}

def prepare_repairs():
 if (OUT/'repair_assignments.json').exists():raise FileExistsError('already prepared')
 from .cold_benchmark import forward
 summary=_load(OUT/'bp/task1/training_summary.json');sources=summary['repair_sources']
 X=np.load(OUT/'data/task1/warm_features.npy');Y=np.load(OUT/'data/task1/warm_utility.npy')
 P=np.load(OUT/'data/task1/practice_features.npy');Q=np.load(OUT/'data/task1/practice_utility.npy')
 with np.load(OUT/'data/task1/warm_split_indices.npz') as z:valid=z['valid'].copy()
 rng=np.random.default_rng(2026110885);v=rng.permutation(valid);p=rng.permutation(len(P))
 initial_x=np.concatenate((X[v[:32]],P[p[:32]]));initial_y=np.concatenate((Y[v[:32]],Q[p[:32]]))
 batches_x=[];batches_y=[]
 for n in range(2):
  batches_x.append(np.concatenate((X[v[32+n*64:96+n*64]],P[p[32+n*64:96+n*64]])))
  batches_y.append(np.concatenate((Y[v[32+n*64:96+n*64]],Q[p[32+n*64:96+n*64]])))
 folder=OUT/'repair_inputs';folder.mkdir(exist_ok=True)
 np.savez(folder/'practice_batches.npz',x=np.asarray(batches_x),y=np.asarray(batches_y))
 identities=[(s,b,r) for s in (42,43,44) for b in (0,2) for r in range(2)];np.random.default_rng(2026110892).shuffle(identities);assign={}
 for i,(seed,budget,rep) in enumerate(identities,1):
  key=f'r{i:02d}';source=ROOT/sources[str(seed)];state=_state(source);pred=forward(state,initial_x)
  np.savez(folder/f'{key}-initial_trace.npz',x=initial_x,y=initial_y,prediction=pred)
  trace={'inputs':initial_x.tolist(),'policy_target':initial_y.tolist(),'current_output':pred.tolist()}
  prompt=public_contract(1,np.arange(22))+'\nThis existing BP-trained network should comply with the policy for supplied profiles and item attributes, including entities not in its original catalogue. Improve its failures while preserving previous good behavior as far as possible. Do not fit examples. Current parameter values:\n'+json.dumps({k:a.tolist() for k,a in state.items()})+'\nOne diagnostic batch of64pairs (32warmvalidation,32independentpracticeentities) has already been evaluated:\n'+json.dumps(trace)+f'\nYou may make {budget} additional externally evaluated candidate forward batches of128practicepairs, one candidate per reply. Edits are cumulative. The final network is your last reply, not a test-selected candidate.\n'+DSL.replace('or forward evaluations are available','or unmetered forward evaluations are available')
  path=OUT/'prompts'/f'{key}.txt';path.write_text(prompt)
  assign[key]={'task':1,'seed':seed,'extra_forward_budget':budget,'replicate':rep,'source':str(source.relative_to(ROOT)),'prompt':str(path.relative_to(ROOT)),'case_directory':str((OUT/'cases'/key).relative_to(ROOT)),'initial_forward_batches':1,'initial_forward_examples':64}
 save(OUT/'repair_assignments.json',assign)
 files=list(folder.glob('*'))+[ROOT/a['source'] for a in assign.values()]+[ROOT/a['prompt'] for a in assign.values()]
 save(OUT/'repair_input_manifest.json',{'sha256':{str(p.relative_to(ROOT)):digest(p) for p in set(files)},'initial_batch':64,'additional_batch':128})
 print('Prepared12repaircases',flush=True)


def run_repairs(workers=3):
 from .cold_benchmark import forward
 assign=_load(OUT/'repair_assignments.json')
 with np.load(OUT/'repair_inputs/practice_batches.npz') as z:xs=z['x'].copy();ys=z['y'].copy()
 def run(pair):
  key,a=pair;counter=[0]
  def feedback(state):
   j=counter[0];counter[0]+=1;prediction,h1,h2=forward(state,xs[j],True)
   data={'inputs':xs[j].tolist(),'policy_target':ys[j].tolist(),'prediction':prediction.tolist(),'mean_squared_error':float(np.mean((prediction-ys[j])**2)),'hidden_mean':[h1.mean(0).tolist(),h2.mean(0).tolist()],'hidden_active_fraction':[(h1>0).mean(0).tolist(),(h2>0).mean(0).tolist()]}
   return 'External practice feedback for the single current candidate. No test data. Submit your next cumulative weight edit; no fitting or unmetered forward calls.\n'+json.dumps(data)
  result=run_trajectory((ROOT/a['prompt']).read_text(),_state(ROOT/a['source']),ROOT/a['case_directory'],feedback=feedback,extra_forward_budget=a['extra_forward_budget'],effort='high',cli=CLI)
  print(key,result['successful'],flush=True)
 with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(run,assign.items()))



def freeze():
 from .mit_wsp_harness import apply_edits
 if (OUT/'frozen_manifest.json').exists():raise FileExistsError('already frozen')
 for name in ('write_input_manifest.json','repair_input_manifest.json'):
  for path,h in _load(OUT/name)['sha256'].items():assert digest(ROOT/path)==h,path
 for task in range(3):assert (OUT/f'bp/task{task}/training_summary.json').exists()
 assert (OUT/'real_bridge/training_summary.json').exists()
 selected={};bridge=[]
 for kind in ('write','repair'):
  for key,a in _load(OUT/f'{kind}_assignments.json').items():
   case=ROOT/a['case_directory'];result=_load(case/'trajectory.json');state=empty_network() if kind=='write' else _state(ROOT/a['source'])
   initial={k:v.copy() for k,v in state.items()}
   for turn,record in enumerate(result['turns']):
    actual=_load(case/f'turn{turn}/run.json');assert actual==record
    if not actual['successful']:break
    try:state,_=apply_edits(state,(case/f'turn{turn}/answer.txt').read_text())
    except ValueError:
     assert not result['successful'];break
    expected=_state(case/f'state{turn}.npz')
    for k in state:assert np.array_equal(state[k],expected[k]),(key,k)
   valid=result['successful']
   if valid:
    for k,v in _state(case/'final_state.npz').items():assert np.array_equal(state[k],v)
    path=case/'final_state.npz'
   else:
    state=initial;path=case/'operational_fallback.npz';np.savez(path,**state)
   selected[key]={'kind':kind,'valid':valid,'path':str(path.relative_to(ROOT)),**a}
   if kind=='write' and a['task']==0:
    canonical={k:v.copy() for k,v in state.items()};canonical['W1']=state['W1'][np.argsort(a['permutation'])].copy()
    directory=OUT/'real_bridge/programs';directory.mkdir(exist_ok=True,parents=True);dest=directory/f'{key}.npz';np.savez(dest,**canonical)
    bridge.append({'id':key,'path':str(dest.relative_to(ROOT)),'valid':valid})
 save(OUT/'real_bridge/frozen_programs.json',bridge)
 files=[p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'empty_workdir' not in p.parts]
 files+=[ROOT/n for n in ('cold_wsp_protocol_v8.json','cold_wsp_amendments_v8.json','wsprec/cold_study.py','wsprec/cold_benchmark.py','wsprec/cold_real_bridge.py','wsprec/mit_wsp_harness.py','wsprec/cold_certify.py')]
 files.append(ROOT/'scripts/certify_cold_v8.py')
 save(OUT/'frozen_manifest.json',{'cases':selected,'sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}})
 print('Frozen',digest(OUT/'frozen_manifest.json'),flush=True)


def final():
 from .cold_benchmark import evaluation_slice,forward,oracle,veto_mask,ranking_metrics,compiled_network,fold_known_status
 frozen=_load(OUT/'frozen_manifest.json')
 for path,h in frozen['sha256'].items():assert digest(ROOT/path)==h,path
 with (OUT/'test_opened.lock').open('x') as f:f.write('one final reporting pass')
 slices={};results={};baselines={}
 def evaluate(state,task,perm=None):
  out={}
  for name in ('warm_warm','cold_user','cold_item','both_cold'):
   if (task,name) not in slices:slices[task,name]=evaluation_slice(task,name)
   x,y,mask=slices[task,name];prediction=forward(state,x if perm is None else x[...,perm]);out[name]=ranking_metrics(prediction,y,mask,veto_mask(x,task))
   error=np.abs(prediction-y)[mask]
   out[name]['utility_error']={'mean_absolute':float(error.mean()),'max_absolute':float(error.max()),'root_mean_squared':float(np.sqrt(np.mean(error**2))),'evaluated_pairs':int(error.size)}
  return out
 for key,a in frozen['cases'].items():
  state=_state(ROOT/a['path']);task=a['task'];entry={'assignment':a,'metrics':evaluate(state,task,a.get('permutation'))}
  if a['kind']=='write':
   x,y,mask=slices[task,'both_cold'];ablations={}
   for mode in ('user_missing','item_missing','both_missing','misaligned'):
    modified=x.copy()
    if mode in ('user_missing','both_missing'):modified[...,[0,1,2,3,8,10,12,14,16,18]]=0;modified[...,14]=1
    if mode in ('item_missing','both_missing'):modified[...,[4,5,6,7,9,11,13,15,17,19]]=0
    if mode=='misaligned':modified=modified[...,np.random.default_rng(2026110899).permutation(22)]
    ablations[mode]=ranking_metrics(forward(state,modified[...,a['permutation']]),y,mask,veto_mask(x,task))
   entry['information_ablations']=ablations
  else:
   source=_state(ROOT/a['source']);x,_,mask=slices[task,'warm_warm'];difference=np.abs(forward(state,x)-forward(source,x))[mask]
   entry['warm_prediction_locality']={'mean_absolute_change':float(difference.mean()),'max_absolute_change':float(difference.max())}
   only={k:v.copy() for k,v in source.items()};only['W1'][20:22]=state['W1'][20:22];only['b1']=state['b1'].copy()
   rollback={k:v.copy() for k,v in state.items()};rollback['W1'][20:22]=source['W1'][20:22];rollback['b1']=source['b1'].copy()
   entry['status_block_only']=evaluate(only,task);entry['status_block_rollback']=evaluate(rollback,task)
   entry['changed_parameters']=sum(int(np.count_nonzero(state[k]!=source[k])) for k in state)
  results[key]=entry
 for task in range(3):
  baselines[f'compiler-task{task}']=evaluate(compiled_network(task),task);baselines[f'zero-task{task}']=evaluate(empty_network(),task)
  random={k:np.random.default_rng(827+task+i).normal(0,.05,size=v.shape).astype(np.float32) for i,(k,v) in enumerate(empty_network().items())};baselines[f'random-task{task}']=evaluate(random,task)
  for trial in _load(OUT/f'bp/task{task}/training_summary.json')['trials']:
   key=f"task{task}-n{trial['train_pairs']}-s{trial['seed']}-{trial['condition']}";source=_state(ROOT/trial['checkpoint']);baselines[key]={'training':trial,'metrics':evaluate(source,task)}
   if task==1 and trial['train_pairs']==16384 and trial['condition']=='warm_only':baselines[key+'-fixed_fold']={'metrics':evaluate(fold_known_status(source),task)}
 save(OUT/'final_results.json',{'cases':results,'baselines':baselines,'frozen_manifest_sha256':digest(OUT/'frozen_manifest.json')})
 from .cold_real_bridge import final as bridge_final
 bridge_final(_load(OUT/'real_bridge/frozen_programs.json'),OUT/'frozen_manifest.json')
 print('Completed final synthetic and real bridge scoring',flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare_writes','run_writes','prepare_repairs','run_repairs','freeze','final']);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
 globals()[a.command](a.workers) if a.command.startswith('run_') else globals()[a.command]()
