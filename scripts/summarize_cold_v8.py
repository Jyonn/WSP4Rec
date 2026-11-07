"""Report-data aggregation only; never fits or executes a research network."""
import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/v8_mit';REPORT=ROOT/'reports/v8_mit'
METRICS=('ndcg@10','top1_normalized_regret','veto_exposure@10')
def load(p):return json.loads(Path(p).read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvsave(name,rows):
 if not rows:return
 columns=list(dict.fromkeys(k for row in rows for k in row))
 with (REPORT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');w.writeheader();w.writerows(rows)
def means(rows,keys,metrics=METRICS):
 groups=defaultdict(list)
 for row in rows:groups[tuple(row.get(k) for k in keys)].append(row)
 result=[]
 for group,items in groups.items():
  out=dict(zip(keys,group));out['n_records']=len(items)
  for metric in metrics:
   values=[r[metric] for r in items if r.get(metric) is not None]
   out[metric]=float(np.mean(values)) if values else None
  result.append(out)
 return result
def paired(vectors,seed=20261108):
 # Average replicate-wise deltas per user first. Call separately per task/seed.
 a=np.mean(np.asarray(vectors,dtype=float),axis=0);rng=np.random.default_rng(seed)
 boot=np.array([np.mean(a[rng.integers(len(a),size=len(a))]) for _ in range(10000)])
 return {'mean_delta':float(a.mean()),'lower95':float(np.quantile(boot,.025)),'upper95':float(np.quantile(boot,.975)),'users':len(a),'scope':'Conditional paired-user bootstrap after averaging editor replicates within THIS task/training seed. Does not estimate task, training or editor population uncertainty.'}
def main():
 REPORT.mkdir(exist_ok=True,parents=True)
 final=load(OUT/'final_results.json');bridge=load(OUT/'real_bridge/final_results.json');cases=final['cases'];case_rows=[];ablations=[];locality=[];baseline_rows=[];cis={}
 for key,entry in cases.items():
  a=entry['assignment']
  for slice_name,m in entry['metrics'].items():
   row={'case':key,'kind':a['kind'],'task':a['task'],'encoding':a.get('encoding'),'seed':a.get('seed'),'replicate':a['replicate'],'extra_forward_budget':a.get('extra_forward_budget'),'valid':a['valid'],'slice':slice_name,**{k:m.get(k) for k in METRICS},'ranking_users':m['ranking_users']}
   row.update({f'utility_{k}':v for k,v in m.get('utility_error',{}).items()});case_rows.append(row)
  for mode,m in entry.get('information_ablations',{}).items():ablations.append({'case':key,'task':a['task'],'encoding':a.get('encoding'),'mode':mode,**{k:m.get(k) for k in METRICS}})
  if a['kind']=='repair':
   locality.append({'case':key,'seed':a['seed'],'replicate':a['replicate'],'extra_forward_budget':a['extra_forward_budget'],'valid':a['valid'],'changed_parameters':entry['changed_parameters'],**entry['warm_prediction_locality']})
   for mode in ('status_block_only','status_block_rollback'):
    for slice_name,m in entry[mode].items():ablations.append({'case':key,'task':a['task'],'seed':a['seed'],'extra_forward_budget':a['extra_forward_budget'],'mode':mode,'slice':slice_name,**{k:m.get(k) for k in METRICS}})
 for name,entry in final['baselines'].items():
  metadata=entry.get('training',{});metrics=entry.get('metrics',entry)
  for slice_name,m in metrics.items():baseline_rows.append({'baseline':name,'task':metadata.get('task',int(name.split('task')[1].split('-')[0])),'train_pairs':metadata.get('train_pairs'),'seed':metadata.get('seed'),'condition':metadata.get('condition',name.split('-')[0]),'slice':slice_name,**{k:m.get(k) for k in METRICS},'ranking_users':m['ranking_users']})
 # Repair user-paired differences: each task1 training seed is kept separate.
 for seed in (42,43,44):
  reference=final['baselines'][f'task1-n16384-s{seed}-warm_only']['metrics']
  for budget in (0,2):
   entries=[e for e in cases.values() if e['assignment']['kind']=='repair' and e['assignment']['seed']==seed and e['assignment']['extra_forward_budget']==budget]
   for slice_name in ('warm_warm','both_cold'):
    base=reference[slice_name]['per_user'];vectors=[]
    for entry in entries:
     own=entry['metrics'][slice_name]['per_user'];assert own['user_id']==base['user_id'];vectors.append(np.array(own['ndcg'])-np.array(base['ndcg']))
    cis[f'repair-seed{seed}-budget{budget}-{slice_name}']=paired(vectors)
 # Writing-vs-compiler remains task-specific; do not pool three worlds as datasets.
 for task in range(3):
  base=final['baselines'][f'compiler-task{task}']['both_cold']['per_user']
  for encoding in ('natural','nonce'):
   entries=[e for e in cases.values() if e['assignment']['kind']=='write' and e['assignment']['task']==task and e['assignment']['encoding']==encoding];vectors=[]
   for entry in entries:
    own=entry['metrics']['both_cold']['per_user'];assert own['user_id']==base['user_id'];vectors.append(np.array(own['ndcg'])-np.array(base['ndcg']))
   cis[f'write-task{task}-{encoding}-minus-compiler']=paired(vectors)
 bridge_rows=[]
 for k,record in bridge['budgets'].items():
  for name,m in record['metrics'].items():bridge_rows.append({'support_K':int(k),'model':name,'ndcg@10':m['ndcg@10'],'recall@10':m['recall@10'],'ranking_users':m['ranking_users'],'candidate_items':record['candidate_items']})
  metrics=record['metrics'];programs=[n for n in metrics if n.startswith('w')]
  for reference in ('fixed_symbolic_task0','genre_profile_cosine','global_genre_positive_rate'):
   vectors=[];base=metrics[reference]['per_user']
   for name in programs:
    own=metrics[name]['per_user'];assert own['user_id']==base['user_id'];vectors.append(np.array(own['ndcg'])-np.array(base['ndcg']))
   if vectors:cis[f'real-K{k}-astra-mean-minus-{reference}']=paired(vectors)
  vectors=[]
  for name in programs:
   own=metrics[name]['per_user']
   for seed in (42,43,44):
    base=metrics[f'bp-s{seed}']['per_user'];assert own['user_id']==base['user_id'];vectors.append(np.array(own['ndcg'])-np.array(base['ndcg']))
  if vectors:cis[f'real-K{k}-astra-mean-minus-bp-mean']=paired(vectors)
 csvsave('per_case.csv',case_rows);csvsave('ablations.csv',ablations);csvsave('repair_locality.csv',locality);csvsave('baselines.csv',baseline_rows);csvsave('real_bridge.csv',bridge_rows)
 write=means([r for r in case_rows if r['kind']=='write'],('task','encoding','slice'))
 repair_seed=means([r for r in case_rows if r['kind']=='repair'],('seed','extra_forward_budget','slice'))
 # Equal two realizations per seed; seed-level means are then averaged.
 repair_budget=means(repair_seed,('extra_forward_budget','slice'))
 bp=means([r for r in baseline_rows if r['train_pairs'] is not None and '-fixed_fold' not in r['baseline']],('task','train_pairs','condition','slice'))
 csvsave('write_task_means.csv',write);csvsave('repair_seed_means.csv',repair_seed);csvsave('repair_budget_means.csv',repair_budget);csvsave('bp_data_efficiency.csv',bp)
 save(REPORT/'paired_user_intervals.json',cis)
 save(REPORT/'summary.json',{'write_task_means':write,'repair_seed_means':repair_seed,'repair_budget_means':repair_budget,'bp_data_efficiency':bp,'invalid_cases':[k for k,e in cases.items() if not e['assignment']['valid']],'invalid_real_bridge_programs':bridge['invalid_frozen_programs'],'frozen_manifest_sha256':final['frozen_manifest_sha256'],'interpretation':'Synthetic explicit-policy compilation and constructed nuisance repair; three worlds are not independent real datasets. Invalid synthetic programs remain operational zero/source fallbacks, not omitted. Real bridge excludes invalid programs and lists them explicitly; do not average only valid cases without stating this.'})
 print('Report tables saved without running models',flush=True)
if __name__=='__main__':main()
