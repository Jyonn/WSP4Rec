"""Aggregate text integration and raw-description access without test selection."""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wsprec.text_study import ROOT,OUT,SEEDS,ACCESS,load,save
R=ROOT/'reports/v7_text'
METHODS=('bp','fixed_simple','fixed_simple_raw','tuned_simple_8','simple_calibration_8','astra','astra_first_patch')


def bootstrap(vectors):
 a=np.mean(np.asarray(vectors,dtype=float),axis=0);rng=np.random.default_rng(202611072)
 draws=np.array([a[rng.integers(len(a),size=len(a))].mean() for _ in range(10000)])
 return {'mean_delta':float(a.mean()),'lower95':float(np.quantile(draws,.025)),'upper95':float(np.quantile(draws,.975)),'users':len(a),'scope':'Users resampled after averaging editor realizations within each seed, then seeds. Conditional on these models, editing realizations and historical split; not training/editor population inference.'}


def write_csv(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
 R.mkdir(parents=True,exist_ok=True);final=load(OUT/'final_results.json');frozen=load(OUT/'frozen_manifest.json');assign=load(OUT/'assignments.json')
 cases=final['cases'];rows=[]
 for key,c in cases.items():
  selected=frozen['choices'][key]['astra']
  for method in METHODS:
   m=c['metrics'][method]
   if m is None:continue
   rows.append({'case':key,'seed':c['seed'],'replicate':c['replicate'],'access':c['access'],'model_condition':c['model_condition'],'method':method,**{x:m['summary'][x] for x in ('ndcg@10','recall@10')},'selected_id':c['selected_id'] if method=='astra' else '', 'relative_delta_l2':selected['relative_delta_l2'] if method=='astra' and selected else 0 if method=='astra' else ''})
 write_csv(R/'per_case.csv',rows)
 means=[]
 for access in ACCESS:
  for method in METHODS:
   perseed=[{m:float(np.mean([r[m] for r in rows if r['access']==access and r['method']==method and r['seed']==seed])) for m in ('ndcg@10','recall@10')} for seed in SEEDS]
   means.append({'access':access,'method':method,**{m:float(np.mean([s[m] for s in perseed])) for m in ('ndcg@10','recall@10')}})
 write_csv(R/'condition_means.csv',means)
 def vector(access,method,seed,metric='ndcg'):
  data=[c['metrics'][method]['per_user'] for c in cases.values() if c['access']==access and c['seed']==seed]
  assert len(data)==2 and data[0]['user_id']==data[1]['user_id']
  return np.mean([np.array(x[metric]) for x in data],axis=0)
 comparisons={
 'text_bp_minus_none_bp':('full_text','bp','none','bp'),
 'full_text_astra_minus_numeric_text_astra':('full_text','astra','numeric_text','astra'),
 'full_text_astra_minus_text_bp':('full_text','astra','full_text','bp'),
 'full_text_astra_minus_text_fixed':('full_text','astra','full_text','fixed_simple'),
 'full_text_astra_minus_text_grid':('full_text','astra','full_text','tuned_simple_8'),
 'full_text_astra_minus_calibration':('full_text','astra','full_text','simple_calibration_8'),
 'numeric_text_astra_minus_text_bp':('numeric_text','astra','numeric_text','bp'),
 'none_astra_minus_none_bp':('none','astra','none','bp')}
 intervals={}
 for name,(a,b,c,d) in comparisons.items():intervals[name]=bootstrap([vector(a,b,seed)-vector(c,d,seed) for seed in SEEDS])
 intervals['text_bp_minus_shuffled_bp']=bootstrap([vector('full_text','bp',seed)-np.array(final['bp_only_shuffled_text'][str(seed)]['per_user']['ndcg']) for seed in SEEDS])
 intervals['editing_gain_text_minus_none']=bootstrap([(vector('full_text','astra',seed)-vector('full_text','bp',seed))-(vector('none','astra',seed)-vector('none','bp',seed)) for seed in SEEDS])
 save(R/'paired_user_intervals.json',intervals)
 groups=[]
 for key,c in cases.items():
  for method in ('bp','fixed_simple','astra'):
   for group,metrics in c['metrics'][method]['item_groups'].items():
    groups.append({'case':key,'seed':c['seed'],'replicate':c['replicate'],'access':c['access'],'method':method,'item_group':group,**{k:metrics[k] for k in ('ndcg@10','recall@10','ranking_users','positives')}})
 write_csv(R/'item_groups.csv',groups)
 alltrials=[]
 for condition in ('none','text','shuffled_text'):
  ts=load(OUT/condition/'training_summary.json');alltrials+=ts['trials']
  save(R/f'{condition}_training_summary.json',ts)
 controlpaths=sorted(set(a['controls'] for a in assign.values()));evals=[load(OUT/'evaluations'/key/f'round{rnd}.json') for key in assign for rnd in (1,2)]
 costs={'bp_training_trials':len(alltrials),'sum_training_trial_wall_seconds':sum(t['wall_seconds'] for t in alltrials),'max_training_trial_wall_seconds':max(t['wall_seconds'] for t in alltrials),
 'bp_validation_epoch_scores':sum(len(t['history']) for t in alltrials),'server_training_jobs':load(OUT/'training_jobs.json'),'server_environment':load(OUT/'server_environment.json'),
 'editor_candidates':sum(len(e['results']) for e in evals),'valid_editor_candidates':sum(r['error'] is None for e in evals for r in e['results']), 'editor_validation_wall_seconds':sum(e['wall_seconds'] for e in evals),
 'exploratory_calibration':load(OUT/'calibration/generation.json'),'control_checkpoint_count':len(controlpaths),'control_candidate_scores':sum(len(load(ROOT/p)['records']) for p in controlpaths),
 'preparation_wall_seconds':load(OUT/'preparation.json')['wall_seconds'],'freeze_wall_seconds':frozen['wall_seconds'],'test_reporting_wall_seconds':final['wall_seconds'],
 'new_llama_encoding_calls':0,'historical_llama_encoding_metadata':'reports/followup/llama3_embedding_metadata.json',
 'pca_preprocessing':load(OUT/'text/metadata.json'),'native':load(R/'native_calls.json'),
 'limits':'Sum trial wall is concurrent job work, not total elapsed nor pure GPU kernel time; validation runs CPU. Native per-request tokens and billedUSD unavailable. Goal counter separate from per-call bills. Query count matching is not wall/compute matching.'}
 if (R/'goal_usage.json').exists():costs['goal_tool_usage']=load(R/'goal_usage.json')
 save(R/'costs.json',costs)
 summary=load(R/'summary.json');summary.update(condition_means=means,comparisons=intervals,frozen_manifest_sha256=final['frozen_manifest_sha256']);save(R/'summary.json',summary)
 print(json.dumps(means,indent=2))

if __name__=='__main__':main()
