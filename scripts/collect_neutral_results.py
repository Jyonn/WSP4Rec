"""Collect completed, frozen results and replayable numeric assets."""
import csv
import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/os.environ.get('WSPREC_RUN_DIR','runs/v3_neutral')
DATA=ROOT/os.environ.get('WSPREC_DATA_DIR','data/ml-100k-description-complete-v3')
REPORT=ROOT/os.environ.get('WSPREC_REPORT_DIR','reports/v3_neutral')
SEEDS=('42','43','44')


def main():
    result=json.loads((OUT/'final_results.json').read_text())
    manifest=json.loads((OUT/'frozen_manifest.json').read_text())
    assignments=json.loads((OUT/'editor_assignments.json').read_text())
    training=json.loads((OUT/'training_summary.json').read_text())
    integrity=json.loads((OUT/'consistency_audit.json').read_text())
    prior_path=OUT/'consistency_audit.superseded.json'
    prior_integrity=json.loads(prior_path.read_text()) if prior_path.exists() else {'replayed_scored_patches':0,'wall_seconds':0}
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'evidence').mkdir(exist_ok=True)
    for source,name in ((OUT/'frozen_manifest.json','frozen_manifest.json'),(OUT/'training_summary.json','training_summary.json'),
        (DATA/'manifest.json','dataset_manifest.json'),(OUT/'editor_assignments.json','editor_assignments.json'),
        (OUT/'consistency_audit.json','consistency_audit.json')):
        shutil.copy2(source,REPORT/'evidence'/name)
    all_evaluations={};raw={};operator_counts=Counter();parameter_counts=Counter();first_parameter_counts=Counter()
    for key in manifest['choices']:
        all_evaluations[key]=[json.loads((OUT/'evaluations'/key/f'round{r}.json').read_text()) for r in (1,2)]
        if key in assignments:
            raw[key]=[json.loads((OUT/'editors'/key/f'round{r}.json').read_text()) for r in (1,2)]
            for index,batch in enumerate(raw[key]):
                if not isinstance(batch,list):continue
                for patch in batch:
                    if not isinstance(patch,dict):continue
                    for op in patch.get('operations',[]):
                        operator_counts[op.get('op','unknown')]+=1;parameter_counts[op.get('parameter','unknown')]+=1
                        if index==0:first_parameter_counts[op.get('parameter','unknown')]+=1
    (REPORT/'evidence/evaluations.json').write_text(json.dumps(all_evaluations,indent=2))
    (REPORT/'evidence/submissions.json').write_text(json.dumps(raw,indent=2))
    prompts={p.name:p.read_text() for p in sorted((OUT/'prompts').glob('*.txt'))}
    (REPORT/'evidence/prompts.json').write_text(json.dumps(prompts,indent=2))
    forbidden=('popularity','debias','calibrat','support_shrink','neighbor_smooth')
    audit={'initial_tasks':len(assignments),'saved_generation_dispatches':sum(name.endswith('.dispatch.txt') for name in prompts),
        'directional_lexicon_absent_from_instructions':not any(word in text.lower() for text in prompts.values() for word in forbidden),
        'hypotheses_or_rationales_requested':False,'schema':'id and operations only','domain_specific_operations':False,
        'all_submitted_patch_fields':sorted(set(k for batches in raw.values() for batch in batches if isinstance(batch,list) for p in batch if isinstance(p,dict) for k in p)),
        'isolation':'fresh fork-none native subagents with file allowlists; not an OS-level isolated filesystem',
        'scope':'audit of supplied instructions and saved outputs, not a claim about private reasoning or absence of pretrained priors'}
    (REPORT/'prompt_audit.json').write_text(json.dumps(audit,indent=2))
    object_paths={};index={};seen=set()
    with zipfile.ZipFile(REPORT/'numeric_patch_assets.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        asset_paths=list((OUT/'evaluations').glob('**/snapshots/**/*.npy'))
        asset_paths+=list((OUT/'editors').glob('**/*.npy'))
        for p in sorted(asset_paths):
            if p.stat().st_size>20971520 or not p.resolve().is_relative_to(OUT.resolve()):continue
            sha=hashlib.sha256(p.read_bytes()).hexdigest();logical=str(p.relative_to(ROOT));name='objects/'+sha+'.npy'
            index[logical]=name
            if sha not in seen:archive.write(p,name);seen.add(sha)
        archive.writestr('index.json',json.dumps(index,indent=2))
    numeric={'logical_assets':len(index),'unique_assets':len(seen),'compressed_bytes':(REPORT/'numeric_patch_assets.zip').stat().st_size}
    aggregates={};rows=[];group_rows=[]
    for obj in ('mse','bpr'):
        aggregates[obj]={'base':{k:float(np.mean([result['objectives'][obj][s]['base']['metrics']['summary'][k] for s in SEEDS])) for k in ('ndcg@10','recall@10','hit_rate@10')}}
        for text in ('none','title','description'):
            for visibility in ('visible','blind'):
                rep_values=[];recalls=[]
                for rep in (0,1):
                    key=f'{obj}-{text}-{visibility}-r{rep}'
                    rep_values.append(float(np.mean([result['objectives'][obj][s][key]['metrics']['summary']['ndcg@10'] for s in SEEDS])))
                    recalls.append(float(np.mean([result['objectives'][obj][s][key]['metrics']['summary']['recall@10'] for s in SEEDS])))
                aggregates[obj][f'{text}-{visibility}']={'ndcg@10':float(np.mean(rep_values)),'editor_rep_means_ndcg':rep_values,'recall@10':float(np.mean(recalls))}
        for s,methods in result['objectives'][obj].items():
            for method,value in methods.items():
                rows.append({'objective':obj,'seed':s,'method':method,**value['metrics']['summary'],
                    'selected_id':value.get('selected_id'),'transfer_error':value.get('transfer_error'),
                    'improved_fraction':value.get('paired_bootstrap_vs_base',{}).get('improved_fraction'),
                    'harmed_fraction':value.get('paired_bootstrap_vs_base',{}).get('harmed_fraction')})
                for ug,bins in value['metrics']['user_item_groups'].items():
                    for ig,metrics in bins.items():group_rows.append({'objective':obj,'seed':s,'method':method,'user_group':ug,'item_group':ig,**metrics})
    for name,records in (('test_results.csv',rows),('user_item_groups.csv',group_rows)):
        with (REPORT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    editor_results=[r for key in assignments for d in all_evaluations[key] for r in d['results']]
    control_results=[r for key in all_evaluations if key not in assignments for d in all_evaluations[key] for r in d['results']]
    costs={'training_trials':len(training['trials']),'training_wall_seconds':sum(t['wall_seconds'] for t in training['trials']),
        'training_validation_evaluations':sum(t['epochs_run'] for t in training['trials']),
        'editor_trajectories':len(assignments),'editor_generation_turns':audit['saved_generation_dispatches'],
        'editor_candidate_queries':len(editor_results),'invalid_editor_candidates':sum(r['error'] is not None for r in editor_results),
        'invalid_by_trajectory':{key:sum(r['error'] is not None for batch in all_evaluations[key] for r in batch['results']) for key in assignments},
        'invalid_reasons':dict(Counter(r.get('audit_error','unknown') for r in editor_results if r['error'] is not None)),
        'control_candidate_queries':len(control_results),'invalid_control_candidates':sum(r['error'] is not None for r in control_results),
        'candidate_execution_wall_seconds':sum(d['wall_seconds'] for v in all_evaluations.values() for d in v),
        'test_reporting_wall_seconds':result['wall_seconds'],'new_embedding_gpu_seconds':0,
        'reference_validation_evaluations':len(json.loads((OUT/'references/validation.json').read_text())['results']),
        'ease_reference_fit_and_validation_wall_seconds':json.loads((OUT/'references/validation.json').read_text())['wall_seconds'],
        'unmetered_compute':['dataset preparation','control asset construction','editor numeric inspection and asset construction','popular-reference and Gram construction'],
        'integrity_replay_evaluations':integrity['replayed_scored_patches']+prior_integrity['replayed_scored_patches'],
        'integrity_replay_wall_seconds':integrity['wall_seconds']+prior_integrity['wall_seconds'],
        'integrity_passes':2 if prior_path.exists() else 1,
        'text_cost_scope':'reuse individually encoded Llama-3 features by metadata filtering/reindexing; prior encoding cost not zero and recorded in reports/followup',
        'editor_input_tokens':None,'editor_output_tokens':None,'editor_billed_usd':None,'editor_total_inference_wall_seconds':None,
        'timing_scope':'inside-operation walls, not end-to-end development/coordination cost','numeric_assets':numeric}
    prior_costs=ROOT/'reports/v2_neutral/costs.json'
    if OUT.name=='v3_neutral' and prior_costs.exists():
        previous=json.loads(prior_costs.read_text())
        costs['superseded_intermediate']=previous
        costs['including_intermediate']={k:costs[k]+previous[k] for k in ('training_trials','training_wall_seconds','editor_trajectories','editor_generation_turns','editor_candidate_queries','control_candidate_queries','candidate_execution_wall_seconds','integrity_replay_evaluations','integrity_replay_wall_seconds','test_reporting_wall_seconds','ease_reference_fit_and_validation_wall_seconds','reference_validation_evaluations')}
    notes=OUT/'execution_notes.json'
    if notes.exists():
        costs['execution_notes']=json.loads(notes.read_text())
        shutil.copy2(notes,REPORT/'evidence/execution_notes.json')
    (REPORT/'costs.json').write_text(json.dumps(costs,indent=2))
    summary={'dataset':json.loads((REPORT/'evidence/dataset_manifest.json').read_text()),'aggregates':aggregates,
        'references':{k:v['summary'] for k,v in result['references'].items()},'costs':costs,'prompt_audit':audit,
        'operator_counts':dict(operator_counts),'parameter_counts':dict(parameter_counts),'first_round_parameter_counts':dict(first_parameter_counts),
        'no_op_trajectories':[k for k in assignments if manifest['choices'][k]['selection'] is None],
        'transfer_invalid_cases':[{'objective':obj,'seed':seed,'trajectory':key,'error':value['transfer_error']} for obj,by_seed in result['objectives'].items() for seed,methods in by_seed.items() for key,value in methods.items() if value.get('transfer_error')],
        'manifest_sha256':result['manifest_sha256'],'same_corpus_scope':'new filtered/split cohort, not untouched external test data',
        'editor_independence_scope':'two independent editor runs; each frozen policy transfers across three training seeds; six scores are not six independent editor runs'}
    intervals={obj:{seed:{key:value['paired_bootstrap_vs_base'] for key,value in methods.items() if 'paired_bootstrap_vs_base' in value} for seed,methods in by_seed.items()} for obj,by_seed in result['objectives'].items()}
    (REPORT/'paired_user_intervals.json').write_text(json.dumps({'scope':'paired users conditional on a fixed trained model and selected policy; not inference over editor calls','results':intervals},indent=2))
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'aggregates':aggregates,'costs':costs,'prompt_audit':audit},indent=2))


if __name__=='__main__':main()
