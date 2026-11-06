"""Export the frozen two-system comparison, evidence and web data; no evaluation."""
import csv,json,hashlib,shutil,sys,zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wsprec.train_aware_study import ROOT,OUT,load,digest
REPORT=ROOT/'reports/v4_train_aware'
def main():
    r=load(OUT/'final_results.json');manifest=load(OUT/'frozen_manifest.json');assign=load(OUT/'assignments.json');REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'evidence').mkdir(exist_ok=True)
    for name in ('assignments.json','baseline_provenance.json','frozen_manifest.json'):
        shutil.copy2(OUT/name,REPORT/'evidence'/name)
    prompts={p.name:p.read_text() for p in (OUT/'prompts').glob('*.txt')};(REPORT/'evidence/prompts.json').write_text(json.dumps(prompts,indent=2))
    evaluations={k:[load(OUT/'evaluations'/k/f'round{n}.json') for n in (1,2)] for k in assign};(REPORT/'evidence/evaluations.json').write_text(json.dumps(evaluations,indent=2))
    metrics=('ndcg@10','recall@10','hit_rate@10');mean={'bp':{k:float(np.mean([v['summary'][k] for v in r['base'].values()])) for k in metrics},'astra':{k:float(np.mean([v['metrics']['summary'][k] for v in r['astra'].values()])) for k in metrics}};rows=[]
    for key,value in r['astra'].items():
        choice=manifest['choices'][key];ops=None if choice is None else [{'parameter':op['parameter'],'op':op['op'],'scalar':op.get('value') if isinstance(op.get('value'),(int,float)) else None} for op in choice['patch']['operations']]
        rows.append({'key':key,'seed':value['seed'],'replicate':value['replicate'],'bp':r['base'][str(value['seed'])]['summary'],'astra':value['metrics']['summary'],'selected_id':value['selected_id'],'operations':ops,'paired_bootstrap':value['paired_bootstrap']})
    flat=[{'seed':row['seed'],'replicate':row['replicate'],'selected_id':row['selected_id'],**{'bp_'+k:v for k,v in row['bp'].items()},**{'astra_'+k:v for k,v in row['astra'].items()}} for row in rows]
    with (REPORT/'test_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    slots=[x for rounds in evaluations.values() for b in rounds for x in b['results']];costs={'new_bp_training_runs':0,'reused_bp_checkpoints':3,'preparation_baseline_validation_evaluations':3,'freeze_baseline_validation_evaluations':6,'final_test_metric_evaluations':9,'editor_trajectories':6,'recorded_editor_generation_turns':len([k for k in prompts if k.endswith('.dispatch.txt')]),'candidate_queries':len(slots),'invalid_candidates':sum(x['error'] is not None for x in slots),'candidate_execution_wall_seconds':sum(b['wall_seconds'] for rounds in evaluations.values() for b in rounds),'integrity_replay':manifest['replay'],'test_reporting_wall_seconds':r['wall_seconds'],'editor_input_tokens':None,'editor_output_tokens':None,'editor_billed_usd':None,'editor_training_data_analysis_wall_seconds':None,'scope':'Current train-aware run only. Reused BP training costs in reports/v3_neutral/costs.json; prior studies not free. Native billing and complete agent compute telemetry unavailable, not zero.'}
    result={'comparison':'BP-trained BPR vs same checkpoint edited by training-aware weight-visible Astra','mean':mean,'rows':rows,'improved':sum(x['astra']['ndcg@10']>x['bp']['ndcg@10'] for x in rows),'costs':costs,'manifest_sha256':r['manifest_sha256'],'limits':load(ROOT/'neutral_protocol_v4.json')['limits']}
    (REPORT/'summary.json').write_text(json.dumps(result,indent=2));(REPORT/'costs.json').write_text(json.dumps(costs,indent=2))
    index={};seen=set()
    with zipfile.ZipFile(REPORT/'numeric_patch_assets.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (OUT/'evaluations').glob('**/*.npy'):
            h=digest(p);name='objects/'+h+'.npy';index[str(p.relative_to(ROOT))]=name
            if h not in seen:z.write(p,name);seen.add(h)
        z.writestr('index.json',json.dumps(index,indent=2))
    (ROOT/'web/explainer/data.js').write_text('window.TRAIN_AWARE = '+json.dumps(result,ensure_ascii=False)+';\n')
    print(json.dumps({'mean':mean,'improved':result['improved'],'costs':costs},indent=2))
if __name__=='__main__':main()
