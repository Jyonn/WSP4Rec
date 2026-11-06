"""BP checkpoint versus training-aware, weight-visible Astra editing."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/ml-100k-description-complete-v3'
from .study import scores,quick_metric,save,checkpoint
from .study_metrics import rank_metrics,paired_bootstrap
from .tensor_edits import compile_patch,_asset_array
OUT=ROOT/'runs/v4_train_aware'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def model(seed):return checkpoint(OUT/f'inputs/bpr-s{seed}.pt')['state']
def data():return np.load(DATA/'train.npy'),np.load(DATA/'valid.npy')
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x

def prepare():
    if (OUT/'assignments.json').exists():raise FileExistsError('already prepared')
    inp=OUT/'inputs';inp.mkdir(parents=True,exist_ok=True)
    train,valid=data();shutil.copy2(DATA/'train.npy',inp/'train.npy')
    count=load(DATA/'manifest.json')['counts']['after'];users,items=count['users'],count['items']
    save(inp/'id_to_embedding_row.json',{'item_id_to_item_embedding_row':list(range(items)),'user_id_to_user_embedding_row':list(range(users)),
        'rule':'IDs in train.npy are zero-based and directly index corresponding embedding and bias rows. No permutation is hidden from you.'})
    interface=load(ROOT/'editor_interface_v2.json');interface['input']='Full learned model parameters, complete training interactions and explicit ID-to-embedding-row mapping.'
    interface['computation']='Training-data inspection, training-set forward predictions/residuals, direct algebra and one-shot analytic updates based on training data are permitted. No backpropagation, gradients, gradient optimizers, iterative model training, automated candidate/parameter search, validation/test data access or internet. Validation scores are returned only by the evaluator.'
    interface['visibility']='All recommendation-model parameters and all training interactions are visible. No external movie text or text vectors are supplied. Read only your allowlist and your own saved submissions/feedback.'
    save(inp/'interface.json',interface)
    assignments={}
    previous=load(ROOT/'runs/v3_neutral/training_summary.json')
    for seed in (42,43,44):
        src=ROOT/'runs/v3_neutral/training'/previous['selected']['bpr'][str(seed)]/'best.pt';shutil.copy2(src,inp/f'bpr-s{seed}.pt')
        base=model(seed);np.savez(inp/f'bpr-s{seed}-parameters.npz',**{k:v.numpy() for k,v in base.items()})
        save(inp/f'bpr-s{seed}-spec.json',{'architecture':'16-dimensional biased matrix factorization, trained with BPR using backpropagation',
            'score':'bias + user_bias.weight[u,0] + item_bias.weight[i,0] + sum_d user.weight[u,d]*item.weight[i,d]',
            'parameters':{k:list(v.shape) for k,v in base.items()},'training_array':{'file':'train.npy','columns':['user_id','item_id','rating','timestamp'],'rows':len(train),'positive_rating_threshold':4},
            'baseline_valid_ndcg@10':quick_metric(scores(base),valid,train),'user_count':users,'item_count':items})
        for rep in (0,1):
            key=f'bpr-s{seed}-r{rep}';folder=OUT/'editors'/key;folder.mkdir(parents=True,exist_ok=True)
            allowed=[str((inp/f).relative_to(ROOT)) for f in ('interface.json','train.npy','id_to_embedding_row.json',f'bpr-s{seed}-parameters.npz',f'bpr-s{seed}-spec.json')]
            assignments[key]={'seed':seed,'replicate':rep,'allowlist':allowed,'output_directory':str(folder.relative_to(ROOT))}
            prompt=('Edit the provided fixed recommendation model to improve validation NDCG@10. Read only this instruction file, the allowlist below and your own previous submissions/feedback. '
                'You can see all learned parameter values and the complete training set. The mapping explicitly identifies every training item ID and user ID with its parameter row. '
                'You may compute training-set predictions, residuals and statistics and use direct algebra or one-shot analytic updates. '
                'Do not use backpropagation, gradients, gradient optimizers, iterative model training, automated candidate/parameter searches, internet or validation/test data. '
                'Submit exactly four distinct patches under the provided interface, all starting from the original checkpoint. After four evaluator results, you may submit one final batch of four. '
                'Do not propose hypotheses or write motivations, rationales or explanations. Save only patches and numeric assets, using unique round-specific asset names. '
                'Do not modify past submissions. Reply only with the submission file path.\n\nAllowlist:\n'+'\n'.join(allowed)+f'\n\nOutput directory: {folder.relative_to(ROOT)}\nRound-one submission: {folder.relative_to(ROOT)}/round1.json\n')
            (OUT/'prompts').mkdir(exist_ok=True);(OUT/'prompts'/f'{key}.txt').write_text(prompt)
            dispatch=f'Follow the exact instructions in runs/v4_train_aware/prompts/{key}.txt. Read only that instruction file and its allowlisted inputs, plus your own previous submissions and feedback. Reply only with the submitted file path.'
            (OUT/'prompts'/f'{key}.dispatch.txt').write_text(dispatch)
    save(OUT/'assignments.json',assignments)
    save(OUT/'baseline_provenance.json',{'source':'v3 fixed BP-trained BPR checkpoints reused without retraining','source_training_summary_sha256':digest(ROOT/'runs/v3_neutral/training_summary.json'),'dataset_manifest_sha256':digest(DATA/'manifest.json'),'existing_test_scope':'Same previously evaluated MovieLens cohort; not an untouched external test set.'})

def evaluate(key,rnd):
    a=load(OUT/'assignments.json')[key];folder=OUT/'editors'/key;dest=OUT/'evaluations'/key/f'round{rnd}.json';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError('already evaluated')
    with dest.with_suffix('.started').open('x') as f:f.write(str(time.time()))
    base=model(a['seed']);train,valid=data();start=time.monotonic()
    try:
        patches=load(folder/f'round{rnd}.json')
        if not isinstance(patches,list) or len(patches)!=4:raise ValueError('exactly four patches')
    except (ValueError,TypeError,OSError):patches=[{'id':f'invalid_{rnd}_{i}'} for i in range(4)]
    results=[];seen=set();ids=set()
    for index,patch in enumerate(patches):
        identifier=patch.get('id',str(index)) if isinstance(patch,dict) else str(index)
        try:
            if not isinstance(patch,dict) or set(patch)!={'id','operations'}:raise ValueError('patch only')
            snapshot=dest.parent/'snapshots'/f'r{rnd}-c{index}';snapshot.mkdir(parents=True,exist_ok=False)
            if not isinstance(patch['operations'],list) or not all(isinstance(op,dict) for op in patch['operations']):raise ValueError('operations')
            payload=[]
            for op in patch['operations']:
                entry={k:v for k,v in op.items() if k!='asset'}
                if 'asset' in op:
                    array=_asset_array(folder,op['asset']);target=snapshot/op['asset'];target.parent.mkdir(parents=True,exist_ok=True);np.save(target,array,allow_pickle=False);entry['asset_sha256']=digest(target)
                payload.append(entry)
            save(snapshot/'patch.json',patch);signature=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
            if signature in seen or patch['id'] in ids:raise ValueError('duplicate in batch')
            seen.add(signature);ids.add(patch['id']);edited,stats=compile_patch(base,patch,snapshot);value=quick_metric(scores(edited),valid,train)
            results.append({'id':patch['id'],'valid_ndcg@10':value,'error':None,'patch':patch,'snapshot':str(snapshot.relative_to(ROOT)),**stats})
        except (ValueError,TypeError,KeyError,OSError,RuntimeError) as e:
            results.append({'id':identifier,'valid_ndcg@10':None,'error':'invalid_patch','audit_error':str(e),'patch':patch})
    save(dest,{'results':results,'wall_seconds':time.monotonic()-start})
    feedback=[{k:v[k] for k in ('id','valid_ndcg@10','error')} for v in results];(OUT/'feedback').mkdir(exist_ok=True);save(OUT/'feedback'/f'{key}-round{rnd}.json',feedback)
    if rnd==1:
        message=f'Read runs/v4_train_aware/feedback/{key}-round1.json in addition to your original allowlist. Submit exactly four final patches to runs/v4_train_aware/editors/{key}/round2.json under the same instructions. Do not propose hypotheses or give explanations. Reply only with the submission file path.'
        (OUT/'prompts'/f'{key}-round2.dispatch.txt').write_text(message)
    print(json.dumps(feedback))

def freeze():
    if (OUT/'frozen_manifest.json').exists():raise FileExistsError('already frozen')
    assignments=load(OUT/'assignments.json');assert len(assignments)==6
    train,valid=data();choices={};replayed=0;start=time.monotonic()
    for key,a in assignments.items():
        for suffix in ('.txt','.dispatch.txt','-round2.dispatch.txt'):assert (OUT/'prompts'/f'{key}{suffix}').is_file()
        base=model(a['seed']);baseline=quick_metric(scores(base),valid,train);records=[]
        for rnd in (1,2):
            batch=load(OUT/'evaluations'/key/f'round{rnd}.json')['results'];assert len(batch)==4
            feedback=load(OUT/'feedback'/f'{key}-round{rnd}.json');assert feedback==[{k:r[k] for k in ('id','valid_ndcg@10','error')} for r in batch]
            records+=batch
            for r in batch:
                if r['error'] is None:
                    edited,_=compile_patch(base,r['patch'],ROOT/r['snapshot']);assert np.isclose(quick_metric(scores(edited),valid,train),r['valid_ndcg@10'],atol=1e-12,rtol=0);replayed+=1
        valid_records=[r for r in records if r['error'] is None and r['valid_ndcg@10']>baseline]
        choices[key]=max(valid_records,key=lambda r:r['valid_ndcg@10']) if valid_records else None
    files=[ROOT/'neutral_protocol_v4.json',ROOT/'wsprec/train_aware_study.py',ROOT/'wsprec/tensor_edits.py',ROOT/'wsprec/study.py',ROOT/'wsprec/study_metrics.py',ROOT/'wsprec/cli.py',DATA/'train.npy',DATA/'valid.npy',DATA/'splits.npz',DATA/'manifest.json']
    files+=list((OUT/'inputs').glob('*'))+list((OUT/'prompts').glob('*'))+list((OUT/'feedback').glob('*'))+list((OUT/'evaluations').rglob('*.json'))+list((OUT/'evaluations').rglob('*.npy'))+[OUT/'assignments.json',OUT/'baseline_provenance.json']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    with (OUT/'frozen_manifest.json').open('x') as f:json.dump({'choices':choices,'sha256':hashes,'replay':{'scored_patches':replayed,'wall_seconds':time.monotonic()-start}},f,indent=2)
    print('Frozen',digest(OUT/'frozen_manifest.json'))

def final():
    manifest=load(OUT/'frozen_manifest.json')
    for f,h in manifest['sha256'].items():assert digest(ROOT/f)==h,f
    with (OUT/'test_opened.lock').open('x') as f:f.write(str(time.time()))
    with np.load(DATA/'splits.npz') as z:test=z['test']
    train,valid=data();seen=np.concatenate([train,valid]);start=time.monotonic();results={};base_results={}
    for key,a in load(OUT/'assignments.json').items():
        base=model(a['seed']);seed=str(a['seed'])
        if seed not in base_results:base_results[seed]=rank_metrics(scores(base),test,seen,train)
        choice=manifest['choices'][key];edited,_=compile_patch(base,choice['patch'],ROOT/choice['snapshot']) if choice else (base,{})
        metrics=rank_metrics(scores(edited),test,seen,train);results[key]={'seed':a['seed'],'replicate':a['replicate'],'selected_id':choice['id'] if choice else 'unchanged','metrics':metrics,'paired_bootstrap':paired_bootstrap(base_results[seed]['per_user'],metrics['per_user'])}
    output={'base':base_results,'astra':results,'manifest_sha256':digest(OUT/'frozen_manifest.json'),'wall_seconds':time.monotonic()-start}
    save(OUT/'final_results.json',serial(output));print('Final reporting complete')

if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','evaluate','freeze','final']);p.add_argument('--key');p.add_argument('--round',type=int,choices=[1,2]);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.command=='prepare':prepare()
    elif a.command=='evaluate':evaluate(a.key,a.round)
    elif a.command=='freeze':freeze()
    else:final()
