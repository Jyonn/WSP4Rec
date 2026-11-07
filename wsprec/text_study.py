"""Training-aware Astra editing of ID-only and description-integrated models.

The fixed text representations are never editable. Editor-specific logical
allowlists distinguish numerical text features from raw movie descriptions.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .study import checkpoint, quick_metric, save
from .study_metrics import rank_metrics, paired_bootstrap
from .tensor_edits import compile_patch, _asset_array
from .text_model import scores, simple_target, simple_fusion_delta, effective_items, feature_path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/v7_text'
DATA=ROOT/'data/ml-100k-description-complete-v3'
PROTOCOL=ROOT/'text_study_protocol_v7.json'
SEEDS=(42,43,44)
ACCESS=('none','numeric_text','full_text')
REPLICATES=(0,1)


def load(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def npstate(state): return {k:v.detach().cpu().numpy().copy() for k,v in state.items()}
def state_digest(state):
    h=hashlib.sha256()
    for name,array in sorted(npstate(state).items()):
        h.update(name.encode());h.update(str(array.shape).encode());h.update(str(array.dtype).encode());h.update(array.tobytes())
    return h.hexdigest()
def serial(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serial(v) for v in value]
    return value


def apply(state,delta):
    result={k:v.clone() for k,v in state.items()}
    for k,v in delta.items(): result[k]+=torch.as_tensor(v,dtype=result[k].dtype)
    return result


def effective_change_stats(base,edited,features):
    """Record shared projection effects separately from the learned-array cap."""
    before=effective_items(base,features).numpy().astype(float)
    after=effective_items(edited,features).numpy().astype(float)
    change=float(np.linalg.norm(after-before))
    result={'effective_item_delta_l2':change,
            'effective_item_relative_delta_l2':change/max(float(np.linalg.norm(before)),1e-30),
            'item_id_delta_l2':float(torch.linalg.vector_norm(edited['item.weight'].double()-base['item.weight'].double()))}
    if 'text_projection' in base:
        delta=(edited['text_projection']-base['text_projection']).numpy().astype(float)
        result['text_projection_delta_l2']=float(np.linalg.norm(delta))
        result['text_item_contribution_delta_l2']=float(np.linalg.norm(features.astype(float)@delta))
    return result


def projected(assignment):
    return np.load(ROOT/assignment['projected_features']) if assignment['projected_features'] else None


def portable_features(assignment):
    """Portable programs receive all numerical text inputs through arguments."""
    if not assignment['projected_features']: return None
    return {'projected':np.load(ROOT/assignment['projected_features']),
            'llama':np.load(ROOT/assignment['llama_features'])}


def unpack_patches(payload):
    """Accept equivalent outer containers; the four patch bodies are unchanged."""
    if isinstance(payload,dict) and set(payload)=={'patches'}:
        payload=payload['patches']
    if not isinstance(payload,list) or len(payload)!=4:
        raise ValueError('exactly four patches required')
    return payload


def make_controls(model_condition,seed,base,features,train,valid):
    """Compute each checkpoint's controls once, shared across access/replicates."""
    folder=OUT/'controls'/f'{model_condition}-s{seed}'
    manifest=folder/'manifest.json'
    source_hash=state_digest(base['state'])
    features_hash=None if features is None else hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
    if manifest.exists():
        previous=load(manifest)
        assert previous['source_learned_state_sha256']==source_hash,'stale controls from different checkpoint'
        assert previous['feature_values_sha256']==features_hash,'stale controls from different text features'
        return str(manifest.relative_to(ROOT))
    folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    s=npstate(base['state']);target=simple_target(s,features,train)
    candidates=[('fixed_simple',.5,.25)]
    candidates.extend((f'grid-a{a}-b{b}',a,b) for a in (.25,.5,.75,1.) for b in (0.,.25))
    records=[]
    for name,alpha,beta in candidates:
        delta=simple_fusion_delta(s,features,train,alpha=alpha,beta=beta,target=target)
        state=apply(base['state'],delta)
        norm=np.sqrt(sum(float(v.double().square().sum()) for v in base['state'].values()))
        dn=np.sqrt(sum(float((state[k].double()-v.double()).square().sum()) for k,v in base['state'].items()))
        assert dn/norm<=.5+1e-7
        path=folder/f'{name}.pt'
        torch.save({'config':base['config'],'state':state},path)
        records.append({'method':name,'alpha':alpha,'beta':beta,'path':str(path.relative_to(ROOT)),
                        'valid_ndcg@10':quick_metric(scores(state,features),valid,train),
                        'relative_delta_l2':float(dn/norm),**effective_change_stats(base['state'],state,features)})
    save(manifest,{'records':records,'wall_seconds':time.monotonic()-started,
                  'source_learned_state_sha256':source_hash,'feature_values_sha256':features_hash,
                  'scope':'One checkpoint; shared by all editor input/replicate conditions for this checkpoint. Nine scores including fixed candidate also present in eight-grid family.'})
    return str(manifest.relative_to(ROOT))


def prepare():
    if (OUT/'assignments.json').exists(): raise FileExistsError('editor inputs already prepared')
    assert PROTOCOL.exists(),'root must lock protocol before preparation/dispatch'
    started=time.monotonic();train=np.load(DATA/'train.npy');valid=np.load(DATA/'valid.npy')
    summaries={c:load(OUT/c/'training_summary.json') for c in ('none','text')}
    X=np.load(OUT/'text/features.npy');E=np.load(DATA/'title_description.npy')
    movies=load(DATA/'movies.json')
    assert len(movies)==len(X)==len(E)
    assert [m['item_id'] for m in movies]==list(range(len(X)))
    assert all(isinstance(m['description'],str) and len(m['description'].strip())>0 for m in movies)
    identities=[(seed,access,rep) for seed in SEEDS for access in ACCESS for rep in REPLICATES]
    np.random.default_rng(202611072).shuffle(identities)
    assignments={};control_paths={}
    for number,(seed,access,rep) in enumerate(identities,1):
        key=f'case{number:02d}';model_condition='none' if access=='none' else 'text'
        source=summaries[model_condition]['selected'][str(seed)]
        base=checkpoint(ROOT/source);state=base['state'];features=None if model_condition=='none' else X
        checkpoint_key=(model_condition,seed)
        if checkpoint_key not in control_paths:
            control_paths[checkpoint_key]=make_controls(model_condition,seed,base,features,train,valid)
        folder=OUT/'inputs'/key;folder.mkdir(parents=True,exist_ok=True)
        torch.save({'config':base['config'],'state':state},folder/'model.pt')
        np.savez(folder/'parameters.npz',**npstate(state));shutil.copy2(DATA/'train.npy',folder/'train.npy')
        users,items=len(state['user.weight']),len(state['item.weight'])
        save(folder/'id_to_embedding_row.json',{'rule':'user_id and item_id are zero-based direct rows; text feature rows and movie records use these same item IDs.',
            'user_id_to_user_embedding_row':list(range(users)),'item_id_to_item_embedding_row':list(range(items))})
        projected_path=llama_path=None
        if features is not None:
            np.save(folder/'features.npy',X,allow_pickle=False);np.save(folder/'llama_features.npy',E,allow_pickle=False)
            projected_path=str((folder/'features.npy').relative_to(ROOT));llama_path=str((folder/'llama_features.npy').relative_to(ROOT))
        if access=='full_text': save(folder/'movies.json',movies)
        interface=load(ROOT/'editor_interface_v2.json')
        interface['input']='Full learned weights, complete training interactions, ID mapping and the explicitly allowlisted metadata/features.'
        interface['submission']='Exactly four patches with id and operations only; concise observations and proposed actions are saved separately before first feedback.'
        interface['computation']='Training predictions/residuals, direct algebra and one-shot analytic fitting are permitted. No BP, gradients, gradient optimizers, iterative training, automated candidate/parameter search, validation/test access, internet or other files.'
        interface['visibility']='Read only the instruction, allowlisted files and your own outputs/feedback. No prior studies, other cases or evaluator internals. Metadata access is determined solely by allowlist.'
        interface['fixed_features']='Only existing learned parameters may be edited; supplied text features and movie records are immutable.'
        save(folder/'interface.json',interface)
        baseline=quick_metric(scores(state,features),valid,train)
        spec={'architecture':'16-dimensional biased matrix factorization with a frozen text-feature branch' if features is not None else '16-dimensional biased matrix factorization',
              'score':'bias + user_bias.weight[u,0] + item_bias.weight[i,0] + sum_d user.weight[u,d] * (item.weight[i,d] + sum_k features[i,k]*text_projection[k,d])' if features is not None else 'bias + user_bias.weight[u,0] + item_bias.weight[i,0] + sum_d user.weight[u,d]*item.weight[i,d]',
              'parameters':{k:list(v.shape) for k,v in state.items()},
              'training_array':{'file':'train.npy','columns':['user_id','item_id','rating','timestamp'],'positive_rating_threshold':4},
              'baseline_valid_ndcg@10':baseline,
              'training':'BPR ranking loss with BP; all learned parameters jointly trained, fixed text encoder/features not fine-tuned.'}
        if features is not None:
            spec['text_features']={'features.npy':{'shape':list(X.shape),'definition':'Frozen 64D centered PCA projection of Llama-3 title-plus-public-description vectors; no whitening, then per-item L2 normalization. PCA uses catalogue metadata only, not interaction labels. Used directly in the score above.'},
                                   'llama_features.npy':{'shape':list(E.shape),'definition':'Cached Llama-3 title-plus-public-description embeddings; item row IDs exactly match features.npy. Not directly multiplied in the model score.'},
                                   'text_projection':'Learned matrix mapping frozen64D features into16D factor space; editable as an existing parameter.',
                                   'portable_argument':'features is a dictionary with keys projected (64D matrix) and llama (4096D matrix); no-text model receives None.'}
        else: spec['portable_argument']='features=None; this model has no text_projection parameter.'
        save(folder/'spec.json',spec)
        editor=OUT/'editors'/key;editor.mkdir(parents=True,exist_ok=True)
        names=['parameters.npz','train.npy','interface.json','spec.json','id_to_embedding_row.json']
        if features is not None: names+=['features.npy','llama_features.npy']
        if access=='full_text': names+=['movies.json']
        allowed=[str((folder/n).relative_to(ROOT)) for n in names]
        assignments[key]={'seed':seed,'access':access,'replicate':rep,'model_condition':model_condition,'source_checkpoint':source,
                          'model':str((folder/'model.pt').relative_to(ROOT)),'projected_features':projected_path,'llama_features':llama_path,
                          'controls':control_paths[checkpoint_key],'allowlist':allowed,'output_directory':str(editor.relative_to(ROOT)),
                          'baseline_valid_ndcg@10':baseline}
        prompt=('Edit the supplied recommendation model to improve validation NDCG@10. Read only this instruction, the allowlist, and your own saved outputs/feedback. '
                'You may inspect all learned weights, the complete training interactions, the exact ID-to-row mapping, and any explicitly allowlisted features or movie information. '
                'No particular problem, explanation, or editing method is suggested. Before evaluator feedback, save diagnosis_before_feedback.json with concise observable findings, supporting numerical evidence where available, and proposed parameter edits. Report facts and planned actions, not private reasoning traces. '
                'Save generation code and standalone portable.py defining build(state, train, features). state is a NumPy-parameter dictionary; train is the interaction array; features is None for the no-text model or a dictionary with projected and llama numerical matrices as specified. Return a dictionary mapping every submitted candidate ID from both rounds to a dictionary of parameter-name delta arrays. The program must reproduce every candidate using its arguments and any compiled constants, without file or internet reads. '
                'Training predictions/residuals, direct algebra and one-shot analytic fitting are allowed. Do not use BP, gradients, gradient optimizers, iterative model training, automated candidate/parameter search, internet, validation/test rows, other cases or other files. '
                'Submit exactly four distinct patches starting from the supplied original checkpoint; after score-only feedback, submit four final patches, again starting from that same original. '
                'Supplied feature matrices and metadata must remain unchanged. Follow the shared interface limits. Use unique asset names across rounds and preserve all prior submissions and assets. Reply only with the submission path.\n\nAllowlist:\n'
                +'\n'.join(allowed)+f'\n\nOutput directory: {editor.relative_to(ROOT)}\nRound-one submission: {editor.relative_to(ROOT)}/round1.json\n')
        prompts=OUT/'prompts';prompts.mkdir(exist_ok=True);(prompts/f'{key}.txt').write_text(prompt)
    save(OUT/'assignments.json',assignments)
    files=[p for p in (OUT/'inputs').rglob('*') if p.is_file()]+list((OUT/'prompts').glob('*.txt'))
    files += [ROOT/a['source_checkpoint'] for a in assignments.values()]+[DATA/n for n in ('train.npy','valid.npy','splits.npz','movies.json','title_description.npy','manifest.json')]
    files += [OUT/c/'training_summary.json' for c in ('none','text')]+[OUT/'text/features.npy',PROTOCOL]
    save(OUT/'preparation.json',{'wall_seconds':time.monotonic()-started,
        'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))},
        'control_checkpoint_count':len(control_paths),'fixed_simple':{'alpha':.5,'beta':.25},
        'architecture_scope':'With a free item residual V, V+XW has the same unrestricted16D MF score-function class. Text changes parameter sharing, regularization and optimization; nonzero W alone does not prove semantic utility.',
        'historical_test_scope':'Same MovieLens cohort and split used in previous studies; locked new model and editing comparisons, not untouched external confirmation.',
        'isolation':'Instruction allowlists; no OS-level file isolation.'})
    print('Prepared',len(assignments),'editor trajectories',flush=True)


def evaluate(key,rnd):
    a=load(OUT/'assignments.json')[key];folder=OUT/'editors'/key
    dest=OUT/'evaluations'/key/f'round{rnd}.json';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): raise FileExistsError('already evaluated')
    diagnosis=folder/'diagnosis_before_feedback.json'
    if rnd==1:
        save(dest.parent/'diagnosis_before_feedback_snapshot.json',load(diagnosis))
        save(dest.parent/'diagnosis_timing.json',{'sha256':digest(diagnosis),'frozen_before_round1_evaluation_at':time.time()})
    else: assert digest(diagnosis)==load(dest.parent/'diagnosis_timing.json')['sha256'],'initial observations changed'
    with dest.with_suffix('.started').open('x') as f:f.write(str(time.time()))
    base=checkpoint(ROOT/a['model'])['state'];X=projected(a)
    train,valid=np.load(DATA/'train.npy'),np.load(DATA/'valid.npy');start=time.monotonic()
    patches=unpack_patches(load(folder/f'round{rnd}.json'))
    results=[];signatures=set();ids=set()
    for index,patch in enumerate(patches):
        identifier=patch.get('id',str(index)) if isinstance(patch,dict) else str(index)
        try:
            if not isinstance(patch,dict) or set(patch)!={'id','operations'}:raise ValueError('patch only')
            if not isinstance(patch['operations'],list) or not all(isinstance(op,dict) for op in patch['operations']):raise ValueError('operations')
            snapshot=dest.parent/'snapshots'/f'r{rnd}-c{index}';snapshot.mkdir(parents=True,exist_ok=False)
            payload=[]
            for op in patch['operations']:
                entry={k:v for k,v in op.items() if k!='asset'}
                if 'asset' in op:
                    array=_asset_array(folder,op['asset']);target=snapshot/op['asset'];target.parent.mkdir(parents=True,exist_ok=True)
                    np.save(target,array,allow_pickle=False);entry['asset_sha256']=digest(target)
                payload.append(entry)
            save(snapshot/'patch.json',patch);signature=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
            if signature in signatures or identifier in ids:raise ValueError('duplicate candidate in batch')
            signatures.add(signature);ids.add(identifier)
            edited,stats=compile_patch(base,patch,snapshot)
            results.append({'id':identifier,'valid_ndcg@10':quick_metric(scores(edited,X),valid,train),'error':None,
                            'patch':patch,'snapshot':str(snapshot.relative_to(ROOT)),**stats,
                            **effective_change_stats(base,edited,X)})
        except (ValueError,TypeError,KeyError,RuntimeError,OSError) as error:
            results.append({'id':identifier,'valid_ndcg@10':None,'error':'invalid_patch','audit_error':str(error),'patch':patch})
    save(dest,{'results':results,'wall_seconds':time.monotonic()-start})
    feedback=[{k:r[k] for k in ('id','valid_ndcg@10','error')} for r in results]
    (OUT/'feedback').mkdir(exist_ok=True);save(OUT/'feedback'/f'{key}-round{rnd}.json',feedback)
    if rnd==1:
        (OUT/'prompts'/f'{key}-round2.txt').write_text(f'Read runs/v7_text/feedback/{key}-round1.json in addition to your original allowlist. Under the same original instructions submit exactly four final patches to runs/v7_text/editors/{key}/round2.json. Preserve diagnosis_before_feedback.json and every previous submission/asset. Ensure portable.py reproduces all eight candidate IDs. Reply only with the final submission path.')
    print(json.dumps(feedback),flush=True)


def selected(records,baseline):
    eligible=[r for r in records if r['error'] is None and r['valid_ndcg@10']>baseline]
    return max(eligible,key=lambda r:r['valid_ndcg@10']) if eligible else None


def control_choices(a,baseline):
    records=load(ROOT/a['controls'])['records'];result={}
    for method,candidates in [('fixed_simple',[r for r in records if r['method']=='fixed_simple']),
                              ('tuned_simple_8',[r for r in records if r['method'].startswith('grid-')])]:
        best=max(candidates,key=lambda r:r['valid_ndcg@10'])
        result[method]=best if best['valid_ndcg@10']>baseline else None
    return result


def verify_portable(key,a,base,records):
    path=OUT/'editors'/key/'portable.py'
    spec=importlib.util.spec_from_file_location('portable_'+key,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    train=np.load(DATA/'train.npy')
    regenerated=module.build(npstate(base),train.copy(),copy.deepcopy(portable_features(a)))
    count=0;maximum=0.
    for record in records:
        if record['error'] is not None:continue
        original,_=compile_patch(base,record['patch'],ROOT/record['snapshot']);delta=regenerated[record['id']]
        if not isinstance(delta,dict):raise TypeError('portable candidate must be a delta dictionary')
        if set(delta)-set(base):raise ValueError('portable candidate includes unknown parameter')
        for name in base:
            actual=base[name].numpy().astype(float)+np.asarray(delta.get(name,0))
            expected=original[name].numpy()
            maximum=max(maximum,float(np.max(np.abs(actual-expected))))
            if not np.allclose(actual,expected,atol=1e-6,rtol=1e-6):raise ValueError(f'{key}: portable mismatch {record["id"]}/{name}')
        count+=1
    return {'valid_candidates_reproduced':count,'max_absolute_parameter_error':maximum,'atol':1e-6,'rtol':1e-6}


def freeze():
    from .text_calibration import choose as choose_calibration
    dest=OUT/'frozen_manifest.json'
    if dest.exists():raise FileExistsError('already frozen')
    prep=load(OUT/'preparation.json')
    for file,h in prep['input_sha256'].items():assert digest(ROOT/file)==h,file
    train,valid=np.load(DATA/'train.npy'),np.load(DATA/'valid.npy')
    assignments=load(OUT/'assignments.json');assert len(assignments)==18
    assert (OUT/'shuffled_text/training_summary.json').exists(),'supplementary shuffled-text BP training must finish before final freeze'
    choices={};replay_count=0;portable_audit={};control_replay={};start=time.monotonic()
    for key,a in assignments.items():
        base=checkpoint(ROOT/a['model'])['state'];X=projected(a);baseline=quick_metric(scores(base,X),valid,train)
        assert baseline==a['baseline_valid_ndcg@10']
        assert digest(OUT/'editors'/key/'diagnosis_before_feedback.json')==load(OUT/'evaluations'/key/'diagnosis_timing.json')['sha256']
        records=[]
        for rnd in (1,2):
            batch=load(OUT/'evaluations'/key/f'round{rnd}.json')['results'];assert len(batch)==4
            assert load(OUT/'feedback'/f'{key}-round{rnd}.json')==[{k:r[k] for k in ('id','valid_ndcg@10','error')} for r in batch]
            records+=batch
            for r in batch:
                if r['error'] is None:
                    edited,_=compile_patch(base,r['patch'],ROOT/r['snapshot'])
                    assert np.isclose(quick_metric(scores(edited,X),valid,train),r['valid_ndcg@10'],atol=1e-12,rtol=0)
                    replay_count+=1
        portable_audit[key]=verify_portable(key,a,base,records)
        if a['controls'] not in control_replay:
            controls=load(ROOT/a['controls'])['records']
            for c in controls:
                state=checkpoint(ROOT/c['path'])['state']
                assert np.isclose(quick_metric(scores(state,X),valid,train),c['valid_ndcg@10'],atol=1e-12,rtol=0)
            control_replay[a['controls']]=len(controls)
        calibration_manifest=OUT/'calibration'/f"{a['model_condition']}-s{a['seed']}"/'manifest.json'
        calibration_key=str(calibration_manifest.relative_to(ROOT))
        if calibration_key not in control_replay:
            calibration_records=load(calibration_manifest)['records']
            for c in calibration_records:
                state=checkpoint(ROOT/c['path'])['state']
                assert np.isclose(quick_metric(scores(state,X),valid,train),c['valid_ndcg@10'],atol=1e-12,rtol=0)
            control_replay[calibration_key]=len(calibration_records)
        choices[key]={'astra':selected(records,baseline),**control_choices(a,baseline),
                      'simple_calibration_8':choose_calibration(a['model_condition'],a['seed'],baseline)}
    save(OUT/'portable_audit.json',portable_audit)
    files=[p for p in OUT.rglob('*') if p.is_file() and 'spectral_cache' not in p.parts and '__pycache__' not in p.parts]
    files+=[PROTOCOL,ROOT/'wsprec/text_study.py',ROOT/'wsprec/text_calibration.py',ROOT/'wsprec/text_model.py',ROOT/'wsprec/tensor_edits.py',ROOT/'wsprec/study.py',ROOT/'wsprec/study_metrics.py',DATA/'train.npy',DATA/'valid.npy',DATA/'splits.npz',DATA/'movies.json',DATA/'title_description.npy']
    save(dest,{'choices':choices,'sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))},
               'replayed_valid_candidates':replay_count,'control_replayed_scores':sum(control_replay.values()),
               'portable_audit':portable_audit,'wall_seconds':time.monotonic()-start,'frozen_at':time.time()})
    print('Frozen',digest(dest),flush=True)


def final():
    frozen=load(OUT/'frozen_manifest.json')
    for file,h in frozen['sha256'].items():assert digest(ROOT/file)==h,file
    with (OUT/'test_opened.lock').open('x') as f:f.write(str(time.time()))
    train,valid=np.load(DATA/'train.npy'),np.load(DATA/'valid.npy')
    with np.load(DATA/'splits.npz') as z:test=z['test']
    seen=np.concatenate([train,valid]);results={};shared_metrics={};start=time.monotonic()
    for key,a in load(OUT/'assignments.json').items():
        base=checkpoint(ROOT/a['model'])['state'];X=projected(a);cache_key=(a['model_condition'],a['seed'])
        if cache_key not in shared_metrics:
            entries={'bp':rank_metrics(scores(base,X),test,seen,train)}
            raw_fixed=next(c for c in load(ROOT/a['controls'])['records'] if c['method']=='fixed_simple')
            entries['fixed_simple_raw']=rank_metrics(scores(checkpoint(ROOT/raw_fixed['path'])['state'],X),test,seen,train)
            for method in ('fixed_simple','tuned_simple_8','simple_calibration_8'):
                chosen=frozen['choices'][key][method];state=base if chosen is None else checkpoint(ROOT/chosen['path'])['state']
                entries[method]=rank_metrics(scores(state,X),test,seen,train)
            shared_metrics[cache_key]=entries
        entries=dict(shared_metrics[cache_key]);chosen=frozen['choices'][key]['astra']
        edited=base if chosen is None else compile_patch(base,chosen['patch'],ROOT/chosen['snapshot'])[0]
        entries['astra']=rank_metrics(scores(edited,X),test,seen,train)
        first=load(OUT/'evaluations'/key/'round1.json')['results'][0]
        entries['astra_first_patch']=None if first['error'] else rank_metrics(scores(compile_patch(base,first['patch'],ROOT/first['snapshot'])[0],X),test,seen,train)
        results[key]={'seed':a['seed'],'access':a['access'],'replicate':a['replicate'],'model_condition':a['model_condition'],
                      'selected_id':chosen['id'] if chosen else 'unchanged','metrics':entries,
                      'paired_astra_vs_bp':paired_bootstrap(entries['bp']['per_user'],entries['astra']['per_user']),
                      'paired_astra_vs_fixed':paired_bootstrap(entries['fixed_simple']['per_user'],entries['astra']['per_user']),
                      'paired_astra_vs_tuned':paired_bootstrap(entries['tuned_simple_8']['per_user'],entries['astra']['per_user'])}
    shuffled_summary=load(OUT/'shuffled_text/training_summary.json')
    shuffled_X=np.load(feature_path('shuffled_text'))
    shuffled={str(seed):rank_metrics(scores(checkpoint(ROOT/shuffled_summary['selected'][str(seed)])['state'],shuffled_X),test,seen,train) for seed in SEEDS}
    save(OUT/'final_results.json',serial({'cases':results,'bp_only_shuffled_text':shuffled,
         'wall_seconds':time.monotonic()-start,'frozen_manifest_sha256':digest(OUT/'frozen_manifest.json')}))
    report()


def report():
    final_results=load(OUT/'final_results.json');summary={}
    for access in ACCESS:
        cases=[c for c in final_results['cases'].values() if c['access']==access]
        # Average editing replicates inside each training seed first.
        summary[access]={method:{metric:float(np.mean([np.mean([c['metrics'][method]['summary'][metric] for c in cases if c['seed']==seed]) for seed in SEEDS]))
                                for metric in ('ndcg@10','recall@10')}
                         for method in ('bp','fixed_simple','fixed_simple_raw','tuned_simple_8','simple_calibration_8','astra')}
    folder=ROOT/'reports/v7_text';folder.mkdir(parents=True,exist_ok=True)
    shuffled={metric:float(np.mean([v['summary'][metric] for v in final_results['bp_only_shuffled_text'].values()])) for metric in ('ndcg@10','recall@10')}
    save(folder/'summary.json',{'conditions':summary,'bp_only_shuffled_text':shuffled,
                               'replicates_per_checkpoint':2,'training_seeds':list(SEEDS),
                               'averaging':'Two editor replicates averaged within each seed, then three training seeds. BP/fixed values shared, not independently retrained for each editor.',
                               'limitations':'Historically reused single MovieLens cohort, one architecture family, three training seeds and two editor realizations per input condition; logical file allowlists only.'})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','evaluate','freeze','final','report']);parser.add_argument('--key');parser.add_argument('--round',type=int,choices=(1,2));args=parser.parse_args()
    evaluate(args.key,args.round) if args.command=='evaluate' else globals()[args.command]()
