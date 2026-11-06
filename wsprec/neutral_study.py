"""Description-complete study with generic, direction-neutral tensor patches."""
import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from .cli import FM
from .study import quick_metric, scores, save, checkpoint
from .study_metrics import rank_metrics, paired_bootstrap
from .tensor_edits import compile_patch, _asset_array

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/ml-100k-description-complete'
OUT=ROOT/'runs/v2_neutral'
OBJECTIVES=('mse','bpr')
TEXTS=('none','title','description')
VISIBILITIES=('visible','blind')
REPLICATES=(0,1)
CONTROL_FAMILIES=('uniform','random','title_neighbors','description_neighbors')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data():
    return np.load(DATA/'train.npy'),np.load(DATA/'valid.npy')


def dimensions():
    m=json.loads((DATA/'manifest.json').read_text())
    return m['counts']['after']['users'],m['counts']['after']['items']


def trajectory(obj,text,visibility,rep):return f'{obj}-{text}-{visibility}-r{rep}'


def train_one(obj,seed,lr,wd):
    t=time.monotonic();torch.manual_seed(seed)
    train,valid=data();users,items=dimensions()
    model=FM(users,items,16)
    x=torch.as_tensor(train,dtype=torch.long);pos=x[x[:,2]>=4]
    known=torch.zeros((users,items),dtype=torch.bool);known[x[:,0],x[:,1]]=True
    if obj=='bpr' and (known.sum(1)[pos[:,0]]==items).any():raise ValueError('no eligible negatives')
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    best=-1.;best_epoch=0;best_state=None;history=[]
    for epoch in range(1,81):
        batches=x if obj=='mse' else pos
        losses=[]
        for ix in torch.randperm(len(batches)).split(2048):
            b=batches[ix]
            if obj=='mse':loss=(model(b[:,0],b[:,1])-b[:,2].float()).square().mean()
            else:
                neg=torch.randint(items,(len(b),));bad=known[b[:,0],neg]
                while bad.any():
                    neg[bad]=torch.randint(items,(int(bad.sum()),));bad=known[b[:,0],neg]
                loss=torch.nn.functional.softplus(-(model(b[:,0],b[:,1])-model(b[:,0],neg))).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        with torch.no_grad():value=quick_metric(scores(model.state_dict()),valid,train)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'valid_ndcg@10':value})
        if value>best:
            best,best_epoch=value,epoch
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch>=12:break
        if time.monotonic()-t>3500:raise TimeoutError('training trial budget')
    name=f'{obj}-s{seed}-lr{lr}-wd{wd}';folder=OUT/'training'/name;folder.mkdir(parents=True,exist_ok=True)
    torch.save({'config':{'users':users,'items':items,'dim':16},'state':best_state},folder/'best.pt')
    result={'name':name,'objective':obj,'seed':seed,'lr':lr,'wd':wd,'best_epoch':best_epoch,
        'epochs_run':len(history),'valid_ndcg@10':best,'wall_seconds':time.monotonic()-t,'history':history}
    save(folder/'history.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
    return result


def train_all():
    if (OUT/'training_summary.json').exists():raise FileExistsError('training already complete')
    results=[];selected={}
    for obj in OBJECTIVES:
        trials=[train_one(obj,42,lr,wd) for lr in (.003,.01) for wd in (1e-5,.001)]
        best=max(trials,key=lambda t:t['valid_ndcg@10']);results+=trials;selected[obj]={'42':best['name']}
        for seed in (43,44):
            trial=train_one(obj,seed,best['lr'],best['wd']);results.append(trial);selected[obj][str(seed)]=trial['name']
    save(OUT/'training_summary.json',{'trials':results,'selected':selected,'torch':torch.__version__,
        'numpy':np.__version__,'threads':torch.get_num_threads(),'dataset_manifest_sha256':digest(DATA/'manifest.json')})


def cp(obj,seed=42):
    m=json.loads((OUT/'training_summary.json').read_text())
    return checkpoint(OUT/'training'/m['selected'][obj][str(seed)]/'best.pt')


def export_inputs():
    train,valid=data();users,items=dimensions()
    movies=json.loads((DATA/'movies.json').read_text());folder=OUT/'inputs';folder.mkdir(parents=True,exist_ok=True)
    save(folder/'title_items.json',[{'item_id':r['item_id'],'title':r['title']} for r in movies])
    save(folder/'description_items.json',movies)
    shutil.copy2(DATA/'title_only.npy',folder/'title_features.npy')
    shutil.copy2(DATA/'title_description.npy',folder/'description_features.npy')
    for obj in OBJECTIVES:
        base=cp(obj);state=base['state']
        save(folder/f'{obj}_model.json',{'objective':obj,'dimensions':base['config'],
            'score_definition':'bias + user_bias[u] + item_bias[i] + sum_d user[u,d] * item[i,d]',
            'parameters':{k:list(v.shape) for k,v in state.items()},
            'user_observation_counts':np.bincount(train[:,0],minlength=users).tolist(),
            'item_observation_counts':np.bincount(train[:,1],minlength=items).tolist(),
            'baseline_valid_ndcg@10':quick_metric(scores(state),valid,train)})
        np.savez(folder/f'{obj}_parameters.npz',**{k:v.numpy() for k,v in state.items()})
    allowed={}
    for obj in OBJECTIVES:
        for text in TEXTS:
            for visibility in VISIBILITIES:
                for rep in REPLICATES:
                    key=trajectory(obj,text,visibility,rep)
                    files=['editor_interface_v2.json',str((folder/f'{obj}_model.json').relative_to(ROOT))]
                    if visibility=='visible':files.append(str((folder/f'{obj}_parameters.npz').relative_to(ROOT)))
                    if text!='none':
                        files.extend(str((folder/f'{text}_{suffix}').relative_to(ROOT)) for suffix in ('items.json','features.npy'))
                    path=OUT/'editors'/key;path.mkdir(parents=True,exist_ok=True)
                    allowed[key]={'objective':obj,'text':text,'visibility':visibility,'replicate':rep,
                        'allowlist':files,'output_directory':str(path.relative_to(ROOT))}
    save(OUT/'editor_assignments.json',allowed)


def evaluate_round(key,round_number,control=False):
    folder=OUT/('controls' if control else 'editors')/key
    destination=OUT/'evaluations'/key/f'round{round_number}.json'
    if destination.exists():raise FileExistsError('evaluation already recorded')
    destination.parent.mkdir(parents=True,exist_ok=True)
    marker=destination.with_suffix('.started')
    with marker.open('x') as stream:stream.write(str(time.time()))
    obj=key.split('-')[0];base=cp(obj)['state'];train,valid=data();t=time.monotonic()
    try:
        patches=json.loads((folder/f'round{round_number}.json').read_text())
        if not isinstance(patches,list) or len(patches)!=4:raise ValueError('four patches required')
    except (ValueError,TypeError,OSError):
        patches=[{'id':f'invalid_submission_{round_number}_{i}'} for i in range(4)]
    results=[];seen_ids=set();seen_payloads=set()
    for i,patch in enumerate(patches):
        began=time.monotonic();identifier=patch.get('id',f'r{round_number}_{i}') if isinstance(patch,dict) else f'r{round_number}_{i}'
        try:
            if not isinstance(patch,dict) or set(patch)-{'id','operations'}:raise ValueError('patch-only schema')
            if not isinstance(patch.get('operations'),list) or not all(isinstance(op,dict) for op in patch['operations']):
                raise ValueError('operations must be a list of dictionaries')
            snapshot=destination.parent/'snapshots'/f'r{round_number}-c{i}'
            snapshot.mkdir(parents=True,exist_ok=False)
            for op in patch['operations']:
                if 'asset' in op:
                    # Materialize a validated immutable byte snapshot before scoring.
                    array=_asset_array(folder,op['asset']);dst=snapshot/op['asset'];dst.parent.mkdir(parents=True,exist_ok=True)
                    np.save(dst,array,allow_pickle=False)
            save(snapshot/'patch.json',patch)
            payload=[]
            for op in patch['operations']:
                record={k:v for k,v in op.items() if k!='asset'}
                if 'asset' in op:record['asset_sha256']=digest(snapshot/op['asset'])
                payload.append(record)
            signature=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
            if identifier in seen_ids or signature in seen_payloads:raise ValueError('duplicate candidate in batch')
            seen_ids.add(identifier);seen_payloads.add(signature)
            edited,stats=compile_patch(base,patch,snapshot)
            score=quick_metric(scores(edited),valid,train)
            results.append({'id':identifier,'valid_ndcg@10':score,'error':None,'patch':patch,
                'snapshot':str(snapshot.relative_to(ROOT)),**stats,'wall_seconds':time.monotonic()-began})
        except (ValueError,KeyError,TypeError,RuntimeError,OSError) as e:
            results.append({'id':identifier,'valid_ndcg@10':None,'error':'invalid_patch','audit_error':str(e),
                'patch':patch,'wall_seconds':time.monotonic()-began})
        if time.monotonic()-t>3500:raise TimeoutError('evaluation budget')
    result={'trajectory':key,'round':round_number,'wall_seconds':time.monotonic()-t,'results':results}
    temporary=destination.with_suffix('.writing.json');save(temporary,result);temporary.replace(destination)
    feedback=[{k:r[k] for k in ('id','valid_ndcg@10','error')} for r in results]
    print(json.dumps({'trajectory':key,'feedback':feedback}),flush=True)
    return result


def build_controls():
    train,_=data();users,items=dimensions();rng=np.random.default_rng(20261106)
    for obj in OBJECTIVES:
        base=cp(obj)['state']
        patches=[]
        for j,factor in enumerate((.5,.65,.8,.9,1.1,1.2,1.35,1.5)):
            patches.append({'id':f'uniform_{j}','operations':[{'parameter':k,'op':'multiply','value':factor} for k in base]})
        families={'uniform':patches}
        folder=OUT/'controls'/f'{obj}-random';folder.mkdir(parents=True,exist_ok=True)
        patches=[]
        for j,strength in enumerate((.01,.03,.1,.2,.01,.03,.1,.2)):
            ops=[]
            for k,tensor in base.items():
                noise=rng.standard_normal(tuple(tensor.shape))*max(float(tensor.std()) if tensor.numel()>1 else abs(float(tensor)),1e-8)*strength
                name=f'noise_{j}_{len(ops)}.npy';np.save(folder/name,np.asarray(noise,dtype=np.float32))
                ops.append({'parameter':k,'op':'add','asset':name})
            patches.append({'id':f'random_{j}','operations':ops})
        families['random']=patches
        for text in ('title','description'):
            features=np.load(OUT/f'inputs/{text}_features.npy');sim=features@features.T;np.fill_diagonal(sim,-np.inf)
            order=np.argsort(-sim,axis=1,kind='stable')[:,:10]
            values=np.maximum(np.take_along_axis(sim,order,1),0);values/=np.maximum(values.sum(1,keepdims=True),1e-8)
            neighbors=np.zeros((items,items),dtype=np.float32);neighbors[np.arange(items)[:,None],order]=values
            name=f'{text}_neighbors';folder=OUT/'controls'/f'{obj}-{name}';folder.mkdir(parents=True,exist_ok=True)
            patches=[]
            for j,alpha in enumerate((.01,.025,.05,.1,.15,.2,.3,.5)):
                matrix=(1-alpha)*np.eye(items,dtype=np.float32)+alpha*neighbors
                asset=f'matrix_{j}.npy';np.save(folder/asset,matrix)
                patches.append({'id':f'{text}_neighbors_{j}','operations':[{'parameter':'item.weight','op':'left_multiply','asset':asset}]})
            families[name]=patches
        for family,patches in families.items():
            key=f'{obj}-{family}';folder=OUT/'controls'/key;folder.mkdir(parents=True,exist_ok=True)
            for r in (1,2):
                save(folder/f'round{r}.json',patches[(r-1)*4:r*4]);evaluate_round(key,r,True)


def references():
    train,valid=data();users,items=dimensions();x=np.zeros((users,items),dtype=np.float64)
    positive=train[train[:,2]>=4];x[positive[:,0],positive[:,1]]=1
    folder=OUT/'references';folder.mkdir(parents=True,exist_ok=True)
    pop=np.broadcast_to(x.sum(0),x.shape).copy();np.save(folder/'popular.npy',pop)
    results=[{'id':'popular','valid_ndcg@10':quick_metric(pop,valid,train)}]
    gram=x.T@x;best=-1.;t=time.monotonic()
    for reg in (100.,300.,1000.,3000.):
        inv=np.linalg.inv(gram+reg*np.eye(items));coeff=-inv/np.diag(inv)[None,:];np.fill_diagonal(coeff,0)
        prediction=x@coeff;value=quick_metric(prediction,valid,train);results.append({'id':f'ease_{reg}','valid_ndcg@10':value})
        if value>best:best=value;np.save(folder/'ease.npy',prediction)
    save(folder/'validation.json',{'results':results,'wall_seconds':time.monotonic()-t})


def choose(key,obj):
    baseline=json.loads((OUT/f'inputs/{obj}_model.json').read_text())['baseline_valid_ndcg@10']
    records=[]
    for r in (1,2):records+=json.loads((OUT/'evaluations'/key/f'round{r}.json').read_text())['results']
    valid=[v for v in records if v['error'] is None and v['valid_ndcg@10']>baseline]
    best=max(valid,key=lambda v:(v['valid_ndcg@10'],-v['changed_parameters'])) if valid else None
    return {'selection':best,'baseline_valid_ndcg@10':baseline,'candidate_queries':len(records)}


def freeze():
    target=OUT/'frozen_manifest.json'
    if target.exists():raise FileExistsError('already frozen')
    audit=json.loads((OUT/'consistency_audit.json').read_text())
    if not audit['passed']:raise ValueError('pre-test consistency replay did not pass')
    for file,value in audit['inputs_sha256'].items():
        if digest(ROOT/file)!=value:raise ValueError('changed consistency-audited input: '+file)
    users,items=dimensions()
    for name in ('popular','ease'):
        path=OUT/f'references/{name}.npy'
        array=np.load(path,allow_pickle=False)
        if array.shape!=(users,items) or not np.isfinite(array).all():raise ValueError('invalid required reference')
    assignments=json.loads((OUT/'editor_assignments.json').read_text());choices={};hashes={}
    if len(assignments)!=24:raise ValueError('expected24 editor trajectories')
    for key,assignment in assignments.items():
        choices[key]=choose(key,assignment['objective'])
        if choices[key]['candidate_queries']!=8:raise ValueError('expected8 slots per trajectory')
        for suffix in ('.txt','.dispatch.txt','-round2.dispatch.txt'):
            if not (OUT/'prompts'/f'{key}{suffix}').is_file():raise ValueError('missing exact prompt record')
        first=json.loads((OUT/'evaluations'/key/'round1.json').read_text())['results']
        feedback=json.loads((OUT/'feedback'/f'{key}-round1.json').read_text())
        expected=[{k:r[k] for k in ('id','valid_ndcg@10','error')} for r in first]
        if feedback!=expected:raise ValueError('feedback is not the prescribed score-only data')
    for obj in OBJECTIVES:
        for family in CONTROL_FAMILIES:choices[f'{obj}-{family}']=choose(f'{obj}-{family}',obj)
    for path in [ROOT/'neutral_protocol_v2.json',ROOT/'editor_interface_v2.json',ROOT/'editor_prompt_v2.txt',
        ROOT/'wsprec/neutral_study.py',ROOT/'wsprec/tensor_edits.py',ROOT/'wsprec/study.py',ROOT/'wsprec/study_metrics.py',ROOT/'wsprec/cli.py',
        OUT/'training_summary.json',OUT/'consistency_audit.json',DATA/'manifest.json',DATA/'train.npy',DATA/'valid.npy',
        DATA/'splits.npz',DATA/'movies.json',DATA/'title_only.npy',DATA/'title_description.npy']:
        hashes[str(path.relative_to(ROOT))]=digest(path)
    for obj in OBJECTIVES:
        m=json.loads((OUT/'training_summary.json').read_text())
        for seed in (42,43,44):
            path=OUT/'training'/m['selected'][obj][str(seed)]/'best.pt';hashes[str(path.relative_to(ROOT))]=digest(path)
    for pattern in ('evaluations/**/*.json','evaluations/**/snapshots/**/*.npy','references/*.npy','references/*.json','inputs/*','prompts/*.txt','feedback/*.json','editor_assignments.json'):
        for path in OUT.glob(pattern):
            if path.is_file():hashes[str(path.relative_to(ROOT))]=digest(path)
    if not list((OUT/'prompts').glob('*.txt')):raise ValueError('record exact prompts first')
    with (OUT/'freeze_started.lock').open('x') as stream:stream.write(str(time.time()))
    temporary=target.with_suffix('.writing.json')
    save(temporary,{'created_unix':time.time(),'choices':choices,'sha256':hashes,'test_scope':'filtered/re-split same original corpus, not an external new test set'})
    temporary.replace(target)
    print('Frozen',digest(target),flush=True)


def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [serial(x) for x in v]
    return v


def final():
    path=OUT/'final_results.json'
    if path.exists():raise FileExistsError('final results already exist')
    manifest=json.loads((OUT/'frozen_manifest.json').read_text())
    for file,h in manifest['sha256'].items():
        if digest(ROOT/file)!=h:raise ValueError('changed frozen input: '+file)
    # Exclusively claim the reporting pass before any test-label decompression.
    with (OUT/'test_opened.lock').open('x') as stream:
        stream.write(json.dumps({'manifest_sha256':digest(OUT/'frozen_manifest.json'),'opened_unix':time.time()}))
    train,valid=data()
    with np.load(DATA/'splits.npz') as z:test=z['test']
    seen=np.concatenate([train,valid]);started=time.monotonic();output={'objectives':{},'references':{},'manifest_sha256':digest(OUT/'frozen_manifest.json')}
    movies=json.loads((DATA/'movies.json').read_text())
    repetitions=Counter((m['title'],m['description']) for m in movies)
    unique_ids={m['item_id'] for m in movies if repetitions[(m['title'],m['description'])]==1}
    unique_test=test[np.isin(test[:,1],list(unique_ids))]
    output['metadata_diagnostic']={'duplicate_text_extra_records':len(movies)-len(repetitions),
        'unique_text_catalogue_items':len(unique_ids),'unique_text_test_rows':len(unique_test)}
    for obj in OBJECTIVES:
        output['objectives'][obj]={}
        for seed in (42,43,44):
            base=cp(obj,seed)['state'];base_scores=scores(base);base_metrics=rank_metrics(base_scores,test,seen,train)
            methods={'base':{'metrics':base_metrics,'unique_text_target_summary':rank_metrics(base_scores,unique_test,seen,train)['summary']}}
            for key,choice in manifest['choices'].items():
                if not key.startswith(obj+'-'):continue
                selected=choice['selection']
                transfer_error=None
                if selected:
                    try:state,stats=compile_patch(base,selected['patch'],ROOT/selected['snapshot'])
                    except ValueError as error:
                        state,stats=compile_patch(base,{'id':'no_op','operations':[]},OUT)
                        transfer_error=str(error)
                else:state,stats=compile_patch(base,{'id':'no_op','operations':[]},OUT)
                edited_scores=scores(state);metrics=rank_metrics(edited_scores,test,seen,train)
                methods[key]={'metrics':metrics,'edit_stats':stats,'selected_id':selected['id'] if selected else 'no_op',
                    'unique_text_target_summary':rank_metrics(edited_scores,unique_test,seen,train)['summary'],
                    'transfer_error':transfer_error,
                    'paired_bootstrap_vs_base':paired_bootstrap(base_metrics['per_user'],metrics['per_user'])}
            output['objectives'][obj][str(seed)]=methods
    for name in ('popular','ease'):output['references'][name]=rank_metrics(np.load(OUT/f'references/{name}.npy'),test,seen,train)
    output['wall_seconds']=time.monotonic()-started
    temporary=path.with_suffix('.writing.json');save(temporary,serial(output));temporary.replace(path)
    print('Completed single final reporting pass:',path,flush=True)


def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','export','controls','references','evaluate','freeze','final'])
    p.add_argument('--trajectory');p.add_argument('--round',type=int,choices=[1,2]);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.command=='train':train_all()
    elif a.command=='export':export_inputs()
    elif a.command=='controls':build_controls()
    elif a.command=='references':references()
    elif a.command=='evaluate':evaluate_round(a.trajectory,a.round)
    elif a.command=='freeze':freeze()
    else:final()


if __name__=='__main__':main()
