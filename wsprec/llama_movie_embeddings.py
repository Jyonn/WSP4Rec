"""Prompted, frozen Llama-3 movie representations, without text generation.

Outputs both title-only and title+description conditions from the SAME model.
Missing descriptions use exactly the title-only prompt, not invented content.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path


INSTRUCTION = ('Represent the following movie for personalized recommendation '
    'based on its plot, themes, genre, mood, and setting. '
    'Use only the supplied movie information.')
SUFFIX = '\nMovie representation:'


def build_prompt(title, description=None):
    text = INSTRUCTION + '\nTitle: ' + title
    if description and description.strip():
        text += '\nDescription: ' + description.strip()
    return text + SUFFIX


def tokenize_preserving_prompt(tokenizer, title, description, max_length):
    description = description.strip() if description else ''
    original_ids = tokenizer.encode(build_prompt(title,description),add_special_tokens=True)
    ids, truncated = original_ids, False
    while len(ids) > max_length:
        if not description:
            raise ValueError('title and fixed prompt exceed context budget')
        description_ids = tokenizer.encode(description,add_special_tokens=False)
        keep = max(0,len(description_ids)-(len(ids)-max_length)-4)
        description = tokenizer.decode(description_ids[:keep],skip_special_tokens=True)
        ids = tokenizer.encode(build_prompt(title,description),add_special_tokens=True)
        truncated = True
    return ids, {'input_tokens':len(ids),'original_input_tokens':len(original_ids),'description_truncated':truncated}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--movies',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('runs/followup/llama3'))
    parser.add_argument('--model',default='meta-llama/Meta-Llama-3-8B')
    parser.add_argument('--revision',default='8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-length',type=int,default=512)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--max-seconds',type=int,default=3500)
    parser.add_argument('--smoke',type=int,default=0,help='positive number selects prefix only, never call this full experiment')
    args=parser.parse_args()
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel,AutoTokenizer
    started=time.monotonic()
    torch.set_num_threads(4)
    records=json.loads(args.movies.read_text())
    if args.smoke:records=records[:args.smoke]
    if [r['item_id'] for r in records] != list(range(len(records))):
        raise ValueError('movie records must be in contiguous zero-based catalogue order')
    args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,local_files_only=True,padding_side='left')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='left'
    model=AutoModel.from_pretrained(args.model,revision=args.revision,local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type=='cuda' else torch.float32,
        attn_implementation='sdpa').to(device).eval()
    loaded=time.monotonic()
    available=np.array([bool(r.get('description')) for r in records],dtype=bool)
    np.save(args.output/'description_available.npy',available)
    all_conditions={}
    arrays={}
    prompt_hashes={}
    for condition in ('title_only','title_description'):
        t=time.monotonic();features=[];tokens=[];details=[];hashes=[]
        for record in records:
            description=record.get('description') if condition=='title_description' else None
            ids,info=tokenize_preserving_prompt(tokenizer,record['title'],description,args.max_length)
            tokens.append(ids);details.append(info)
            hashes.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest())
        tokenized=time.monotonic()
        for offset in range(0,len(tokens),args.batch_size):
            batch=tokenizer.pad({'input_ids':tokens[offset:offset+args.batch_size]},padding=True,return_tensors='pt').to(device)
            with torch.inference_mode():
                # Left padding means the final position is valid for every row.
                vectors=model(**batch,use_cache=False).last_hidden_state[:,-1].float()
                vectors=torch.nn.functional.normalize(vectors,p=2,dim=1)
            features.append(vectors.cpu().numpy())
            print(condition,min(offset+args.batch_size,len(tokens)),len(tokens),flush=True)
            if time.monotonic()-started > args.max_seconds:
                raise TimeoutError('embedding experiment exceeded wall budget')
        array=np.concatenate(features)
        if not np.isfinite(array).all():raise ValueError('nonfinite embedding')
        np.save(args.output/f'{condition}.npy',array)
        arrays[condition]=array
        prompt_hashes[condition]=hashes
        all_conditions[condition]={'shape':list(array.shape),'input_tokens':sum(d['input_tokens'] for d in details),
            'max_input_tokens':max(d['input_tokens'] for d in details),
            'descriptions_truncated':sum(d['description_truncated'] for d in details),
            'wall_seconds':time.monotonic()-t,'tokenization_seconds':tokenized-t,
            'embeddings_sha256':hashlib.sha256((args.output/f'{condition}.npy').read_bytes()).hexdigest(),
            'row_norm_min':float(np.linalg.norm(array,axis=1).min()),'row_norm_max':float(np.linalg.norm(array,axis=1).max())}
    # Force a bit-identical fallback despite possible padded-batch rounding.
    missing=~available
    fallback_raw_delta=float(np.max(np.abs(arrays['title_description'][missing]-arrays['title_only'][missing]))) if missing.any() else 0.
    arrays['title_description'][missing]=arrays['title_only'][missing]
    np.save(args.output/'title_description.npy',arrays['title_description'])
    all_conditions['title_description']['embeddings_sha256']=hashlib.sha256((args.output/'title_description.npy').read_bytes()).hexdigest()
    metadata={'model':args.model,'requested_revision':args.revision,'resolved_revision':getattr(model.config,'_commit_hash',None),
        'model_is_purpose_trained_embedding_model':False,'fine_tuning':False,'generated_descriptions':False,
        'pooling':'last layer, final valid input token of fixed Movie representation: suffix; left padding',
        'normalization':'L2','instruction':INSTRUCTION,'suffix':SUFFIX,'chat_template':False,
        'input_format':'instruction + newline Title: raw MovieLens title including year + optional newline Description: retrieved overview + suffix',
        'max_length':args.max_length,'truncation':'description tokens only; instruction, title and suffix preserved',
        'missing_description_policy':'exact title-only fallback after encoding; same prompt IDs; mask saved',
        'missing_description_raw_numerical_delta_before_exact_fallback':fallback_raw_delta,
        'movies_sha256':hashlib.sha256(args.movies.read_bytes()).hexdigest(),'movies':len(records),
        'descriptions_available':int(available.sum()),'conditions':all_conditions,
        'wall_seconds':time.monotonic()-started,'model_load_seconds':loaded-started,
        'device':str(device),'dtype':'bfloat16' if device.type=='cuda' else 'float32',
        'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
        'billing_usd':None,'smoke_prefix_only':bool(args.smoke)}
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (args.output/'prompt_token_hashes.json').write_text(json.dumps(prompt_hashes))
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()
