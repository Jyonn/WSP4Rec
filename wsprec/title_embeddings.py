"""Extract fixed Qwen title embeddings; never reads interaction labels."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--titles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    titles = json.loads(args.titles.read_text())
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side='left')
    model = AutoModel.from_pretrained(args.model, revision=args.revision,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32).to(device).eval()
    loaded = time.monotonic()
    features = []
    lengths = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(titles), args.batch_size):
        batch = tokenizer([r['title'] for r in titles[start:start+args.batch_size]],
            padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        lengths.extend(batch['attention_mask'].sum(1).cpu().tolist())
        with torch.inference_mode():
            vectors = model(**batch).last_hidden_state[:, -1].float()
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        features.append(vectors.cpu().numpy())
        print(f'embedded {min(start+args.batch_size,len(titles))}/{len(titles)}', flush=True)
        if time.monotonic() - started > 3500:
            raise TimeoutError('text embedding exceeded experiment budget')
    array = np.concatenate(features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array)
    metadata = {'model': args.model, 'requested_revision': args.revision,
        'resolved_revision': getattr(model.config, '_commit_hash', None),
        'shape': list(array.shape), 'pooling': 'last_nonpadding_token_left_padding',
        'instruction': None, 'max_length': 128, 'titles_sha256': hashlib.sha256(args.titles.read_bytes()).hexdigest(),
        'embeddings_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'wall_seconds': time.monotonic()-started, 'load_seconds': loaded-started,
        'encode_seconds': time.monotonic()-loaded, 'input_tokens': sum(lengths),
        'device': str(device), 'gpu_name': torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        'torch': torch.__version__, 'python': platform.python_version(), 'billing_usd': None}
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
