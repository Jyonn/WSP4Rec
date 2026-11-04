import argparse
import hashlib
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from .ranking import evaluate_ranking


class FM(nn.Module):
    """Two-field FM: global/user/item bias plus user-item interaction."""
    def __init__(self, users, items, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(3.0))
        self.user_bias = nn.Embedding(users, 1)
        self.item_bias = nn.Embedding(items, 1)
        self.user = nn.Embedding(users, dim)
        self.item = nn.Embedding(items, dim)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)

    def forward(self, u, i):
        return (self.bias + self.user_bias(u).squeeze(-1)
                + self.item_bias(i).squeeze(-1) + (self.user(u) * self.item(i)).sum(-1))


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'ml-100k.zip'
    if not archive.exists():
        urllib.request.urlretrieve('https://files.grouplens.org/datasets/movielens/ml-100k.zip', archive)
    # Extract only the two known files, avoiding arbitrary archive paths.
    with zipfile.ZipFile(archive) as z:
        for name in ('u.data', 'u.item'):
            (root / name).write_bytes(z.read('ml-100k/' + name))
    rows = np.loadtxt(root / 'u.data', dtype=np.int64)
    rows[:, :2] -= 1
    # Per-user chronological 80/10/10; ties broken deterministically by item ID.
    splits = [[], [], []]
    for u in np.unique(rows[:, 0]):
        r = rows[rows[:, 0] == u]
        r = r[np.lexsort((r[:, 1], r[:, 3]))]
        a, b = int(len(r) * .8), int(len(r) * .9)
        for dest, part in zip(splits, (r[:a], r[a:b], r[b:])):
            dest.extend(part)
    np.savez(root / 'splits.npz', **dict(zip(('train', 'valid', 'test'), map(np.asarray, splits))))
    metadata = {'source_sha256': hashlib.sha256((root / 'u.data').read_bytes()).hexdigest(),
                'split': 'per-user chronological 80/10/10; timestamp ties by item',
                'sizes': dict(zip(('train', 'valid', 'test'), map(len, splits)))}
    (root / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def evaluate(model, rows):
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(rows, dtype=torch.long)
        pred = model(x[:, 0], x[:, 1])
        error = pred - x[:, 2].float()
        return {'rmse': error.square().mean().sqrt().item(), 'mae': error.abs().mean().item(), 'n': len(rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'train', 'evaluate', 'edit'])
    p.add_argument('--data', type=Path, default=Path('data/ml-100k'))
    p.add_argument('--run', type=Path, default=Path('runs/fm-seed42'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dim', type=int, default=16)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--split', choices=['valid', 'test'], default='valid')
    p.add_argument('--checkpoint', default='best.pt')
    p.add_argument('--patch', type=Path)
    p.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20])
    p.add_argument('--positive-threshold', type=int, default=4)
    p.add_argument('--selection-k', type=int, default=10)
    args = p.parse_args()
    if any(k < 1 for k in args.ks) or args.selection_k not in args.ks:
        p.error('K must be positive and --selection-k must appear in --ks')
    if args.command == 'prepare':
        prepare(args.data)
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.data / 'splits.npz')
    args.run.mkdir(parents=True, exist_ok=True)
    def metrics(model, split):
        seen = data['train'] if split == 'valid' else np.concatenate([data['train'], data['valid']])
        return {**evaluate(model, data[split]),
                **evaluate_ranking(model, data[split], seen, args.ks, args.positive_threshold)}

    if args.command == 'train':
        train = data['train']
        config = {'users': int(train[:, 0].max()) + 1,
                  'items': int(max(data[k][:, 1].max() for k in data.files)) + 1, 'dim': args.dim}
        model = FM(**config)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        x = torch.as_tensor(train, dtype=torch.long)
        best = -float('inf')
        history = []
        for epoch in range(args.epochs):
            model.train()
            for idx in torch.randperm(len(x)).split(args.batch_size):
                batch = x[idx]
                loss = (model(batch[:, 0], batch[:, 1]) - batch[:, 2].float()).square().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            result = metrics(model, 'valid')
            history.append({'epoch': epoch + 1, **result})
            print(json.dumps(history[-1]))
            score = result[f'ndcg@{args.selection_k}']
            if score is None:
                raise ValueError('No validation users with eligible positives')
            if score > best:
                best = score
                torch.save({'config': config, 'state': model.state_dict()}, args.run / 'best.pt')
        (args.run / 'history.json').write_text(json.dumps(history, indent=2))
        (args.run / 'config.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
        return
    cp = torch.load(args.run / args.checkpoint, map_location='cpu', weights_only=True)
    model = FM(**cp['config'])
    model.load_state_dict(cp['state'])
    if args.command == 'edit':
        if args.patch is None:
            p.error('edit requires --patch')
        patch = json.loads(args.patch.read_text())
        # Declarative bounded tensor edits only; no executable agent code.
        if len(patch) > 10000:
            raise ValueError('patch exceeds 10000 scalar edits')
        state = model.state_dict()
        for edit in patch:
            tensor = state[edit['parameter']]
            index = tuple(edit['index'])
            if len(index) != tensor.ndim or any(j < 0 or j >= s for j, s in zip(index, tensor.shape)):
                raise ValueError('invalid index')
            value = float(edit['value'])
            if not np.isfinite(value):
                raise ValueError('non-finite edit')
            tensor[index] = value
        model.load_state_dict(state)
        torch.save({'config': cp['config'], 'state': state}, args.run / 'edited.pt')
        (args.run / 'patch.json').write_text(json.dumps(patch, indent=2))
        print(json.dumps(metrics(model, 'valid')))
    else:
        result = metrics(model, args.split)
        (args.run / f'{Path(args.checkpoint).stem}-{args.split}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result))


if __name__ == '__main__':
    main()
