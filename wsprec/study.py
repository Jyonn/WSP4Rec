"""Reproducible, validation-only discovery and sealed final evaluation.

Run train, export, controls, candidates; then use wsprec.finalize_study.
No test labels are loaded by train/export/controls/candidates.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from .cli import FM

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/study'
DATA = ROOT / 'data/ml-100k'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def load_data():
    # Opening a ZIP archive's directory does not decompress its test array.
    with np.load(DATA / 'splits.npz') as z:
        return z['train'], z['valid']


def catalogue():
    return len((DATA / 'u.item').read_text(encoding='latin-1').splitlines())


def scores(state):
    return (state['user.weight'] @ state['item.weight'].T
            + state['item_bias.weight'].T).numpy().astype(np.float64)


def quick_metric(s, rows, seen, k=10):
    s = s.copy()
    s[seen[:, 0], seen[:, 1]] = -np.inf
    relevant = np.zeros_like(s, dtype=bool)
    r = rows[rows[:, 2] >= 4]
    relevant[r[:, 0], r[:, 1]] = True
    relevant[seen[:, 0], seen[:, 1]] = False
    eligible = relevant.sum(1) > 0
    if not eligible.any():
        return None
    k = min(k, s.shape[1])
    top = np.argsort(-s, axis=1, kind='stable')[:, :k]
    hits = np.take_along_axis(relevant, top, axis=1)
    discount = 1 / np.log2(np.arange(k) + 2)
    ideal = np.array([discount[:min(k, n)].sum() for n in relevant.sum(1)])
    return float(((hits * discount).sum(1)[eligible] / ideal[eligible]).mean())


def checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def train_one(objective, seed, lr, wd, epochs=80, patience=12):
    started = time.monotonic()
    train, valid = load_data()
    torch.manual_seed(seed)
    np.random.seed(seed)
    users, items = int(train[:, 0].max()) + 1, catalogue()
    model = FM(users, items, 16)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    x = torch.tensor(train, dtype=torch.long)
    positives = x[x[:, 2] >= 4]
    known = torch.zeros((users, items), dtype=torch.bool)
    known[x[:, 0], x[:, 1]] = True
    if objective == 'bpr' and (known.sum(1)[positives[:, 0]] == items).any():
        raise ValueError('BPR user has no eligible negative items')
    best, best_epoch, best_state = -np.inf, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        batches = x if objective == 'mse' else positives
        losses = []
        for idx in torch.randperm(len(batches)).split(2048):
            b = batches[idx]
            if objective == 'mse':
                loss = (model(b[:, 0], b[:, 1]) - b[:, 2].float()).square().mean()
            else:
                negative = torch.randint(items, (len(b),))
                bad = known[b[:, 0], negative]
                while bad.any():
                    negative[bad] = torch.randint(items, (int(bad.sum()),))
                    bad = known[b[:, 0], negative]
                margin = model(b[:, 0], b[:, 1]) - model(b[:, 0], negative)
                loss = torch.nn.functional.softplus(-margin).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            value = quick_metric(scores(model.state_dict()), valid, train)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'valid_ndcg@10': value})
        if value > best:
            best, best_epoch = value, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
        if time.monotonic() - started > 3500:
            raise TimeoutError('training trial exceeded time budget')
    name = f'{objective}-s{seed}-lr{lr}-wd{wd}'
    folder = OUT / 'training' / name
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({'config': {'users': users, 'items': items, 'dim': 16}, 'state': best_state}, folder / 'best.pt')
    result = {'name': name, 'objective': objective, 'seed': seed, 'lr': lr, 'wd': wd,
              'best_epoch': best_epoch, 'epochs_run': len(history), 'valid_ndcg@10': best,
              'wall_seconds': time.monotonic() - started, 'device': 'cpu', 'history': history}
    save(folder / 'history.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)
    return result


def train():
    result = []
    selected = {}
    for objective in ('mse', 'bpr'):
        trials = []
        for lr in (.003, .01):
            for wd in (1e-5, .001):
                trials.append(train_one(objective, 42, lr, wd))
        best = max(trials, key=lambda x: x['valid_ndcg@10'])
        selected[objective] = {'42': best['name']}
        result.extend(trials)
        for seed in (43, 44):
            trial = train_one(objective, seed, best['lr'], best['wd'])
            selected[objective][str(seed)] = trial['name']
            result.append(trial)
        save(OUT / 'training_summary.json', {'trials': result, 'selected': selected,
             'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                             'numpy': np.__version__, 'platform': platform.platform(), 'threads': torch.get_num_threads()}})


def selected_cp(objective, seed):
    summary = json.loads((OUT / 'training_summary.json').read_text())
    return checkpoint(OUT / 'training' / summary['selected'][objective][str(seed)] / 'best.pt')


def counts():
    train, _ = load_data()
    return np.bincount(train[:, 0]), np.bincount(train[:, 1], minlength=catalogue())


def group_mask(count, group):
    if group == 'all':
        return np.ones(len(count), dtype=bool)
    q1, q2 = np.quantile(count[count > 0], [1/3, 2/3])
    return {'zero': count == 0, 'low': (count > 0) & (count <= q1),
            'mid': (count > q1) & (count <= q2), 'high': count > q2}[group]


def apply_recipe(base, recipe, text=False, random_seed=2026, text_features=None):
    """Compile algebraic operations to weights, without optimization or gradients.

    All operations are relative to the SAME original checkpoint, sequentially
    applied inside one recipe. Neighbours always come from original weights.
    """
    state = {k: v.detach().clone() for k, v in base.items()}
    uc, ic = counts()
    if len(recipe.get('ops', [])) > 5:
        raise ValueError('at most five operations')
    for op in recipe.get('ops', []):
        kind = op['kind']
        for field, low, high in (('factor', 0, 3), ('strength', 0, 1), ('coefficient', -1, 1)):
            if field in op and not low <= float(op[field]) <= high:
                raise ValueError(f'{field} outside shared bounds')
        if 'factors' in op and any(not 0 <= float(v) <= 3 for v in op['factors']):
            raise ValueError('dimension factors outside shared bounds')
        if 'neighbors' in op and (not isinstance(op['neighbors'], int) or not 1 <= op['neighbors'] <= 50):
            raise ValueError('invalid neighbor count')
        if kind == 'popularity_shift':
            c = np.log1p(ic).astype(float)
            c = (c - c.mean()) / max(c.std(), 1e-8)
            state['item_bias.weight'][:, 0] += float(op['coefficient']) * torch.tensor(c, dtype=torch.float32)
            continue
        if kind == 'random_noise':
            rng = np.random.default_rng(random_seed)
            for key in ('user.weight', 'item.weight', 'item_bias.weight'):
                a = state[key]
                noise = torch.tensor(rng.standard_normal(tuple(a.shape)), dtype=a.dtype)
                state[key] += float(op['strength']) * max(float(a.std()), 1e-6) * noise
            continue
        key = op['parameter']
        if key not in ('user.weight', 'item.weight', 'item_bias.weight'):
            raise ValueError('only ranking-effective parameters are editable')
        count = uc if key == 'user.weight' else ic
        mask = torch.tensor(group_mask(count, op.get('group', 'all')))
        a = state[key]
        if kind == 'scale':
            a[mask] *= float(op['factor'])
        elif kind == 'dimension_scale':
            factors = torch.tensor(op['factors'], dtype=a.dtype)
            if factors.shape != (a.shape[1],):
                raise ValueError('dimension mismatch')
            a[mask] *= factors
        elif kind == 'norm_shrink':
            strength = float(op['strength'])
            norm = a.norm(dim=1, keepdim=True).clamp_min(1e-8)
            target = float(op.get('target', float(norm.median())))
            if not np.isfinite(target) or target < 0:
                raise ValueError('target norm must be finite and nonnegative')
            a[mask] *= (1 - strength + strength * target / norm[mask])
        elif kind == 'support_shrink':
            tau = float(op['tau'])
            if not np.isfinite(tau) or tau < 0:
                raise ValueError('tau must be finite and nonnegative')
            factors = torch.tensor(count / (count + tau + 1e-12), dtype=a.dtype)[:, None]
            a[mask] *= factors[mask]
        elif kind == 'neighbor_smooth':
            source = op.get('source', 'weights')
            if key != 'item.weight':
                raise ValueError('neighbor smoothing only edits item embeddings')
            if source == 'text':
                if not text:
                    raise ValueError('text information forbidden in this condition')
                features = np.load(OUT / 'text/title_embeddings.npy') if text_features is None else text_features
            elif source == 'weights':
                features = base[key].numpy()
            else:
                raise ValueError('unknown source')
            features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
            similarity = features @ features.T
            np.fill_diagonal(similarity, -np.inf)
            # Zero-support neighbours cannot supply learned collaborative signal.
            similarity[:, ic == 0] = -np.inf
            neighbors = np.argsort(-similarity, axis=1, kind='stable')[:, :int(op.get('neighbors', 10))]
            sim = np.take_along_axis(similarity, neighbors, 1)
            weights = np.maximum(sim, 0)
            empty = weights.sum(1) == 0
            weights /= np.maximum(weights.sum(1, keepdims=True), 1e-8)
            smoothed = (base[key].numpy()[neighbors] * weights[:, :, None]).sum(1)
            smoothed[empty] = base[key].numpy()[empty]
            strength = float(op['strength'])
            a[mask] = (1-strength) * a[mask] + strength * torch.tensor(smoothed, dtype=a.dtype)[mask]
        else:
            raise ValueError(f'unsupported operation: {kind}')
    for key, value in state.items():
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite edit')
        if value.abs().max() > 100:
            raise ValueError('weight bound exceeded')
    return state


def edit_stats(base, state):
    changes = torch.cat([(state[k] - base[k]).flatten() for k in base])
    norms = torch.cat([v.flatten() for v in base.values()]).norm()
    return {'changed_parameters': int((changes != 0).sum()), 'total_parameters': len(changes),
            'delta_l2': float(changes.norm()), 'relative_delta_l2': float(changes.norm() / norms)}


def evaluate_candidates(path, objective='bpr', text=False):
    start = time.monotonic()
    train_rows, valid = load_data()
    recipes = json.loads(Path(path).read_text())
    if not isinstance(recipes, list) or len(recipes) > 8:
        raise ValueError('expected at most eight recipes')
    cp = selected_cp(objective, 42)
    results = []
    for i, recipe in enumerate(recipes):
        t = time.monotonic()
        try:
            state = apply_recipe(cp['state'], recipe, text, int(recipe.get('random_seed', 2026 + i)))
            results.append({'id': recipe['id'], 'recipe': recipe, 'valid_ndcg@10': quick_metric(scores(state), valid, train_rows),
                            **edit_stats(cp['state'], state), 'seconds': time.monotonic() - t, 'error': None})
        except (ValueError, KeyError, TypeError) as e:
            results.append({'id': recipe.get('id', str(i)), 'recipe': recipe, 'valid_ndcg@10': None, 'error': str(e), 'seconds': time.monotonic() - t})
    result = {'objective': objective, 'text': text, 'candidate_file': str(path), 'wall_seconds': time.monotonic()-start,
              'baseline_valid_ndcg@10': quick_metric(scores(cp['state']), valid, train_rows), 'results': results}
    save(Path(path).with_suffix('.validation.json'), result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def export():
    train_rows, valid = load_data()
    uc, ic = counts()
    for objective in ('mse', 'bpr'):
        cp = selected_cp(objective, 42)
        folder = OUT / 'bundles' / objective
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {k: v.numpy() for k, v in cp['state'].items() if k in ('user.weight', 'item.weight', 'item_bias.weight')}
        np.savez(folder / 'weights.npz', **arrays)
        save(folder / 'common.json', {'config': cp['config'], 'objective': objective,
             'user_train_counts': uc.tolist(), 'item_train_counts': ic.tolist(),
             'baseline_valid_ndcg@10': quick_metric(scores(cp['state']), valid, train_rows),
             'user_count_quantiles': np.quantile(uc, [0, 1/3, 2/3, 1]).tolist(),
             'item_count_quantiles': np.quantile(ic[ic > 0], [0, 1/3, 2/3, 1]).tolist()})
    titles = [line.split('|')[1] for line in (DATA / 'u.item').read_text(encoding='latin-1').splitlines()]
    save(OUT / 'text/titles.json', [{'item_id': i, 'title': title} for i, title in enumerate(titles)])


def controls():
    folder = OUT / 'controls'
    calibrations = [
        {'id': f'bias_scale_{v}', 'ops': [{'kind': 'scale', 'parameter': 'item_bias.weight', 'factor': v}]} for v in (0., .5, 1.5, 2.)]
    calibrations += [{'id': f'pop_shift_{v}', 'ops': [{'kind': 'popularity_shift', 'coefficient': v}]} for v in (-.2, -.1, .1, .2)]
    # A fixed random control over FOUR operation families (a narrower search
    # space than Astra's compositions, dimensions and neighbour transformations).
    rng = np.random.default_rng(2026)
    random = []
    for i in range(8):
        key = str(rng.choice(['user.weight', 'item.weight', 'item_bias.weight']))
        group = str(rng.choice(['all', 'low', 'mid', 'high']))
        kind = i % 4
        if kind == 0:
            op = {'kind': 'scale', 'parameter': key, 'group': group, 'factor': float(rng.uniform(.5, 1.5))}
        elif kind == 1:
            op = {'kind': 'popularity_shift', 'coefficient': float(rng.uniform(-.3, .3))}
        elif kind == 2:
            op = {'kind': 'support_shrink', 'parameter': key, 'group': group, 'tau': float(rng.uniform(1, 30))}
        else:
            op = {'kind': 'norm_shrink', 'parameter': key, 'group': group, 'strength': float(rng.uniform(0, .5))}
        random.append({'id': f'random_{i}', 'ops': [op]})
    semantic = [{'id': f'title_{group}_{v}', 'ops': [{'kind': 'neighbor_smooth', 'parameter': 'item.weight', 'source': 'text', 'group': group, 'neighbors': 10, 'strength': v}]} for group in ('all', 'low') for v in (.05, .1, .25, .5)]
    for obj in ('mse', 'bpr'):
        for name, recipes, text in (('calibration', calibrations, False), ('random', random, False), ('semantic', semantic, True)):
            path = folder / f'{obj}-{name}.json'
            save(path, recipes)
            if text and not (OUT / 'text/title_embeddings.npy').exists():
                continue
            evaluate_candidates(path, obj, text)


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'export', 'controls', 'candidates'])
    parser.add_argument('--path', type=Path)
    parser.add_argument('--objective', default='bpr', choices=['mse', 'bpr'])
    parser.add_argument('--text', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == 'train':
        train()
    elif args.command == 'export':
        export()
    elif args.command == 'controls':
        controls()
    else:
        evaluate_candidates(args.path, args.objective, args.text)


if __name__ == '__main__':
    main()
