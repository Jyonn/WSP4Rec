"""BPR matrix factorization with a frozen Llama-3 text-feature branch.

Only catalogue metadata enters PCA. Training/selection never opens test labels.
All saved learned arrays are CPU tensors; frozen text features are separate.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from .cli import FM
from .study import quick_metric, save

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/ml-100k-description-complete-v3'
OUT = ROOT / os.environ.get('WSPREC_TEXT_RUN_DIR', 'runs/v7_text')
SEEDS = (42, 43, 44)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TextMF(FM):
    """score(u,i) = global + user_bias + item_bias + U[u] dot (V[i]+X[i]W)."""
    def __init__(self, users, items, dim, text_dim=64):
        super().__init__(users, items, dim)
        self.text_projection = nn.Parameter(torch.empty(text_dim, dim))
        nn.init.normal_(self.text_projection, std=.01)

    def forward(self, u, i, features):
        effective = self.item(i) + features[i] @ self.text_projection
        return (self.bias + self.user_bias(u).squeeze(-1)
                + self.item_bias(i).squeeze(-1)
                + (self.user(u) * effective).sum(-1))


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def effective_items(state, features=None):
    """CPU float-precision model factors, preserving the learned array dtype."""
    item = _tensor(state['item.weight'])
    if 'text_projection' not in state:
        return item
    text = torch.as_tensor(features, dtype=item.dtype, device='cpu')
    return item + text @ _tensor(state['text_projection']).to(item.dtype)


@torch.no_grad()
def scores(state, features=None):
    """Full-catalogue scores; global/user offsets included for audit clarity."""
    item = effective_items(state, features)
    user = _tensor(state['user.weight'])
    value = user @ item.T + _tensor(state['item_bias.weight']).T
    if 'user_bias.weight' in state:
        value = value + _tensor(state['user_bias.weight'])
    if 'bias' in state:
        value = value + _tensor(state['bias'])
    return value.numpy().astype(np.float64)


def fit_pca(embedding, rank=64):
    """Deterministic sign convention; no whitening; L2-normalized projected rows."""
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError('finite embedding matrix required')
    if rank < 1 or rank > min(embedding.shape):
        raise ValueError('invalid PCA rank')
    mean = embedding.mean(0)
    centered = embedding - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    components = right[:rank].copy()
    largest = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(rank), largest])
    signs[signs == 0] = 1
    components *= signs[:, None]
    projected = centered @ components.T
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    features = (projected / np.maximum(norms, 1e-12)).astype(np.float32)
    variance_fraction = float(np.square(singular[:rank]).sum() / np.square(singular).sum())
    return features, {'mean': mean, 'components': components, 'singular_values': singular[:rank],
                      'explained_variance_fraction': variance_fraction}


def preprocess(rank=64):
    dest = OUT / 'text'
    if (dest / 'metadata.json').exists():
        raise FileExistsError('text preprocessing already complete')
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source = DATA / 'title_description.npy'
    raw = np.load(source, allow_pickle=False)
    movies = load(DATA / 'movies.json')
    mapping = load(DATA / 'mapping.json')
    if raw.shape != (len(movies), 4096):
        raise ValueError('unexpected catalogue/text embedding alignment')
    assert [m['item_id'] for m in movies] == list(range(len(movies)))
    original = ROOT / 'runs/followup/llama3/title_description.npy'
    alignment = 'Mapping and row IDs checked; original cache unavailable for byte-subset check'
    if original.exists():
        original_raw = np.load(original, allow_pickle=False)
        assert np.array_equal(raw, original_raw[np.asarray(mapping['item_new_to_old'])])
        alignment = 'All1568rows byte-identical to original4096D cache subset by item_new_to_old'
    if not all(m.get('description', '').strip() and m['description'].strip().lower() != 'no overview found.' for m in movies):
        raise ValueError('missing or placeholder description')
    features, pca = fit_pca(raw, rank)
    np.save(dest / 'features.npy', features, allow_pickle=False)
    np.savez(dest / 'pca.npz', **{k: v for k, v in pca.items() if isinstance(v, np.ndarray)})
    metadata = {'source': str(source.relative_to(ROOT)), 'source_sha256': digest(source),
                'raw_shape': list(raw.shape), 'feature_shape': list(features.shape),
                'features_sha256': digest(dest / 'features.npy'), 'pca_sha256': digest(dest / 'pca.npz'),
                'movies_sha256': digest(DATA / 'movies.json'), 'mapping_sha256': digest(DATA / 'mapping.json'),
                'original_cache_alignment': alignment, 'pca_centering': True, 'pca_whitening': False,
                'component_sign': 'largest absolute loading is nonnegative', 'projected_row_normalization': 'L2',
                'pca_fitting_scope': 'All retained catalogue metadata only; no interaction or validation/test labels used.',
                'explained_variance_fraction': pca['explained_variance_fraction'],
                'llama_metadata_sha256': digest(ROOT / 'reports/followup/llama3_embedding_metadata.json'),
                'wall_seconds': time.monotonic() - started}
    save(dest / 'metadata.json', metadata)
    preprocess_shuffled()
    print(json.dumps(metadata), flush=True)
    return metadata


def feature_path(condition='text'):
    if condition not in ('none', 'text', 'shuffled_text'):
        raise ValueError('invalid text condition')
    return OUT / 'text' / ('shuffled_features.npy' if condition == 'shuffled_text' else 'features.npy')


def preprocess_shuffled():
    """Fixed metadata-only row shuffle, same feature values/geometry and size."""
    dest = OUT / 'text'
    features = np.load(dest / 'features.npy', allow_pickle=False)
    permutation = np.random.default_rng(2026110741).permutation(len(features))
    path = dest / 'shuffled_features.npy'
    if (dest / 'shuffled_metadata.json').exists():
        assert np.array_equal(np.load(path, allow_pickle=False), features[permutation])
        return load(dest / 'shuffled_metadata.json')
    np.save(path, features[permutation], allow_pickle=False)
    np.save(dest / 'shuffled_permutation.npy', permutation, allow_pickle=False)
    meta = {'random_seed': 2026110741, 'rule': 'Apply fixed anonymous item-row permutation to PCA features; interactions/model IDs are unchanged.',
            'source_features_sha256': digest(dest / 'features.npy'),
            'features_sha256': digest(path), 'permutation_sha256': digest(dest / 'shuffled_permutation.npy'),
            'shape': list(features.shape), 'selection': 'No interaction or validation/test label used; no tuning.',
            'interpretation': 'Matched-dimension and same-feature-distribution control; breaks movie-content correspondence without changing architecture or trainable parameter count.'}
    save(dest / 'shuffled_metadata.json', meta)
    return meta


def _cpu_state(model):
    return {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}


def train_one(seed, lr, wd, device='cpu', max_epochs=240, patience=40, condition='text'):
    if condition not in ('none', 'text', 'shuffled_text'):
        raise ValueError('condition must be none, text or shuffled_text')
    folder = OUT / condition / 'training' / f's{seed}-lr{lr}-wd{wd}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'history.json').exists():
        return load(folder / 'history.json')
    started = time.monotonic()
    torch.manual_seed(seed)
    train, valid = np.load(DATA / 'train.npy'), np.load(DATA / 'valid.npy')
    counts = load(DATA / 'manifest.json')['counts']['after']
    features = np.load(feature_path(condition))
    users, items, dim, text_dim = counts['users'], counts['items'], 16, features.shape[1]
    model = (TextMF(users, items, dim, text_dim) if condition != 'none' else FM(users, items, dim)).to(device)
    text = torch.as_tensor(features, device=device)
    def predict(u, i):
        return model(u, i, text) if condition != 'none' else model(u, i)
    x = torch.as_tensor(train, dtype=torch.long, device=device)
    positives = x[x[:, 2] >= 4]
    known = torch.zeros((users, items), dtype=torch.bool, device=device)
    known[x[:, 0], x[:, 1]] = True
    if (known.sum(1)[positives[:, 0]] == items).any():
        raise ValueError('no eligible BPR negatives')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    history, best, best_epoch = [], -np.inf, 0
    initial = _cpu_state(model)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for ix in torch.randperm(len(positives), device=device).split(2048):
            batch = positives[ix]
            negative = torch.randint(items, (len(batch),), device=device)
            bad = known[batch[:, 0], negative]
            while bad.any():
                negative[bad] = torch.randint(items, (int(bad.sum()),), device=device)
                bad = known[batch[:, 0], negative]
            positive_score = predict(batch[:, 0], batch[:, 1])
            negative_score = predict(batch[:, 0], negative)
            loss = torch.nn.functional.softplus(-(positive_score - negative_score)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        state = _cpu_state(model)
        # Use the same CPU evaluator, even for CUDA-trained models.
        value = quick_metric(scores(state, features), valid, train)
        history.append({'epoch': epoch, 'valid_ndcg@10': value, 'last_batch_loss': float(loss.detach())})
        if value > best:
            best, best_epoch = value, epoch
            config = {'users': users, 'items': items, 'dim': dim}
            if condition != 'none':
                config['text_dim'] = text_dim
            torch.save({'config': config,
                        'state': state, 'epoch': epoch,
                        'feature_sha256': digest(feature_path(condition))}, folder / 'best.pt')
        if epoch - best_epoch >= patience:
            break
        if time.monotonic() - started > 3500:
            raise TimeoutError('single training trial budget')
    saved = torch.load(folder / 'best.pt', weights_only=True, map_location='cpu')['state']
    result = {'condition': condition, 'seed': seed, 'lr': lr, 'wd': wd, 'best_epoch': best_epoch, 'epochs_run': epoch,
              'max_epochs': max_epochs, 'patience': patience, 'batch_size': 2048,
              'valid_ndcg@10': best, 'wall_seconds': time.monotonic() - started,
              'checkpoint': str((folder / 'best.pt').relative_to(ROOT)), 'device': device,
              'text_projection_l2_change_from_init': float(torch.linalg.vector_norm(saved['text_projection'] - initial['text_projection'])) if condition != 'none' else 0.,
              'text_projection_l2': float(torch.linalg.vector_norm(saved['text_projection'])) if condition != 'none' else 0.,
              'text_item_contribution_l2': float(torch.linalg.vector_norm(torch.as_tensor(features) @ saved['text_projection'])) if condition != 'none' else 0.,
              'item_id_factor_l2': float(torch.linalg.vector_norm(saved['item.weight'])),
              'history': history}
    save(folder / 'history.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)
    return result


def train_all(device='cpu', condition='text'):
    if (OUT / condition / 'training_summary.json').exists():
        raise FileExistsError('training complete')
    trials = [train_one(42, lr, wd, device, condition=condition) for lr in (.003, .01) for wd in (1e-5, .001)]
    choice = max(trials, key=lambda r: r['valid_ndcg@10'])
    selected = {'42': choice['checkpoint']}
    for seed in (43, 44):
        trial = train_one(seed, choice['lr'], choice['wd'], device, condition=condition)
        trials.append(trial)
        selected[str(seed)] = trial['checkpoint']
    save(OUT / condition / 'training_summary.json', {'condition': condition, 'trials': trials, 'selected': selected,
          'feature_sha256': digest(feature_path(condition)), 'dataset_manifest_sha256': digest(DATA / 'manifest.json'),
          'torch': torch.__version__, 'numpy': np.__version__, 'device': device,
          'architecture': 'bias+user_bias+item_bias+U dot (V+XW); X frozen metadata PCA; U,V,W and biases trained jointly with BPR BP' if condition != 'none' else 'ordinary biased MF, U,V and biases jointly trained with BPR BP'})


def cached_svd(matrix, rank):
    cache = OUT / 'spectral_cache'
    cache.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(matrix.tobytes()).hexdigest()
    path = cache / f'{h}-k{rank}.npz'
    started = time.monotonic()
    hit = path.exists()
    if hit:
        with np.load(path) as z:
            result = tuple(z[k].copy() for k in ('u', 's', 'v'))
    else:
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        result = left[:, :rank].copy(), singular[:rank].copy(), right[:rank].copy()
        np.savez(path, u=result[0], s=result[1], v=result[2])
    with (cache / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps({'matrix_sha256': h, 'rank': rank, 'cache_hit': hit,
                                'wall_seconds': time.monotonic() - started}) + '\n')
    return result


def simple_target(state, features, train, svd=None):
    """The pre-existing classical target applied to effective text+ID factors."""
    U = _tensor(state['user.weight']).numpy().astype(float)
    V = effective_items(state, features).numpy().astype(float)
    x = np.zeros((len(U), len(V)))
    u, i = train[:, :2].astype(int).T
    x[u, i] = .38 + .20 * (train[:, 2] - 1)
    observed = np.zeros_like(x)
    observed[u, i] = 1
    x /= (observed.sum(1, keepdims=True) + 12)**.5
    x /= (observed.sum(0) + 12)**.5
    x -= x.mean(0)
    left, singular, right = (svd or cached_svd)(x, U.shape[1])
    F, G = left * np.sqrt(singular), right.T * np.sqrt(singular)
    F *= np.linalg.norm(U) / max(np.linalg.norm(F), 1e-30)
    G *= np.linalg.norm(V) / max(np.linalg.norm(G), 1e-30)
    left, _, right = np.linalg.svd(F.T @ U + G.T @ V)
    rotation = left @ right
    pop = np.log(observed.sum(0) + 5)
    pop -= pop.mean()
    return F @ rotation, G @ rotation, pop


def simple_fusion_delta(state, features, train, alpha=.5, beta=.25, target=None, svd=None):
    """Preserve W; changing ID residual realizes blending of effective items.

    Joint clipping uses the actual learned parameterization, including W, rather
    than counting the frozen catalogue features as editable model parameters.
    """
    F, G, pop = target if target is not None else simple_target(state, features, train, svd)
    U = _tensor(state['user.weight']).numpy().astype(float)
    V = effective_items(state, features).numpy().astype(float)
    delta = {'user.weight': alpha * (F - U), 'item.weight': alpha * (G - V),
             'item_bias.weight': beta * pop[:, None]}
    norm = np.sqrt(sum(np.square(_tensor(value).numpy().astype(float)).sum() for value in state.values()))
    change = np.sqrt(sum(np.square(value).sum() for value in delta.values()))
    factor = min(1., .495 * norm / max(change, 1e-30))
    return {name: (value * factor).astype(np.float32) for name, value in delta.items()}


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', required=True, choices=['preprocess', 'train'])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--condition', default='text', choices=['none', 'text', 'shuffled_text'])
    parser.add_argument('--rank', type=int, default=64)
    args = parser.parse_args()
    if args.command == 'preprocess':
        preprocess(args.rank)
    else:
        train_all(args.device, args.condition)
