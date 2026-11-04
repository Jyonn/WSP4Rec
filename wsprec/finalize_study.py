"""Freeze all validation choices, then evaluate the complete matrix once."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from .study import ROOT, OUT, DATA, save, selected_cp, scores, apply_recipe, edit_stats, load_data
from .study_metrics import rank_metrics, paired_bootstrap


CONDITIONS = ('weights', 'blind', 'weights_text', 'blind_text')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def freeze():
    destination = OUT / 'frozen_manifest.json'
    if destination.exists():
        raise FileExistsError('Manifest already frozen; do not overwrite')
    training = json.loads((OUT / 'training_summary.json').read_text())
    choices, hashes = {}, {}
    for objective in ('mse', 'bpr'):
        common = json.loads((OUT / f'bundles/{objective}/common.json').read_text())
        base_value = common['baseline_valid_ndcg@10']
        choices[objective] = {}
        for name in (*CONDITIONS, 'calibration', 'random', 'semantic'):
            if name in CONDITIONS:
                files = [OUT / f'editors/{objective}-{name}/round{r}.validation.json' for r in (1, 2)]
            else:
                files = [OUT / f'controls/{objective}-{name}.validation.json']
            candidates = []
            for file in files:
                data = json.loads(file.read_text())
                candidates.extend(data['results'])
                hashes[str(file.relative_to(ROOT))] = sha(file)
            if len(candidates) != 8:
                raise ValueError(f'{objective}/{name}: expected exactly8 queries')
            eligible = [c for c in candidates if c['valid_ndcg@10'] is not None and c['valid_ndcg@10'] > base_value]
            best = max(eligible, key=lambda c: (c['valid_ndcg@10'], -c['changed_parameters'])) if eligible else None
            choices[objective][name] = {'recipe': best['recipe'] if best else {'id': 'no_op', 'ops': []},
                'valid_ndcg@10': best['valid_ndcg@10'] if best else base_value,
                'feedback_queries': 8, 'valid_curve_best_so_far': [], 'candidate_files': [str(f.relative_to(ROOT)) for f in files]}
            running = base_value
            for c in candidates:
                if c['valid_ndcg@10'] is not None:
                    running = max(running, c['valid_ndcg@10'])
                choices[objective][name]['valid_curve_best_so_far'].append(running)
        for seed in (42, 43, 44):
            path = OUT / 'training' / training['selected'][objective][str(seed)] / 'best.pt'
            hashes[str(path.relative_to(ROOT))] = sha(path)
    for path in (ROOT / 'study_protocol.json', ROOT / 'study_amendments.json', ROOT / 'editor_interface.json', ROOT / 'wsprec/study.py', ROOT / 'wsprec/study_metrics.py', ROOT / 'wsprec/finalize_study.py', OUT / 'text/title_embeddings.npy', OUT / 'references/ease.npy', OUT / 'references/popularity.npy'):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    save(destination, {'created_unix': time.time(), 'choices': choices, 'sha256': hashes,
        'shuffled_text_diagnostic': 'Fixed RNG2026 permutation of Qwen item representations at edit compilation; same selected recipes, no feedback/reselection; tests graph correspondence, not a shuffled-prompt editor',
        'test_policy': 'all choices frozen before this runner reads test; recipes transfer to seeds43/44 without further selection',
        'reference_addition': 'Popularity and EASE were added as contextual references before test access, not matched editor controls',
        'selection_repeats': 'Nontext controls evaluated twice while awaiting text features; unique selection budget8 but physical repeated evaluation cost included separately'})
    print('Frozen:', destination, sha(destination))


def final():
    target = OUT / 'final_results.json'
    if target.exists():
        raise FileExistsError('Final evaluation already exists; do not reselect after test')
    start = time.monotonic()
    manifest_path = OUT / 'frozen_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for path, value in manifest['sha256'].items():
        if sha(ROOT / path) != value:
            raise ValueError(f'changed frozen input: {path}')
    train, valid = load_data()
    # First and only final-reporting pass decompresses the test labels here.
    with np.load(DATA / 'splits.npz') as z:
        test = z['test']
    seen = np.concatenate([train, valid])
    output = {'manifest_sha256': sha(manifest_path), 'objectives': {}, 'references': {}}
    for objective in ('mse', 'bpr'):
        output['objectives'][objective] = {}
        for seed in (42, 43, 44):
            cp = selected_cp(objective, seed)
            base = cp['state']
            base_eval = rank_metrics(scores(base), test, seen, train)
            methods = {'base': {'metrics': base_eval, 'recipe': {'id': 'no_op', 'ops': []},
                                'edit_stats': edit_stats(base, base)}}
            for method, selection in manifest['choices'][objective].items():
                t = time.monotonic()
                recipe = selection['recipe']
                state = apply_recipe(base, recipe, method.endswith('text') or method == 'semantic', int(recipe.get('random_seed', 2026)))
                result = rank_metrics(scores(state), test, seen, train)
                methods[method] = {'metrics': result, 'recipe': recipe, 'edit_stats': edit_stats(base, state),
                    'paired_bootstrap_vs_base': paired_bootstrap(base_eval['per_user'], result['per_user']),
                    'edit_and_evaluate_seconds': time.monotonic()-t}
                if method in ('weights_text', 'blind_text', 'semantic'):
                    features = np.load(OUT / 'text/title_embeddings.npy')
                    permutation = np.random.default_rng(2026).permutation(len(features))
                    shuffled = apply_recipe(base, recipe, True, text_features=features[permutation])
                    shuffled_eval = rank_metrics(scores(shuffled), test, seen, train)
                    methods[method + '_shuffled'] = {'metrics': shuffled_eval, 'recipe': recipe,
                        'edit_stats': edit_stats(base, shuffled),
                        'paired_bootstrap_vs_base': paired_bootstrap(base_eval['per_user'], shuffled_eval['per_user']),
                        'diagnostic': manifest['shuffled_text_diagnostic']}
            output['objectives'][objective][str(seed)] = methods
            print(objective, seed, {name: x['metrics']['summary']['ndcg@10'] for name, x in methods.items()}, flush=True)
    for name in ('popularity', 'ease'):
        prediction = np.load(OUT / f'references/{name}.npy')
        output['references'][name] = rank_metrics(prediction, test, seen, train)
    output['wall_seconds'] = time.monotonic()-start
    save(target, serial(output))
    print('Saved final report data:', target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'final'])
    args = parser.parse_args()
    freeze() if args.command == 'freeze' else final()
