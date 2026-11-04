"""Full-catalogue binary-relevance ranking evaluation."""
import numpy as np
import torch


def evaluate_ranking(model, rows, seen, ks=(5, 10, 20), positive_threshold=4):
    ks = tuple(sorted(set(ks)))
    if not ks or min(ks) < 1:
        raise ValueError('K must be positive')
    relevant = {}
    for u, i, rating, *_ in rows:
        if rating >= positive_threshold:
            relevant.setdefault(int(u), set()).add(int(i))
    known = {}
    for u, i, *_ in seen:
        known.setdefault(int(u), set()).add(int(i))
    totals = {f'{name}@{k}': 0. for k in ks for name in ('recall', 'ndcg', 'hit_rate')}
    count = 0
    model.eval()
    items = torch.arange(model.item.num_embeddings)
    with torch.no_grad():
        for u, targets in sorted(relevant.items()):
            excluded = known.get(u, set())
            targets = targets - excluded
            if not targets:
                continue
            scores = model(torch.full_like(items, u), items).cpu().numpy()
            candidates = np.array([i for i in range(len(scores)) if i not in excluded], dtype=int)
            # Descending score; item ID breaks ties deterministically.
            order = candidates[np.lexsort((candidates, -scores[candidates]))][:max(ks)]
            for k in ks:
                hits = np.array([int(i) in targets for i in order[:k]], dtype=float)
                discounts = 1 / np.log2(np.arange(len(hits)) + 2)
                ideal = (1 / np.log2(np.arange(min(k, len(targets))) + 2)).sum()
                totals[f'recall@{k}'] += hits.sum() / len(targets)
                totals[f'ndcg@{k}'] += float((hits * discounts).sum() / ideal)
                totals[f'hit_rate@{k}'] += float(hits.any())
            count += 1
    return {**{key: value / count if count else None for key, value in totals.items()},
            'ranking_users': count, 'positive_threshold': positive_threshold,
            'candidate_protocol': 'full catalogue excluding known interactions'}
