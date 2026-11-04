"""NumPy-only, full-catalogue metrics for controlled recommendation studies.

Only users with eligible held-out positives contribute to macro averages and
exposure. Seen items are removed from both candidates and held-out positives.
Repeated positive rows denote one relevant item. Popularity counts include every
training row, regardless of its rating. No grouping uses held-out information.
"""
from collections.abc import Mapping
import numpy as np


USER_GROUPS = ('low', 'mid', 'high')
ITEM_GROUPS = ('zero', 'low', 'mid', 'high')


def _rows(rows, shape, name):
    rows = np.asarray(rows)
    if rows.size == 0:
        return np.empty((0, 4), dtype=float)
    if rows.ndim != 2 or rows.shape[1] < 3:
        raise ValueError(f'{name} must contain [user, item, rating, ...] rows')
    rows = rows.astype(float)
    if not np.isfinite(rows[:, :3]).all():
        raise ValueError(f'{name} IDs and ratings must be finite')
    for column, limit in enumerate(shape):
        ids = rows[:, column]
        if np.any(ids != np.floor(ids)) or np.any(ids < 0) or np.any(ids >= limit):
            raise ValueError(f'{name} contains invalid user or item IDs')
    return rows


def _quantile_groups(counts, positive_only=False):
    population = counts[counts > 0] if positive_only else counts
    thresholds = np.quantile(population, [1 / 3, 2 / 3]) if population.size else np.zeros(2)
    labels = np.where(counts <= thresholds[0], 0, np.where(counts <= thresholds[1], 1, 2))
    if positive_only:
        labels = np.where(counts == 0, 0, labels + 1)
    return labels, thresholds


def _aggregate(values, positives, k):
    values = np.asarray(values, dtype=float).reshape(-1, 3)
    means = values.mean(axis=0) if len(values) else [None] * 3
    return {f'{name}@{k}': float(value) if value is not None else None
            for name, value in zip(('ndcg', 'recall', 'hit_rate'), means)} | {
                'ranking_users': len(values), 'positives': int(positives)}


def rank_metrics(scores, heldout, seen, train, k=10, threshold=4):
    """Evaluate full-catalogue rankings, with train-only activity/popularity bins.

    ``scores`` has shape (users, items); rows contain user, item, rating, time.
    ``per_user`` is a column dictionary of NumPy arrays; ``group`` is 0/1/2
    for low/mid/high training activity. Groups average only users with eligible
    positives in that group. Item-group NDCG retains *global* ranking discounts
    and uses the group's positive count for IDCG (no candidate reranking).
    ``exposure`` gives each item group's share of recommendation slots, and
    ``catalogue_coverage`` divides unique recommended items by all score columns.
    Empty averages/slot shares are None; empty catalogue coverage is zero.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError('scores must be a finite two-dimensional array')
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('k must be a positive integer')
    if not np.isfinite(threshold):
        raise ValueError('threshold must be finite')
    heldout = _rows(heldout, scores.shape, 'heldout')
    seen = _rows(seen, scores.shape, 'seen')
    train = _rows(train, scores.shape, 'train')
    n_users, n_items = scores.shape
    user_counts = np.bincount(train[:, 0].astype(int), minlength=n_users)
    item_counts = np.bincount(train[:, 1].astype(int), minlength=n_items)
    user_bins, user_thresholds = _quantile_groups(user_counts)
    item_bins, item_thresholds = _quantile_groups(item_counts, positive_only=True)
    known, relevant = {}, {}
    for u, i, *_ in seen:
        known.setdefault(int(u), set()).add(int(i))
    for u, i, rating, *_ in heldout:
        if rating >= threshold:
            relevant.setdefault(int(u), set()).add(int(i))
    discounts = 1 / np.log2(np.arange(min(k, n_items)) + 2)
    values, ids, labels, positive_counts = [], [], [], []
    item_values = {name: [] for name in ITEM_GROUPS}
    item_positives = dict.fromkeys(ITEM_GROUPS, 0)
    cross_values = {u: {i: [] for i in ITEM_GROUPS} for u in USER_GROUPS}
    cross_positives = {u: dict.fromkeys(ITEM_GROUPS, 0) for u in USER_GROUPS}
    exposure_counts = np.zeros(4, dtype=int)
    recommended = set()
    all_items = np.arange(n_items)

    def metrics(order, targets):
        hits = np.isin(order, list(targets)).astype(float)
        ideal = discounts[:min(k, len(targets))].sum()
        return [float(hits @ discounts[:len(order)] / ideal),
                float(hits.sum() / len(targets)), float(hits.any())]

    for u in sorted(relevant):
        excluded = known.get(u, set())
        targets = relevant[u] - excluded
        if not targets:
            continue
        candidates = all_items[~np.isin(all_items, list(excluded))]
        order = candidates[np.lexsort((candidates, -scores[u, candidates]))][:k]
        group = int(user_bins[u])
        ids.append(u)
        labels.append(group)
        values.append(metrics(order, targets))
        positive_counts.append(len(targets))
        exposure_counts += np.bincount(item_bins[order], minlength=4)
        recommended.update(order.tolist())
        for index, name in enumerate(ITEM_GROUPS):
            group_targets = {i for i in targets if item_bins[i] == index}
            if not group_targets:
                continue
            result = metrics(order, group_targets)
            item_values[name].append(result)
            item_positives[name] += len(group_targets)
            cross_values[USER_GROUPS[group]][name].append(result)
            cross_positives[USER_GROUPS[group]][name] += len(group_targets)

    values = np.asarray(values, dtype=float).reshape(-1, 3)
    labels = np.asarray(labels, dtype=int)
    positive_counts = np.asarray(positive_counts, dtype=int)
    slots = int(exposure_counts.sum())
    return {
        'summary': _aggregate(values, positive_counts.sum(), k),
        'per_user': {'user_id': np.asarray(ids, dtype=int), 'ndcg': values[:, 0],
                     'recall': values[:, 1], 'hit_rate': values[:, 2], 'group': labels},
        'groups': {name: _aggregate(values[labels == j], positive_counts[labels == j].sum(), k)
                   for j, name in enumerate(USER_GROUPS)},
        'item_groups': {name: _aggregate(item_values[name], item_positives[name], k)
                        for name in ITEM_GROUPS},
        'user_item_groups': {
            u: {i: _aggregate(cross_values[u][i], cross_positives[u][i], k)
                for i in ITEM_GROUPS} for u in USER_GROUPS},
        'exposure': {name: float(exposure_counts[j] / slots) if slots else None
                     for j, name in enumerate(ITEM_GROUPS)},
        'exposure_counts': dict(zip(ITEM_GROUPS, exposure_counts.tolist())),
        'recommendation_slots': slots,
        'catalogue_coverage': len(recommended) / n_items if n_items else 0.,
        'grouping': {'user_counts': user_counts, 'item_counts': item_counts,
                     'user_groups': user_bins, 'item_groups': item_bins,
                     'user_thresholds': user_thresholds, 'item_thresholds': item_thresholds,
                     'user_group_names': USER_GROUPS, 'item_group_names': ITEM_GROUPS},
    }


def paired_bootstrap(base, edited, repetitions=2000, seed=2026):
    """Bootstrap paired per-user NDCG deltas, aligning by exact user ID sets.

    Inputs are rank_metrics(...)["per_user"] column dictionaries (lists work).
    Returns a percentile 95% CI; proportions use exact delta comparisons.
    Missing/duplicate users, empty samples, and invalid numbers raise ValueError.
    """
    if isinstance(repetitions, (bool, np.bool_)) or not isinstance(repetitions, (int, np.integer)) or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')

    def columns(data):
        if not isinstance(data, Mapping) or 'user_id' not in data or 'ndcg' not in data:
            raise ValueError('per_user must contain user_id and ndcg columns')
        ids = np.asarray(data['user_id'], dtype=float)
        ndcg = np.asarray(data['ndcg'], dtype=float)
        if ids.ndim != 1 or ndcg.ndim != 1 or len(ids) != len(ndcg) or not len(ids):
            raise ValueError('user_id and ndcg must be nonempty equal-length vectors')
        if not np.isfinite(ids).all() or np.any(ids != np.floor(ids)) or np.any(ids < 0):
            raise ValueError('user IDs must be nonnegative finite integers')
        if len(np.unique(ids)) != len(ids) or not np.isfinite(ndcg).all():
            raise ValueError('user IDs must be unique and NDCG must be finite')
        order = np.argsort(ids)
        return ids[order], ndcg[order]

    base_ids, base_values = columns(base)
    edited_ids, edited_values = columns(edited)
    if not np.array_equal(base_ids, edited_ids):
        raise ValueError('base and edited must contain exactly the same users')
    delta = edited_values - base_values
    rng = np.random.default_rng(seed)
    # Bound temporary memory independently of the number of repetitions.
    means = np.empty(repetitions)
    for index in range(repetitions):
        means[index] = delta[rng.integers(0, len(delta), size=len(delta))].mean()
    lower, upper = np.quantile(means, [.025, .975])
    return {'mean_delta': float(delta.mean()), 'ci95': [float(lower), float(upper)],
            'improved_fraction': float(np.mean(delta > 0)),
            'harmed_fraction': float(np.mean(delta < 0)),
            'unchanged_fraction': float(np.mean(delta == 0)),
            'ranking_users': len(delta), 'repetitions': int(repetitions), 'seed': seed}
