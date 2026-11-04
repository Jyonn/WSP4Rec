"""Validation-selected non-LLM reference recommenders (no test access)."""
import time
import numpy as np
from .study import OUT, load_data, catalogue, quick_metric, save


def main():
    started = time.monotonic()
    train, valid = load_data()
    n_users, n_items = int(train[:, 0].max())+1, catalogue()
    x = np.zeros((n_users, n_items), dtype=np.float64)
    p = train[train[:, 2] >= 4]
    x[p[:, 0], p[:, 1]] = 1.
    popularity = np.broadcast_to(x.sum(0), x.shape).copy()
    (OUT / 'references').mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'references/popularity.npy', popularity)
    results = [{'id': 'popularity', 'valid_ndcg@10': quick_metric(popularity, valid, train)}]
    gram = x.T @ x
    best_value, best_scores, best_reg = -1., None, None
    for regularization in (100., 300., 1000., 3000.):
        t = time.monotonic()
        precision = np.linalg.inv(gram + regularization*np.eye(n_items))
        coefficients = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(coefficients, 0.)
        prediction = x @ coefficients
        value = quick_metric(prediction, valid, train)
        results.append({'id': f'ease-{regularization}', 'valid_ndcg@10': value, 'seconds': time.monotonic()-t})
        if value > best_value:
            best_value, best_scores, best_reg = value, prediction, regularization
    np.save(OUT / 'references/ease.npy', best_scores)
    save(OUT / 'references/validation.json', {'results': results, 'selected_ease_regularization': best_reg,
        'wall_seconds': time.monotonic()-started, 'positive_threshold': 4, 'device': 'cpu',
        'scope': 'contextual reference, not an information/budget-matched editor control'})
    print(results, flush=True)


if __name__ == '__main__':
    main()
