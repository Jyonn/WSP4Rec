# WSP4Rec

MovieLens-100K + a two-field Factorization Machine. With only user/item fields,
this FM is mathematically equivalent to biased matrix factorization. The initial
training objective remains explicit rating prediction (MSE). Primary evaluation
reports full-catalogue Recall, NDCG and HitRate at K=5/10/20; unclipped RMSE
and MAE are auxiliary. Checkpoints are selected by validation NDCG@10.
This is an MSE-trained ranking baseline, not a BPR-trained model.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
wsprec prepare
wsprec train --epochs 50
wsprec evaluate --split valid
wsprec edit --patch patch.json
wsprec evaluate --checkpoint edited.pt --split test
```

Data is downloaded from GroupLens. Supply `u.data`, `u.item` and run preparation
on a network-enabled machine if downloads are unavailable. The archive can also
be placed at `data/ml-100k/ml-100k.zip` before `prepare`.

Splits are deterministic per-user chronological 80/10/10, not globally temporal.
Train on train, select checkpoints and edits on valid, and reserve test for final
reporting. Do not repeatedly select edits against test. Movie titles are retained
in `u.item` but are not used by this ID-only baseline. Cite the MovieLens dataset
paper and follow GroupLens terms; raw data is not committed.

## Weight editing

A patch is a JSON list of absolute scalar assignments:

```json
[{"parameter": "item_bias.weight", "index": [0, 0], "value": 0.1}]
```

Parameters: `bias` (scalar; index `[]`), `user_bias.weight`, `item_bias.weight`,
`user.weight`, `item.weight`. Edits always start from `--checkpoint` (default
`best.pt`) and save `edited.pt`; use separate run directories per trajectory.
The edit command uses no gradients and evaluates validation only. Never pass
pickle checkpoints from untrusted sources. The baseline checkpoint is reusable.

## Planned experimental extensions

- Warm/low-support user-item groups defined from training counts only.
- Semantic anchors and title embeddings with matched information controls.
- LLM editor integration, feedback budgets and complete token/cost logs.
- Budget-matched random edits and deterministic transformation baselines.

The current framework deliberately keeps these extensions explicit rather than
reporting them as already implemented.

## Ranking protocol

Ratings >=4 are relevant (configurable with `--positive-threshold`). Metrics
are macro-averaged over users with at least one eligible held-out positive;
`ranking_users` reports the denominator. Users without positives are skipped;
empty evaluation returns null metrics. All catalogue items are candidates,
including unobserved items; no sampled negatives are used. Observed low ratings
are non-relevant, but unobserved items have unknown true relevance.
Validation excludes all training interactions, regardless of rating. Final test
excludes training and validation interactions. The catalogue is fixed across
splits and is not filtered by release time. Ties are broken by ascending item ID.

```bash
wsprec train --run runs/fm-ranking-seed42 --epochs 5
wsprec evaluate --run runs/fm-ranking-seed42 --split valid
python -m unittest discover -s tests -v
```

Existing checkpoints selected by RMSE can be evaluated with the new metrics,
but must not be described as selected by NDCG; train into a new run directory.
