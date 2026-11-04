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

## Completed exploratory study (2026-11-04)

[Chinese research report](output/pdf/WSP4Rec_research_report.pdf),
[aggregate results](reports/study_summary.json), [cost record](reports/costs.json),
and [tables](reports/tables/) document the first controlled study. It includes
MSE/BPR MF, three training seeds, eight independent Astra editor conditions
(two objectives x weight visibility x title access), 64 proposed candidates,
calibration/random/title-smoothing controls, train-count groups, collateral harm,
and contextual popularity/EASE references. Every editor used two rounds of four
validation queries. Seed42 selected recipes transfer unchanged to seeds43/44.

Mean test NDCG@10 for base / no-title WSP / WSP+titles is
0.05801 / 0.07328 / 0.07534 for MSE and
0.10441 / 0.10490 / 0.10566 for BPR. No-title BPR WSP selected the same bias
scaling as the calibration control. EASE scores 0.12031. These results do **not**
establish a unique benefit from LLM weight access, cold-start effectiveness or
competitive recommendation performance. Title smoothing has a small BPR signal;
no low-support-item Recall@10 improvement was observed in the selected editors.

Native Astra token usage and per-call subscription charges are unavailable and
explicitly null, not zero. CPU compute walls, query counts and Qwen GPU memory
and extraction time are recorded. Query-matched controls are not fully matched
in search space or monetary cost. See the report for isolation and timing limits.

## Study reproduction

The study protocol is in `study_protocol.json`; pre-test clarifications are in
`study_amendments.json`, and the common editor interface is
`editor_interface.json`. The final manifest hashes inputs before test labels
are read. Historical CLI scoring and study matrix-product scoring are
mathematically equivalent but not bitwise identical at float32 near ties.

```bash
pip install -e '.[text,report]'
wsprec prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m wsprec.study train
python -m wsprec.study export
python -m wsprec.title_embeddings --titles runs/study/text/titles.json \
  --output runs/study/text/title_embeddings.npy \
  --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --device cuda:0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/replay_study.py
python scripts/build_study_report.py
```

`replay_study.py` uses archived Astra recipes; it does not regenerate model
responses. It refuses to overwrite an existing frozen study. Exact numbers
can vary across Torch/BLAS versions. For new discoveries, use isolated Astra
sessions with the exported per-condition allowlists; return only candidate IDs
and validation NDCG as feedback. No arbitrary agent code, fitting or test access
is part of the editing interface. The report is a research report, not a
submission-ready four-page paper.

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
