# WSP4Rec

The latest experiment compares **BP-trained BPR** with **the same checkpoint
edited by Astra with full weights, training interactions and explicit ID-to-row
mappings**. The updated [interactive explanation](output/web/WSP4Rec_interactive.html)
shows only these two systems. Previous blind/text experiments below are historical.


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

## Description and Llama-3 follow-up (2026-11-06)

The frozen first study is unchanged. New metadata preparation retrieves genuine
movie overviews from the public
[movie_descriptors dataset](https://huggingface.co/datasets/mt0rm0/movie_descriptors),
pinning revision and content hashes. Conservative title/alias/year matching
initially reported 1,570/1,682 matches (93.3%); later QA identified two of these
as explicit missing-overview placeholders. The corrected reliable count is 1,568.
The original matching output is preserved historically. The other 112 stayed explicitly missing
or ambiguous; no fuzzy match or generated plot is silently substituted.
See [coverage](reports/followup/description_coverage.json) and
[identity matches](reports/followup/description_matches.csv).

`wsprec.llama_movie_embeddings` extracts frozen **Meta-Llama-3-8B** final-layer
representations for both title-only and title+description using the same model
and prompt. It uses the final valid token of a fixed readout suffix, L2
normalization, and a 512-token limit that truncates only description content.
This is a prompted base language model, not a purpose-trained embedding model.
The extraction uses locally cached, access-authorized model weights only.
Missing descriptions use exactly the title-only representation plus a mask.
The full prompt and comparison scope are fixed in
`followup_protocol_2026-11-06.json`.
Both 1,682 x 4,096 arrays have been extracted and verified. The final pass took
47.62 seconds on one A100 with 15.54 GB peak allocated GPU memory; two completed
passes including the superseded numerical-QA version totaled 106.95 seconds.
See the [follow-up findings](reports/followup/findings_2026-11-06.txt),
[embedding metadata](reports/followup/llama3_embedding_metadata.json),
[quality checks](reports/followup/llama3_quality_checks.json), and
[execution attempts](reports/followup/execution_attempts.json).
Description coverage is lower in zero/low-support items; base-model vectors
also show high random-pair cosine similarity. Extraction is not evidence of
better recommendation performance or a validated semantic similarity metric.

```bash
pip install -e '.[text,descriptions]'
# Download descriptors_data.parquet and source API metadata into
# data/movie_descriptions/ from the pinned public source.
python -m wsprec.movie_descriptions
CUDA_VISIBLE_DEVICES=0 timeout 3500 python -m wsprec.llama_movie_embeddings \
  --movies data/movie_descriptions/matched_movies.json \
  --output runs/followup/llama3
OPENBLAS_NUM_THREADS=1 python scripts/check_followup_embeddings.py
```

The [Astra motivation audit](reports/followup/astra_motivation_audit.json) finds
bias/support hypotheses already present in all eight first-round proposals,
before their candidate feedback. Feedback then selected direction and magnitude.
However, the shared interface already illustrated bias scaling and popularity
adjustment, and parent prompts contained bias cues. This is **prior hypotheses
under designer-provided cues followed by feedback selection**, not evidence of
unprompted discovery. BPR often reduced learned item bias; all selected MSE
trajectories added a positive support prior. Ranking gains do not establish
popularity debiasing. The audit concerns saved explicit rationales and diagnostics,
not unavailable private internal reasoning.

No recommendation/test score is inferred from representation extraction alone.
Future comparisons must separate encoder replacement from added descriptions,
account for missing-description coverage, and freeze a new evaluation protocol.

## Description-complete neutral rerun (2026-11-06)

The corrected cohort has 943 users, 1,568 films and 97,445 interactions.
Filtering removes 114 films and 2,555 interactions before rebuilding per-user
chronological splits (77,573 train / 9,691 validation / 10,181 test).
Retained titles have reliable matches in one pinned public source; removal does
not establish that a description is unavailable everywhere. Two explicit
`No overview found.` placeholders invalidate the intermediate 1,570-film run,
which is preserved under `reports/v2_neutral/` with its status and full cost record.

Protocol: `neutral_protocol_v3.json`; shared generic interface:
`editor_interface_v2.json`; exact instruction template: `editor_prompt_v2.txt`.
Fresh native Astra editors see only their allowlisted model specification,
observation counts and condition-specific weights/text. They receive no prior
results or directional hypotheses and submit only numeric patches, without
requested hypotheses or rationales. The interface provides generic arithmetic
operations over all actual parameters. It is allowlist-based instruction
isolation, not an operating-system security boundary.

Each of 24 editor trajectories has two rounds of four candidates; invalid
candidates consume a slot. The validation-selected policy, including unchanged
fallback, is frozen and transferred unchanged to two further training seeds.
Two independent editor replicates are distinct from three training-seed scores.
There are also four predefined eight-query controls per objective; their search
spaces and adaptivity are not fully matched to the editors.

The text conditions reuse aligned title-only and title+description features from
the same pinned Meta-Llama-3-8B encoder. No retained item uses missing-description
fallback, and no additional GPU encoding is needed. Historical encoding cost is
recorded separately. Both training objectives use 16-dimensional biased matrix
factorization and full-catalogue ranking at 10.

```bash
python -m wsprec.complete_dataset
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m wsprec.neutral_study train
python -m wsprec.neutral_study export
python scripts/prepare_neutral_prompts.py
python -m wsprec.neutral_study controls
python -m wsprec.neutral_study references
# Run fresh editors using the exact saved prompt and each exported allowlist.
# Evaluate each round once and return only candidate IDs, scores and errors.
python scripts/process_neutral_submissions.py --round 1 TRAJECTORY
python scripts/process_neutral_submissions.py --round 2 TRAJECTORY
python scripts/audit_neutral_snapshots.py
python -m wsprec.neutral_study freeze
python -m wsprec.neutral_study final
python scripts/collect_neutral_results.py
python scripts/build_neutral_report.py
```

Final mean test NDCG@10 (three training seeds; editor rows also average two
independent editor replicates):

| Condition | MSE | BPR |
| --- | ---: | ---: |
| Original model | 0.05826 | 0.10568 |
| No text / weights visible | 0.06939 | 0.10153 |
| No text / weights hidden | 0.06613 | 0.10761 |
| Title / weights visible | 0.06858 | 0.10885 |
| Title / weights hidden | 0.06503 | 0.10842 |
| Title + description / weights visible | 0.06266 | 0.10752 |
| Title + description / weights hidden | 0.06502 | 0.10812 |

In this run, description-condition means did not exceed their corresponding
title-only means, and access to weights did not consistently beat blind editing.
The EASE contextual reference scored 0.12324. These are observations of the
frozen policies, not claims about internal motivations or general superiority.
One trajectory selected the unchanged model; no transfer violated the bounds.
All 227 valid editor/control patches replayed exactly before test opening.

[Chinese report](output/pdf/WSP4Rec_description_complete_neutral_report.pdf),
[complete results](reports/v3_neutral/summary.json),
[per-seed results](reports/v3_neutral/test_results.csv),
[paired-user intervals](reports/v3_neutral/paired_user_intervals.json),
[prompt audit](reports/v3_neutral/prompt_audit.json), and
[costs](reports/v3_neutral/costs.json) contain the evidence. The final run used
192 editor candidates (29 invalid) and 48 recorded generation turns. Including
the superseded intermediate run gives 384 editor candidates and 96 recorded
turns; one interrupted, unacknowledged dispatch attempt is listed separately.
Its inference execution/charge cannot be confirmed. Token counts and monetary
charges are unavailable, and compute timings do not cover every preparation,
inspection or coordination operation.

Dataset preparation, scoring and final reporting refuse to overwrite existing
outputs. `final` claims an exclusive test-open marker before reading labels.
The snapshot audit replays validation only and provides no editor feedback.
Raw ratings, descriptions, features, model checkpoints and private ID mappings
are not committed. Native subscription token counts and per-call monetary
charges remain unavailable (`null`), not zero.

## BP versus training-aware Astra (v4)

At the user's request, the main comparison now has exactly two systems: the
BP-trained BPR model, and Astra directly editing that same checkpoint after
access to every learned parameter, the full training set, and an explicit
user/item-ID-to-embedding-row mapping. No blind or external-text condition.
Astra may compute training predictions, residuals and statistics and perform
one-shot analytical updates; extra BP, gradients, iterative model training
and automated parameter searches are disallowed by the editor instructions.
No directional hypotheses or requested rationales are supplied. Validation
and test rows remain unavailable to editors.

Three fixed BP training seeds are reused; each gets two fresh independent
Astra runs with two rounds of four candidates. The original checkpoint remains
eligible by validation. All 48 candidates were valid and replayed exactly
before policy/input freezing and one final test reporting pass.

| System | Test NDCG@10 | Test Recall@10 |
| --- | ---: | ---: |
| BP-trained BPR | 0.10568 | 0.12156 |
| Astra with weights + training data | 0.12178 | 0.14309 |

All six editor outputs improved test NDCG over their respective BP checkpoints.
Every selected patch changed user embeddings, item embeddings and item biases.
These results concern a previously evaluated MovieLens cohort, not untouched
external data, and do not separate the individual contribution of weights and
training access. One-shot analytical data-dependent updates are allowed, so
this is not data-free editing. Instruction allowlists are not OS isolation.
Native tokens, charges and total editor numerical-analysis wall time remain
unavailable; the [cost record](reports/v4_train_aware/costs.json) records them
as null, with reused training cost identified separately.

[Protocol](neutral_protocol_v4.json), [results and paired-user intervals](reports/v4_train_aware/summary.json),
[per-run table](reports/v4_train_aware/test_results.csv), and
[prompts and replay evidence](reports/v4_train_aware/evidence/) preserve the record.
The offline single-file interactive page is checked without network access.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m wsprec.train_aware_study prepare
# Dispatch fresh Astra sessions using runs/v4_train_aware/prompts/KEY.txt.
python -m wsprec.train_aware_study evaluate --key KEY --round 1
# Send the saved score-only feedback and exact second-round dispatch.
python -m wsprec.train_aware_study evaluate --key KEY --round 2
python -m wsprec.train_aware_study freeze
python -m wsprec.train_aware_study final
python scripts/collect_train_aware_results.py
python scripts/build_train_aware_web.py
```
