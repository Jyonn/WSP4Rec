# WSP4Rec: semantic weight programming study (v8)

This branch contains experiment code and the locked protocol. The manuscript,
local research report, raw MovieLens data and native account credentials are not
published here. Official source audit pins MIT WSP release commit
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` in
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs).

## Completed findings

- 12 fresh Astra zero-tool, zero-forward numeric programs (3 policies x 2
  semantic-coordinate encodings x 2 replies) implement a supplied ranking rule.
  All have synthetic NDCG@10=1 and independent exact-real error<=1e-6 proofs.
- 12 existing-model edits also implement the rule, but change all 1,825 weights.
  This is replacement, not evidence of localized neural repair.
- The exact classical compiler ties Astra. Task1 both-cold mean BP=.77219,
  algebraic status fold=.99429, cold-aware BP=.99403, Astra/compiler=1.
- All four frozen task0 programs equal the compiler on a real MovieLens bridge:
  K0/5/10 NDCG=.05505/.07855/.08335; three-seed content BP averages
  .11641/.11605/.11532. Real-data utility superiority is not established.

Synthetic identities are newly generated but explicit profiles are supplied;
this does not recover unknown private tastes or exclude pretrained concepts.
Real film pretraining membership is unknown. Status sensitivity is a designed
stress case, not a claim about naturally prevalent recommender defects. The
feedback implementation mixes warm-validation and separate-practice examples
(32+32 initially;64+64 per additional batch), despite protocol shorthand calling
these practice pairs. No final entities enter the editor.

## Components and reproduction

Install this project, optional `verify` (Z3) and `report` dependencies. Run
`python -m unittest discover -s tests -q` for the independent toy checks.
The synthetic generator and BP trainer are `wsprec.cold_benchmark`; the real
bridge is `wsprec.cold_real_bridge`, which requires the previously prepared
1568-film description cohort and original official MovieLens genre files.
The synthetic primary task itself does not consume descriptions or Llama vectors.

`wsprec.cold_study` exposes `prepare_writes`, `run_writes`, `prepare_repairs`,
`run_repairs`, `freeze`, `final`. Follow the locked protocol order: prepare
protocol/data and train baselines, collect ALL native replies before freeze,
then run the one final reporting pass. `scripts/certify_cold_v8.py` adds
post-freeze universal policy checks; `scripts/summarize_cold_v8.py` builds CSVs
and conditional paired-user intervals. Existing artifacts are never silently
reused or overwritten. Training outputs are intentionally excluded from Git.

The recorded native client is official `@openai/codex` 0.167.0, isolated under
`/private/tmp/wsp-v8-cli`; adjust the `CLI` constant to your installed compatible
client on other systems. It uses the existing authorized native account and no
API-key forwarding. `mit_wsp_harness` disables tools, audits event streams and
accepts numeric tensor assignments only. Restricting arbitrary execution is
stronger than merely instructing an editor not to fit data. The global user
configuration is ignored for each ephemeral call. External practice forwards
are separately budgeted and their full visible transcript is replayed.

All 36 research requests and all 33 BP trials are retained locally. Per-request
input/cached/output counters are exposed, monetary subscription billing is not.
The administrative recovery script preserves raw replies when reconnect notices
are followed by a completed turn; it never retries scientific content.

Historical text experiments remain available on branch `codex/text-v7`.
