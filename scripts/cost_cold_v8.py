"""Summarize exposed counters; native subscription billing is unknown."""
import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/v8_mit';REPORT=ROOT/'reports/v8_mit'
def load(p):return json.loads(p.read_text())
rows=[]
for p in sorted(OUT.glob('cases/*/turn*/run.json')):
 d=load(p);rows.append({'case':p.parent.parent.name,'turn':p.parent.name,'model':d['model_requested'],'effort':d['effort_requested'],'wall_seconds':d['wall_seconds'],'successful':d['successful'],'tool_free':d['audit']['tool_free'],'usage':d['audit'].get('usage'),'record':str(p.relative_to(ROOT))})
fields=('input_tokens','cached_input_tokens','cache_write_input_tokens','output_tokens','reasoning_output_tokens')
training=[]
for p in sorted(OUT.glob('bp/task*/training_summary.json'))+[OUT/'real_bridge/training_summary.json']:
 for d in load(p)['trials']:training.append({k:d.get(k) for k in ('task','seed','condition','train_pairs','wall_seconds','epochs_run','best_epoch','checkpoint')})
result={'recorded_at':time.time(),'research_requests':len(rows),'research_contexts':len({r['case'] for r in rows}),'research_token_counters':{f:sum((r['usage']or{}).get(f,0) for r in rows) for f in fields},'request_wall_seconds_sum':sum(r['wall_seconds'] for r in rows),'research_calls':rows,'training_trials':training,'training_wall_seconds_sum':sum(t['wall_seconds'] for t in training),'training_parallel_jobs':load(OUT/'training_jobs.json'),'billing_usd':None,'billing_explanation':'Native Codex subscription did not expose an invoice. Cached input is a subset; do not add to input. Reasoning counter relationship to output depends on CLI schema: reported separately, not summed as billed tokens. Root goal counter has different scope and may overlap, never sum as verified billing. No API key used.','other_native_calls':'Infrastructure READY smoke recorded separately under cli_smoke_latest; source/design/code/review agents are outside the 36 planned research requests and included only where enclosing goal counter exposes scope.','administrative_recovery':'Some completed replies contained temporary routing/reconnection notices. Original event streams preserved; audit correction reuses identical replies, no resampling. Recovery details separate if present.'}
(REPORT/'costs.json').write_text(json.dumps(result,indent=2)+'\n');print(result['research_requests'],result['research_token_counters'],len(training))
