"""Numeric replay evidence; excludes raw ratings and movie title/description text."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/v7_text';R=ROOT/'reports/v7_text'
files=[]
for directory in ('editors','evaluations','feedback','prompts'):
 files.extend(p for p in (OUT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.npy','.json','.txt'))
for p in (OUT/'inputs').rglob('*'):
 if p.is_file() and p.name not in ('train.npy','movies.json','llama_features.npy','features.npy'):files.append(p)
for condition in ('none','text','shuffled_text'):
 files.extend(p for p in (OUT/condition).rglob('*') if p.is_file() and p.suffix in ('.json','.pt') and 'spectral_cache' not in p.parts)
for directory in ('controls','calibration'):
 files.extend(p for p in (OUT/directory).rglob('*.json'))
files.extend(p for p in (OUT/'text').iterdir() if p.is_file() and p.suffix in ('.json','.npz','.npy'))
files.extend(p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.log'))
files.extend(ROOT/'data/ml-100k-description-complete-v3'/n for n in ('title_description.npy','mapping.json','manifest.json'))
for name in ('wsprec/text_model.py','wsprec/text_study.py','wsprec/text_calibration.py','wsprec/study.py','wsprec/study_metrics.py','wsprec/tensor_edits.py','wsprec/cli.py','wsprec/paper_source.py','text_study_protocol_v7.json','reports/followup/llama3_embedding_metadata.json'):
 files.append(ROOT/name)
for directory in ('scripts','tests'):
 files.extend((ROOT/directory).glob('*text*.py'))
files.extend(p for p in R.iterdir() if p.is_file() and p.suffix in ('.json','.csv'))
files=sorted(set(files));hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
readme='''WSP4Rec v7 text integration evidence

Includes all 18 BP training trials/checkpoints, 18 anonymous editor weight inputs,
144 candidate patches/assets and immutable evaluation snapshots, 36 score-only
feedback files, portable programs, full per-user final results, source code,
protocol/amendments, original cached 1568x4096 numeric Llama features, PCA and
64D true/shuffled features, cohort mapping, and independent review. Numeric
features are derived encoder outputs, not the Llama model weights.

Raw MovieLens ratings and raw movie titles/descriptions are excluded. This is a
numerical evidence supplement, not a complete rerun of new LLM discoveries.
Acquire MovieLens-100K under the GroupLens terms. Rebuild the historical split
using the included mapping; never resplit or reselect against test results:
python -m wsprec.paper_source --raw /path/to/u.data --output /tmp/v7-rebuilt --mapping data/ml-100k-description-complete-v3/mapping.json
Use the reconstructed train.npy, valid.npy and splits.npz with the included
original manifest. The earlier v5 package also supplies exact reconstruction.

In an existing project checkout with the original data, use:
python scripts/audit_text_inputs.py
python scripts/audit_text_portable.py case01  # repeat through case18
The latter checks saved deltas without test scoring or another LLM call.
Portable build takes features=None for no-text, or a dictionary with projected
(X64) and llama (E4096). Common feature copies replace omitted duplicate per-case
features.npy/llama_features.npy. Raw text must be separately restored to repeat
the full input audit or discover new full-text responses. The archived numerical
transformations do not explicitly use raw descriptions.

The archive includes all candidate arrays. Individual generated control model.pt
files and SVD caches are omitted and reconstructible from the fixed generators.
The global original manifest refers to omitted paths too; SHA256SUMS.json verifies
this archive's actual members. Training replay need not be bitwise identical on
another hardware/backend; the included source checkpoints fix numerical replay.

Primary fixed SVD+BP includes validation fallback; raw forced fusion is reported
separately. Eight-candidate scalar calibration was added during validation-stage
inspection and is explicitly posthoc. Outer JSON arrays and {patches:[...]} are
equivalent accepted containers; original submission bytes are preserved.
The architecture U dot (V+XW) does not expand MF's formal function class because
V is free. Native editors used logical file allowlists, not OS isolation.
Per-request tokens and billed dollars were unavailable; the goal-tool cumulative
counter is separate from account invoices.
'''
path=R/'editor_artifacts.zip'
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,str(p.relative_to(ROOT)))
 z.writestr('SHA256SUMS.json',json.dumps(hashes,indent=2))
 z.writestr('README.txt',readme)
print(len(files),'files',path.stat().st_size,'bytes',hashlib.sha256(path.read_bytes()).hexdigest())
