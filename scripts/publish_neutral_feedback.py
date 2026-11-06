"""Publish score-only feedback and exact second-round instructions."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/os.environ.get('WSPREC_RUN_DIR','runs/v3_neutral')


def main():
    assignments=json.loads((OUT/'editor_assignments.json').read_text())
    (OUT/'feedback').mkdir(exist_ok=True)
    for key,a in assignments.items():
        source=OUT/'evaluations'/key/'round1.json'
        if not source.exists():continue
        feedback=[{k:r[k] for k in ('id','valid_ndcg@10','error')} for r in json.loads(source.read_text())['results']]
        (OUT/'feedback'/f'{key}-round1.json').write_text(json.dumps(feedback,indent=2))
        message=(f'Read {OUT.relative_to(ROOT).as_posix()}/feedback/{key}-round1.json in addition to your original allowlist. '
            f'Submit exactly four final patches to {a["output_directory"]}/round2.json under the same instructions. '
            'Do not read other files, propose hypotheses or give explanations. Reply only with the submission file path.')
        (OUT/'prompts'/f'{key}-round2.dispatch.txt').write_text(message)


if __name__=='__main__':main()
