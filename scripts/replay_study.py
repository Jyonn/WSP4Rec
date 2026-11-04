"""Replay archived editor proposals on newly trained baseline checkpoints.

Never regenerates an LLM response, and never overwrites a frozen study.
"""
import json
import sys
from pathlib import Path

# Permit running this script from the repository without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from wsprec.study import ROOT, OUT, save, controls, evaluate_candidates
from wsprec.references import main as references
from wsprec.finalize_study import freeze, final


def main():
    if (OUT / 'frozen_manifest.json').exists() or (OUT / 'final_results.json').exists():
        raise FileExistsError('Frozen study exists; preserve it and use a separate clean checkout for reproduction')
    if not (OUT / 'text/title_embeddings.npy').exists():
        raise FileNotFoundError('Extract Qwen title embeddings first; see README')
    torch.set_num_threads(1)
    archive = json.loads((ROOT / 'reports/evidence/editor_trajectories.json').read_text())
    controls()
    references()
    for trajectory, artifacts in archive.items():
        objective, condition = trajectory.split('-', 1)
        for round_id in (1, 2):
            name = f'round{round_id}.json'
            path = OUT / 'editors' / trajectory / name
            save(path, artifacts[name])
            evaluate_candidates(path, objective, condition.endswith('text'))
    freeze()
    final()


if __name__ == '__main__':
    main()
