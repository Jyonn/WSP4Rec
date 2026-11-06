"""Save exact neutral instructions before native editor dispatch."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsprec.neutral_study import ROOT, OUT


def main():
    assignments = json.loads((OUT / 'editor_assignments.json').read_text())
    folder = OUT / 'prompts'
    folder.mkdir(exist_ok=True)
    for key, assignment in assignments.items():
        text = ((ROOT / 'editor_prompt_v2.txt').read_text()
                + '\nAllowlist:\n' + '\n'.join(assignment['allowlist'])
                + '\n\nAssigned output directory: ' + assignment['output_directory']
                + '\nRound-one submission: ' + assignment['output_directory'] + '/round1.json\n')
        dispatch = (f'Follow the exact instructions in {OUT.relative_to(ROOT).as_posix()}/prompts/{key}.txt. '
                    'Read only that instruction file and its allowlisted inputs, plus your own prior submitted files. '
                    'Reply only with the submitted file path.')
        for name, value in ((key + '.txt', text), (key + '.dispatch.txt', dispatch)):
            path = folder / name
            if path.exists():
                if path.read_text() != value:
                    raise ValueError(f'Refusing to alter an existing prompt: {path}')
            else:
                path.write_text(value)
    print(f'{len(assignments)} prompt pairs saved or verified')


if __name__ == '__main__':
    main()
