"""Check original submitted arrays from portable programs, no held-out scoring."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wsprec.text_study import OUT,ROOT,load,verify_portable
from wsprec.study import checkpoint
key=sys.argv[1];a=load(OUT/'assignments.json')[key]
records=[r for rnd in (1,2) for r in load(OUT/'evaluations'/key/f'round{rnd}.json')['results']]
print(key,verify_portable(key,a,checkpoint(ROOT/a['model'])['state'],records))
