"""Run one prompt through the configured tool-free Astra account CLI."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wsprec.mit_wsp_harness import run_turn
p=argparse.ArgumentParser();p.add_argument('--prompt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--effort',default='xhigh');p.add_argument('--cli');p.add_argument('--timeout',type=int,default=3500);a=p.parse_args()
kwargs={'effort':a.effort,'timeout':a.timeout}
if a.cli:kwargs['cli']=a.cli
text,record=run_turn(a.prompt.read_text(),a.output,**kwargs)
print(json.dumps(record,indent=2))
raise SystemExit(0 if record['successful'] else 1)
