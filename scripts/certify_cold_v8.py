"""Post-freeze independent universal checks; never supplies editor feedback."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wsprec.cold_certify import certify_written,fold_certificate
OUT=ROOT/'runs/v8_mit'
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
 frozen=json.loads((OUT/'frozen_manifest.json').read_text())
 assert (OUT/'test_opened.lock').exists(),'Only after frozen final reporting pass'
 for path,h in frozen['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
 dest=OUT/'formal_certificates.json';assert not dest.exists()
 results={}
 for key,a in frozen['cases'].items():
  state=load(ROOT/a['path'])
  if a['kind']=='write':result=certify_written(state,a['task'],a['permutation'])
  else:
   result=fold_certificate(load(ROOT/a['source']),state)
   result['policy_verification']=certify_written(state,a['task'])
  results[key]=result;print(key,result.get('status',result.get('recognized_fold')),flush=True)
 dest.write_text(json.dumps({'scope':'Post-freeze independent tests, no editor feedback or selection. Exact real arithmetic does not certify floating point execution. SAT refutes exact implementation, not necessarily good ranking. Unrecognized fold does not mean unsuccessful repair.','cases':results},indent=2)+'\n')
if __name__=='__main__':main()
