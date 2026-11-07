"""Parallel laboratory baseline training; no held-out cold scoring."""
import json,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/v8_mit';OUT.mkdir(exist_ok=True,parents=True)
if (OUT/'training_jobs.json').exists():raise FileExistsError('already trained')
started=time.time();jobs=[]
envinfo={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
(OUT/'server_environment.json').write_text(json.dumps(envinfo,indent=2)+'\n')
commands=[(f'task{t}',t,[sys.executable,'-m','wsprec.cold_benchmark','train','--task',str(t),'--device','cuda']) for t in (0,1,2)]
commands.append(('real_bridge',3,[sys.executable,'-m','wsprec.cold_real_bridge','train','--device','cuda']))
for name,gpu,command in commands:
 env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 log=(OUT/f'{name}-gpu.log').open('w');p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((name,gpu,p,log))
print('Launched',len(jobs),'baseline jobs',flush=True);records=[]
for name,gpu,p,log in jobs:
 status=p.wait();log.close();records.append({'job':name,'gpu':gpu,'returncode':status})
(OUT/'training_jobs.json').write_text(json.dumps({'jobs':records,'wall_seconds':time.time()-started,'started_at':started,'finished_at':time.time()},indent=2)+'\n')
print(json.dumps(records),flush=True)
if any(x['returncode'] for x in records):raise SystemExit(1)
