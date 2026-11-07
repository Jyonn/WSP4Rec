"""Run matched BP training conditions on independent A100 devices."""
import json,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/v7_text';OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'training_jobs.json').exists():raise FileExistsError('training launch already recorded')
started=time.time()
hardware={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
          'gpu_devices':[{'index':i,'name':torch.cuda.get_device_name(i),'total_memory':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],
          'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
          'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
(OUT/'server_environment.json').write_text(json.dumps(hardware,indent=2)+'\n')
processes=[]
for condition,gpu in (('none',0),('text',1),('shuffled_text',2)):
 env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 log=(OUT/f'{condition}-gpu.log').open('w')
 cmd=[sys.executable,'-m','wsprec.text_model','--command','train','--condition',condition,'--device','cuda']
 p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 processes.append((condition,gpu,p,log))
print('Launched',len(processes),'conditions',flush=True)
records=[]
for condition,gpu,p,log in processes:
 code=p.wait();log.close();records.append({'condition':condition,'physical_gpu_index':gpu,'returncode':code})
(OUT/'training_jobs.json').write_text(json.dumps({'started_at':started,'finished_at':time.time(),'wall_seconds':time.time()-started,'jobs':records},indent=2)+'\n')
print(json.dumps(records),flush=True)
if any(r['returncode'] for r in records):raise SystemExit(1)
