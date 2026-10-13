#!/usr/bin/env python3
"""Read-only environment/asset audit; never installs or mutates driver/CUDA/torch."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'isaaclab_migration'))
from himloco_lab.config import TASKS,load_config,asset_contract
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args()
report={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{},'assets':{}}
for name in ['isaacsim','isaaclab','torch','numpy','warp-lang','tensorboard']:
    try:report['packages'][name]=metadata.version(name)
    except metadata.PackageNotFoundError:report['packages'][name]=None
lab=ROOT/'IsaacGymEnvs/IsaacLab'
report['isaaclab_repository_version']=(lab/'VERSION').read_text().strip()
report['isaaclab_repository_commit']=subprocess.check_output(['git','-C',str(lab),'rev-parse','HEAD'],text=True).strip()
try:
    import torch
    report['cuda_available']=torch.cuda.is_available()
    if torch.cuda.is_available():
        report['gpu']=torch.cuda.get_device_name();report['capability']=torch.cuda.get_device_capability()
        # Actual kernel execution verifies Blackwell compatibility, not just device enumeration.
        x=torch.randn(128,128,device='cuda');float((x@x).sum());torch.cuda.synchronize()
        report['cuda_kernel_test']='passed'
except Exception as e:report['torch_error']=repr(e)
try:report['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True,stderr=subprocess.STDOUT,timeout=10)
except (OSError,subprocess.SubprocessError) as e:report['nvidia_smi_error']=str(e)
for name in TASKS:
    c,tr=load_config(name,1024,4)
    try:
        path,joints,limits,root=asset_contract(c)
        report['assets'][name]={'path':str(path),'root':root,'joint_order':joints,'limits':limits,
            'actor_observations':c.env.num_observations,'critic_observations':c.env.num_privileged_obs,
            'control_dt':c.sim.dt*c.control.decimation,'action_scale':c.control.action_scale,
            'rollout_steps':tr['runner']['num_steps_per_env']}
    except Exception as e:report['assets'][name]={'error':repr(e)}
text=json.dumps(report,indent=2)
if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
print(text)
if not report.get('cuda_available') or any('error' in v for v in report['assets'].values()):sys.exit(1)
