#!/usr/bin/env python3
"""Inventory local imports, every explicit Gym call and imported tensor helper call."""
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]

GYM_HELPERS={'quat_apply','quat_rotate_inverse','quat_rotate','quat_mul','quat_conjugate',
             'get_axis_params','to_torch','torch_rand_float','normalize','get_euler_xyz',
             'tf_apply','tf_inverse','quat_from_euler_xyz'}
entries=[]
for directory in ['legged_gym/legged_gym','rsl_rl/rsl_rl']:
    for path in sorted((ROOT/directory).rglob('*.py')):
        tree=ast.parse(path.read_text());lines=path.read_text().splitlines()
        imports=[];helpers=set();wildcard=False
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom) and (n.module or '').startswith('isaacgym'):
                imports.append(n)
                if n.module=='isaacgym.torch_utils':
                    helpers.update(x.asname or x.name for x in n.names)
                    wildcard|=any(x.name=='*' for x in n.names)
            elif isinstance(n,ast.Import) and any(x.name.startswith('isaacgym') for x in n.names):imports.append(n)
        for n in imports:entries.append((str(path.relative_to(ROOT)),n.lineno,'import',lines[n.lineno-1].strip()))
        for n in ast.walk(tree):
            if not isinstance(n,ast.Call):continue
            func=ast.unparse(n.func)
            if func.startswith(('self.gym.','gym.','gymapi.','gymtorch.','gymutil.','terrain_utils.')):
                entries.append((str(path.relative_to(ROOT)),n.lineno,'simulation/API',lines[n.lineno-1].strip()))
            elif func in helpers or (wildcard and func in GYM_HELPERS):
                entries.append((str(path.relative_to(ROOT)),n.lineno,'tensor math',lines[n.lineno-1].strip()))
out=ROOT/'isaaclab_migration/docs/isaacgym_api_inventory.txt'
out.write_text('\n'.join(f'{p}:{line} [{kind}] {text}' for p,line,kind,text in sorted(set(entries)))+'\n')
changed=[]
manifest=json.loads((ROOT/'isaaclab_migration/docs/source_manifest.json').read_text())
for name,digest in manifest.items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:changed.append(name)
print(json.dumps({'gym_api_locations':len(set(entries)),'original_source_files_checked':len(manifest),'changed_original_sources':changed},indent=2))
if changed:sys.exit(1)
