"""Load independent configuration snapshots without importing Isaac Gym."""
from pathlib import Path
from importlib import import_module
import copy
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    'a1': ('a1.a1_config', 'A1RoughCfg'),
    'go1': ('go1.go1_config', 'Go1RoughCfg'),
    'aliengo': ('aliengo.aliengo_config', 'AlienGoRoughCfg'),
    'mybot': ('mybot.mybot_config', 'MybotRoughCfg'),
    'mybot_v2': ('mybot_v2.mybot_v2_config', 'MybotV2RoughCfg'),
    'mybot_v2_1': ('mybot_v2_1.mybot_v2_1_config', 'MybotV21RoughCfg'),
    'mybot_v3': ('mybot_v3.mybot_v3_config', 'MybotV3RoughCfg'),
    'mybot_v3_with_arm': ('mybot_v3_with_arm.mybot_v3_with_arm_config', 'MybotV3WithArmRoughCfg'),
    'mybot_v3_bridge_a': ('mybot_v3.mybot_v3_bridge_a_config', 'MybotV3BridgeACfg'),
    'mybot_v3_bridge_b': ('mybot_v3.mybot_v3_bridge_b_config', 'MybotV3BridgeBCfg'),
}

def class_to_dict(obj):
    if isinstance(obj, dict): return {k: class_to_dict(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [class_to_dict(v) for v in obj]
    if not hasattr(obj, '__dict__'): return obj
    return {k: class_to_dict(getattr(obj,k)) for k in dir(obj)
            if not k.startswith('_') and not callable(getattr(obj,k))}

def load_config(task='mybot_v3', num_envs=1024, stage=3):
    module, name = TASKS[task]
    mod = import_module('himloco_lab.legacy.'+module)
    cfg = copy.deepcopy(getattr(mod,name)())
    train = class_to_dict(getattr(mod,name+'PPO')())
    cfg.env.num_envs = num_envs
    if stage < 4:
        cfg.terrain.mesh_type = 'plane'
        cfg.terrain.selected = None
        cfg.terrain.curriculum = False
    if stage == 1:
        cfg.noise.add_noise = False
        cfg.noise.noise_level = 0.
        for k,v in vars(cfg.domain_rand).items():
            if isinstance(v,bool): setattr(cfg.domain_rand,k,False)
        # Defaults live on the class, so also disable inherited flags.
        for k in dir(cfg.domain_rand):
            if not k.startswith('_') and isinstance(getattr(cfg.domain_rand,k),bool): setattr(cfg.domain_rand,k,False)
        cfg.commands.curriculum = False
    return cfg, train

def asset_contract(cfg):
    path = Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(ROOT/'legged_gym')))
    if path.suffix == '.xml':
        return mjcf_contract(path,cfg)
    tree = ET.parse(path)
    joints = [j for j in tree.getroot().findall('joint') if j.attrib['type'] != 'fixed']
    # Gym loads articulations depth-first; this repository's URDF sibling order defines the policy order.
    children = {}
    child_links = set()
    for j in tree.getroot().findall('joint'):
        children.setdefault(j.find('parent').attrib['link'],[]).append(j)
        child_links.add(j.find('child').attrib['link'])
    roots = [l.attrib['name'] for l in tree.getroot().findall('link') if l.attrib['name'] not in child_links]
    ordered=[]
    def walk(link):
        for j in children.get(link,[]):
            if j.attrib['type']!='fixed': ordered.append(j)
            walk(j.find('child').attrib['link'])
    walk(roots[0])
    names=[j.attrib['name'] for j in ordered]
    if len(names)!=cfg.env.num_actions or set(names)!=set(cfg.init_state.default_joint_angles):
        raise ValueError(f'URDF/algorithm joint mismatch: {names}; actions={cfg.env.num_actions}')
    limits={j.attrib['name']: {k:float(v) for k,v in j.find('limit').attrib.items()} for j in ordered}
    meshes=[]
    for m in tree.iter('mesh'):
        name=m.attrib['filename']
        if name.startswith('package://'):
            name=name.split('/',3)[-1]; candidate=path.parent.parent/name
        else: candidate=path.parent/name
        if not candidate.exists(): raise FileNotFoundError(candidate)
        meshes.append(candidate.resolve())
    return path, names, limits, roots[0]


def mjcf_contract(path,cfg):
    tree=ET.parse(path); root=tree.getroot(); body=root.find('worldbody/body')
    joints=[j for j in body.iter('joint') if j.get('type')!='free']
    names=[j.attrib['name'] for j in joints]
    if len(names)!=cfg.env.num_actions or set(names)!=set(cfg.init_state.default_joint_angles):
        raise ValueError('MJCF joint/observation contract mismatch')
    defaults=root.find('default/motor').attrib
    motors={m.attrib['joint']:m for m in root.findall('actuator/motor')}
    limits={}
    for joint in joints:
        name=joint.attrib['name']; motor=motors[name]
        force=[float(v) for v in motor.get('ctrlrange',defaults['ctrlrange']).split()]
        lower,upper=map(float,joint.attrib['range'].split())
        # MJCF does not declare a velocity cap. Read the imported cap after creation.
        limits[name]={'lower':lower,'upper':upper,'effort':max(abs(v) for v in force),'velocity':None}
    compiler=root.find('compiler'); meshdir=path.parent/compiler.get('meshdir','.')
    for m in root.findall('asset/mesh'):
        if not (meshdir/m.attrib['file']).exists():raise FileNotFoundError(meshdir/m.attrib['file'])
    return path,names,limits,body.attrib['name']


def prepare_mjcf(path):
    """Isolate robot-only MJCF; the Lab scene owns lights and ground."""
    tree=ET.parse(path); root=tree.getroot(); world=root.find('worldbody')
    for child in list(world):
        if child.tag!='body':world.remove(child)
    compiler=root.find('compiler')
    meshdir=(path.parent/compiler.get('meshdir','.')).resolve()
    out=ROOT/'isaaclab_migration/.cache'/path.stem/'robot.xml'
    out.parent.mkdir(parents=True,exist_ok=True)
    link=out.parent/'meshes'
    if not link.exists():link.symlink_to(meshdir,target_is_directory=True)
    compiler.set('meshdir','meshes')
    tree.write(out,encoding='utf-8',xml_declaration=True)
    return out
