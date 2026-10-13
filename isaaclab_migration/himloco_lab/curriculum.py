"""Observe actual terrain/reset transitions without changing curriculum rules."""
import json
from pathlib import Path
import torch


class CurriculumMonitor:
    def __init__(self, env, directory, every_steps=100):
        self.env=env;self.core=env.core
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.every_steps=every_steps;self.steps=0
        self.resets=0;self.timeouts=0;self.increases=0;self.decreases=0
        self.origin_changes=0;self.command_range_changes=0
        self.history=[]
        self.last_command=list(self.core.command_ranges['lin_vel_x'])
        self.initial=self.snapshot()
        self.history.append(self.initial)

    def before_step(self):
        self.levels=self.core.terrain_levels.clone()
        self.origins=self.core.env_origins.clone()

    def after_step(self, terminated, truncated):
        self.steps+=1
        self.resets+=int((terminated|truncated).sum())
        self.timeouts+=int(truncated.sum())
        delta=self.core.terrain_levels-self.levels
        self.increases+=int((delta>0).sum())
        self.decreases+=int((delta<0).sum())
        self.origin_changes+=int((self.core.env_origins!=self.origins).any(-1).sum())
        command=list(self.core.command_ranges['lin_vel_x'])
        self.command_range_changes+=int(command!=self.last_command)
        self.last_command=command
        if self.steps%self.every_steps==0:
            self.history.append(self.snapshot())

    def snapshot(self):
        c=self.core
        levels=c.terrain_levels
        return {'policy_step':self.steps,'mean_level':float(levels.float().mean()),
                'min_level':int(levels.min()),'max_level':int(levels.max()),
                'level_histogram':torch.bincount(levels,minlength=c.cfg.terrain.num_rows).tolist(),
                'total_resets':self.resets,'timeouts':self.timeouts,
                'level_increases':self.increases,'level_decreases':self.decreases,
                'origin_changes':self.origin_changes,
                'command_range':list(c.command_ranges['lin_vel_x']),
                'command_range_changes':self.command_range_changes}

    def save(self):
        c=self.core;cfg=c.cfg.terrain
        final=self.snapshot()
        heights=self.env.terrain.height_field_raw
        report={'mesh_type':cfg.mesh_type,'terrain_curriculum':cfg.curriculum,
                'command_curriculum':c.cfg.commands.curriculum,
                'terrain_proportions':cfg.terrain_proportions,
                'rows':cfg.num_rows,'columns':cfg.num_cols,
                'terrain_length_m':cfg.terrain_length,'terrain_width_m':cfg.terrain_width,
                'height_range_m':[float(heights.min())*cfg.vertical_scale,
                                  float(heights.max())*cfg.vertical_scale],
                'max_observed_level':max(row['max_level'] for row in self.history+[final]),
                'transition_count_definition':'Observed level value increases/decreases; highest-level random recycling can also decrease the value',
                'initial':self.initial,'final':final,
                'history':self.history+[final]}
        if cfg.mesh_type=='plane' or report['height_range_m'][0]==report['height_range_m'][1]:
            raise RuntimeError('Terrain curriculum test requires nonflat terrain')
        if not cfg.curriculum:raise RuntimeError('Terrain curriculum is disabled')
        if final['min_level']<0 or final['max_level']>=cfg.num_rows:
            raise RuntimeError('Terrain level outside configured bounds')
        target=self.directory/'curriculum.json'
        target.write_text(json.dumps(report,indent=2))
        print('CURRICULUM_RECORDED',target.resolve(),
              json.dumps({k:v for k,v in final.items() if k!='level_histogram'}),flush=True)
        return report
