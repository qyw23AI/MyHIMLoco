# Mybot_v3 HIMLoco training config mapping

## Status summary
- Current config defines 4096 envs and 5000 iterations; runtime depends on hardware, so 1 hour completion cannot be confirmed from config alone.
- Paper training settings are not present in repo docs; please provide the exact paper or a table/section to cross-check.

## How to train mybot_v3 (HIMLoco)
From repo root:

```bash
python legged_gym/legged_gym/scripts/train.py --task mybot_v3 --headless --rl_device cuda:0 --num_envs 4096 --max_iterations 5000
```

## Resume training from model_700.pt
From repo root:

```bash
python legged_gym/legged_gym/scripts/train.py --task mybot_v3 --resume --load_run May10_12-18-51_ --checkpoint 700
```

Notes:
- `--num_envs` and `--max_iterations` override defaults from config.
- The runner is HIMOnPolicyRunner and algorithm is HIMPPO (HIMLoco PPO variant).

## How to play the trained policy
From repo root:

```bash
python legged_gym/legged_gym/scripts/play.py --task mybot_v3 --load_run May04_18-17-50_ --checkpoint 1000
```

Notes:
- `--load_run` is the run folder name under `legged_gym/logs/rough_mybot_v3/`.
- `--checkpoint -1` loads the latest checkpoint in that run.

## Runtime estimate (cannot be verified without hardware)
Current total env-steps:

$$
4096 \times 100 \times 5000 = 2.048\times10^9
$$

Time to finish depends on steps/sec of your GPU. To fit ~1 hour, you can set:

$$
\text{max\_iterations} \approx \frac{3600 \times \text{steps/sec}}{4096 \times 100}
$$

## Config mapping (code locations)

### Base env config
- Base config: [legged_gym/legged_gym/envs/base/legged_robot_config.py](legged_gym/legged_gym/envs/base/legged_robot_config.py)
- Key fields used by mybot_v3 unless overridden:
  - `env.num_envs = 4096`
  - `env.episode_length_s = 20`
  - `terrain.mesh_type = 'trimesh'`
  - `control.decimation = 4`
  - `rewards` defaults (many overridden in mybot_v3)

### Mybot_v3 overrides
- Mybot v3 config: [legged_gym/legged_gym/envs/mybot_v3/mybot_v3_config.py](legged_gym/legged_gym/envs/mybot_v3/mybot_v3_config.py)
- Overrides include:
  - `init_state.pos = [0.0, 0.0, 0.43]`
  - `control.action_scale = 0.2` and per-joint PD gains
  - `commands.ranges.lin_vel_x/lin_vel_y = [-0.2, 0.2]`
  - `asset.file = .../mybot_v3.urdf`, `self_collisions = 1`
  - `rewards.scales.*` customized for mybot_v3
  - `runner.experiment_name = 'rough_mybot_v3'`

### Training config (PPO)
- Base PPO config: [legged_gym/legged_gym/envs/base/legged_robot_config.py](legged_gym/legged_gym/envs/base/legged_robot_config.py)
- Defaults used by mybot_v3 unless overridden:
  - `algorithm.learning_rate = 1e-3`
  - `algorithm.num_learning_epochs = 5`
  - `algorithm.num_mini_batches = 4`
  - `runner.num_steps_per_env = 100`
  - `runner.max_iterations = 5000`

### Task registration
- Task mapping: [legged_gym/legged_gym/envs/__init__.py](legged_gym/legged_gym/envs/__init__.py)
- Task name: `mybot_v3`

## Paper alignment (needs source)
Please provide the paper config (table or section). I will add a point-by-point comparison here once I have it.
