# issues.md

本文记录本次环境和训练过程中遇到的问题与解决方法。

## 1. PyTorch 版本无法通过 pip 获取

**现象**

- 执行 `pip3 install torch==1.10.0+cu113 ...` 报错：找不到对应版本。

**原因**

- PyPI 上没有 `+cu113` 这一构建标签；且当时使用了系统 `pip3`，未安装到 `himloco` 环境内。

**解决**

- 使用 `conda` 安装旧版本 CUDA 11.3 组合：
  - `conda install -n himloco pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge`

---

## 2. requirements.txt 中 torch +cu113 触发 pip 依赖失败

**现象**

- `python -m pip install -r requirements.txt` 报 `No matching distribution found for torch==1.10.0+cu113`。

**原因**

- 依旧是 PyPI 不提供 `+cu113` 版本。

**解决**

- 先用 conda 安装 torch/cuda 组合，再用 pip 安装其余依赖：
  - `python -m pip install matplotlib==3.5.3 ninja==1.11.1.1 tensorboard==2.11.2 tensorboard-data-server==0.6.1 tensorboard-plugin-wit==1.8.1`

---

## 3. rsl_rl 安装报 Python 版本不满足

**现象**

- `pip install -e rsl_rl` 报 `Package 'rsl-rl' requires a different Python: 3.7.12 not in '>=3.7.16'`。

**原因**

- 环境 Python 版本为 3.7.12，低于 `>=3.7.16` 要求。

**解决**

- 升级环境 Python：
  - `conda install -y python=3.7.16`

---

## 4. 训练启动时报 distutils 相关错误

**现象**

- 运行 `train.py` 报错：`AttributeError: module 'distutils' has no attribute 'version'`。

**原因**

- `torch.utils.tensorboard` 需要 `distutils.version`，但当前导入路径下该属性不可用。

**解决**

- 在 rsl_rl 的 runner 中预先导入 `distutils.version`：
  - `rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - `rsl_rl/rsl_rl/runners/him_on_policy_runner.py`

---

## 5. CUDA 初始化 cuBLAS 失败

**现象**

- 训练阶段报错：`CUBLAS_STATUS_NOT_INITIALIZED`。

**原因**

- 运行库冲突或初始化时机问题（CUDA 11.3 与系统环境混用）。

**解决**

- 强制优先使用 torch 自带 CUDA 库，并启用懒加载：
  - `export TORCH_LIB="$CONDA_PREFIX/lib/python3.7/site-packages/torch/lib"`
  - `export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`
  - `export CUDA_MODULE_LOADING=LAZY`

---

## 6. 训练时 GPU OOM / illegal memory access

**现象**

- Isaac Gym 报 `out of memory`，随后 `illegal memory access`。

**原因**

- 默认 `num_envs=1024` 对 6GB 显存压力过大。

**解决**

- 降低环境数量：
  - 命令行：`python train.py --headless --num_envs 128`
  - 或永久修改配置 `legged_robot_config.py` 中 `num_envs`

---

## 7. 动态链接库找不到 (LD_LIBRARY_PATH)

**现象**

- 运行时提示动态链接库找不到，需要临时设置 `LD_LIBRARY_PATH`。

**临时解决**

- `export LD_LIBRARY_PATH=~/miniconda3/envs/himloco/lib:$LD_LIBRARY_PATH`

**长期解决 (Conda 自动加载)**

1) 进入环境目录并创建脚本目录：

- `cd ~/miniconda3/envs/himloco`
- `mkdir -p ./etc/conda/activate.d`
- `mkdir -p ./etc/conda/deactivate.d`

2) 创建激活脚本 `./etc/conda/activate.d/env_vars.sh`：

```
#!/bin/sh
# 备份并设置路径
export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

3) 创建退出脚本 `./etc/conda/deactivate.d/env_vars.sh`：

```

#!/bin/sh
# 恢复原始状态
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
unset OLD_LD_LIBRARY_PATH
```


## 8、集成解决5和问题7


对，应该放到 `activate.d` 的 `env_vars.sh` 里，这样每次 `conda activate himloco` 会自动生效。你可以把 **LD_LIBRARY_PATH 备份** 和 **torch 自带 CUDA 库优先 + LAZY** 一起写进去。

推荐脚本如下（放到 `~/miniconda3/envs/himloco/etc/conda/activate.d/env_vars.sh`）：

```sh
#!/bin/sh
# 备份原始路径
export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# torch 自带 CUDA 库优先
export TORCH_LIB="$CONDA_PREFIX/lib/python3.7/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 启用懒加载
export CUDA_MODULE_LOADING=LAZY
```

对应的退出脚本（`deactivate.d/env_vars.sh`）：

```sh
#!/bin/sh
# 恢复原始状态
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
unset OLD_LD_LIBRARY_PATH
unset TORCH_LIB
unset CUDA_MODULE_LOADING
```

注意：你刚才在终端执行 `./etc/conda/activate.d/env_vars.sh` 会报 127，是因为脚本还没创建或不可执行。放好脚本后，**不需要手动执行**，只要重新 `conda activate himloco` 就会自动加载。
