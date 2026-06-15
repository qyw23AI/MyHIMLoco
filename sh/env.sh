#!/usr/bin/env bash
# 自动生成的环境变量配置，由 deploy.sh 生成
# 使用方式：source $(dirname "${BASH_SOURCE[0]}")/env.sh

export MUJOCO_GL=egl
export MUJOCO_DIR="/root/.mujoco"

# conda 路径
export PATH="/opt/conda/bin:${PATH}"

# 动态库搜索路径
export LD_LIBRARY_PATH="/opt/conda/envs/himloco/lib:/opt/conda/lib:/usr/lib/x86_64-linux-gnu:/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

# 显示设置（如需 VNC）
export DISPLAY="${DISPLAY:-:1}"

# VirtualGL 和 TurboVNC 路径
export VGL_HOME="/opt/VirtualGL"
export TURBOVNC_HOME="/opt/TurboVNC"
export PATH="${TURBOVNC_HOME}/bin:${VGL_HOME}/bin:${PATH}"
export VGL_GLFLUSH=1
export VGL_VSYNC=1

# 激活 conda 环境
if [[ "$1" != "--no-activate" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh" && conda activate himloco
fi

echo "[env] conda env: himloco"
echo "[env] MUJOCO_GL: ${MUJOCO_GL}"
echo "[env] MUJOCO_DIR: ${MUJOCO_DIR}"
echo "[env] LD_LIBRARY_PATH configured"
