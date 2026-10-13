#!/usr/bin/env bash
# Start the installed full editor with process-local Conda activation.
set -euo pipefail
source /home/qyw/miniconda3/etc/profile.d/conda.sh
conda activate /home/qyw/miniconda3/envs/himloco
task_log_dir=/home/qyw/.local/state/isaacsim-desktop
mkdir -p "$task_log_dir"
exec /home/qyw/miniconda3/envs/himloco/bin/isaacsim isaacsim.exp.full "$@" \
    >> "$task_log_dir/launcher.log" 2>&1
