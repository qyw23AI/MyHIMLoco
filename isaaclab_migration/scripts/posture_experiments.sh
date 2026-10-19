#!/usr/bin/env bash
# Sequential diagnostic training; Python path can be overridden without installing anything.
set -euo pipefail
POSTURE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
POSTURE_PYTHON="${TASK_GAIT_PYTHON:-/home/qyw/miniconda3/envs/himloco/bin/python}"
cd "$POSTURE_ROOT"
exec "$POSTURE_PYTHON" isaaclab_migration/scripts/posture_experiments.py "$@"
