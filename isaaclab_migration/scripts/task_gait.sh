#!/usr/bin/env bash
# Local task-competition workflow. No installations or changes to system drivers.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${TASK_GAIT_PYTHON:-}" ]]; then
    TASK_PYTHON="$TASK_GAIT_PYTHON"
elif [[ -x /home/qyw/miniconda3/envs/himloco/bin/python ]]; then
    TASK_PYTHON=/home/qyw/miniconda3/envs/himloco/bin/python
else
    TASK_PYTHON=python
fi
export PYTHONPATH="$TASK_ROOT/isaaclab_migration:$TASK_ROOT/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
ACTION="${1:-help}"
if (($#)); then shift; fi
cd "$TASK_ROOT"
case "$ACTION" in
    check)
        "$TASK_PYTHON" isaaclab_migration/tests/test_task_gait.py "$@"
        exec "$TASK_PYTHON" isaaclab_migration/scripts/preflight.py
        ;;
    smoke|train)
        TRAIN_ARGS=(--task mybot_v3_task_gait --stage 3 --num_envs 2048 --headless)
        if [[ "$ACTION" == smoke ]]; then
            TRAIN_ARGS+=(--num_envs 64 --iterations 2 --rollout_steps 24)
        fi
        # --phase is a human-facing alias; all other options go to train.py.
        EXTRA=()
        GUI=false
        TASK_LOG_DIR="$TASK_ROOT/isaaclab_migration/logs/mybot_v3_task_gait/run_$(date +%Y%m%d_%H%M%S)_$$"
        while (($#)); do
            case "$1" in
                --phase)
                    [[ $# -ge 2 ]] || { echo '--phase needs basic, precision or robust' >&2; exit 2; }
                    EXTRA+=(--gait_phase "$2"); shift 2 ;;
                --gui) GUI=true; shift ;;
                --log_dir)
                    [[ $# -ge 2 ]] || { echo '--log_dir needs a path' >&2; exit 2; }
                    TASK_LOG_DIR="$2"; shift 2 ;;
                --log_dir=*) TASK_LOG_DIR="${1#*=}"; shift ;;
                *) EXTRA+=("$1"); shift ;;
            esac
        done
        if "$GUI"; then
            CLEAN=("${TRAIN_ARGS[@]}")
            TRAIN_ARGS=()
            for arg in "${CLEAN[@]}"; do [[ "$arg" == --headless ]] || TRAIN_ARGS+=("$arg"); done
        fi
        [[ ! -e "$TASK_LOG_DIR/config.json" ]] || { echo 'Use a new log directory; resume with --checkpoint.' >&2; exit 2; }
        mkdir -p "$TASK_LOG_DIR"
        printf 'Task gait: %s, Python: %s\n' "$ACTION" "$TASK_PYTHON"
        "$TASK_PYTHON" isaaclab_migration/scripts/train.py "${TRAIN_ARGS[@]}" "${EXTRA[@]}" \
            --log_dir "$TASK_LOG_DIR" 2>&1 | tee "$TASK_LOG_DIR/console.log"
        # Kit can close with status 0 while an earlier Python traceback exists.
        # Require the end-of-run marker AND a fresh validation artifact.
        "$TASK_PYTHON" - "$TASK_LOG_DIR" <<'PY'
import json, pathlib, sys
folder = pathlib.Path(sys.argv[1])
if 'RUN_COMPLETED ' not in (folder / 'console.log').read_text():
    raise SystemExit('Training did not complete; inspect console.log')
if json.loads((folder / 'validation.json').read_text()).get('status') != 'passed':
    raise SystemExit('Training validation failed')
print('TRAINING_VERIFIED', folder.resolve())
PY
        ;;
    evaluate|play)
        EVAL_ARGS=(--num_envs 64 --headless)
        if [[ "$ACTION" == play ]]; then EVAL_ARGS=(--num_envs 4 --play --seconds 60); fi
        TASK_OUTPUT="$TASK_ROOT/isaaclab_migration/outputs/task_gait/${ACTION}_$(date +%Y%m%d_%H%M%S)_$$"
        EXTRA=()
        while (($#)); do
            case "$1" in
                --output)
                    [[ $# -ge 2 ]] || { echo '--output needs a path' >&2; exit 2; }
                    TASK_OUTPUT="$2"; shift 2 ;;
                --output=*) TASK_OUTPUT="${1#*=}"; shift ;;
                *) EXTRA+=("$1"); shift ;;
            esac
        done
        [[ ! -e "$TASK_OUTPUT/evaluation.json" ]] || { echo 'Use a new evaluation output directory.' >&2; exit 2; }
        mkdir -p "$TASK_OUTPUT"
        "$TASK_PYTHON" isaaclab_migration/scripts/task_gait_eval.py "${EVAL_ARGS[@]}" "${EXTRA[@]}" \
            --output "$TASK_OUTPUT" 2>&1 | tee "$TASK_OUTPUT/console.log"
        "$TASK_PYTHON" - "$TASK_OUTPUT" <<'PY'
import json, pathlib, sys
folder = pathlib.Path(sys.argv[1])
if 'EVALUATION_SAVED ' not in (folder / 'console.log').read_text():
    raise SystemExit('Evaluation did not complete; inspect console.log')
json.loads((folder / 'evaluation.json').read_text())
print('EVALUATION_VERIFIED', folder.resolve())
PY
        ;;
    *)
        cat <<'HELP'
Usage: bash isaaclab_migration/scripts/task_gait.sh ACTION [options]
  check                         CPU contracts and read-only GPU/dependency check
  smoke [--phase basic]         64 envs, 2 updates; pipeline check, not convergence
  train --phase basic          2048 envs, headless; --num_envs 3072 is supported
  train --phase precision --checkpoint /absolute/path/model.pt
  train --phase robust --checkpoint /absolute/path/model.pt
  evaluate --checkpoint /absolute/path/model.pt [--short --seconds 3]
  play --checkpoint /absolute/path/model.pt [--command 0.03 0 0] [--standby]
Train options: --iterations N --seed N --log_dir PATH --export_policy PATH --gui
Only new task-gait checkpoints can resume. Old 270-input policies are incompatible.
TASK_GAIT_PYTHON overrides the Python executable; this script installs nothing.
HELP
        ;;
esac
