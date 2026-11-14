#!/usr/bin/env bash
# Shortcuts for the migrated A1 HIMLoco paper-style experiment.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${TASK_PYTHON:-${PYTHON:-python}}"
LOG_DIR="${A1_LOG_DIR:-$ROOT/isaaclab_migration/logs/a1/upstream_paper_4096x100_5000_20261023}"
OUTPUT_DIR="${A1_OUTPUT_DIR:-$ROOT/isaaclab_migration/outputs/a1/upstream_paper_4096x100_5000_20261023}"
CHECKPOINT="${A1_CHECKPOINT:-$LOG_DIR/model_5000.pt}"
POLICY="${A1_POLICY:-$OUTPUT_DIR/policy.pt}"
VIDEO_DIR="${A1_VIDEO_DIR:-$OUTPUT_DIR/video}"

export PYTHONPATH="$ROOT/isaaclab_migration:$ROOT/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

usage() {
    cat <<'HELP'
Usage: bash isaaclab_migration/scripts/a1_himloco.sh ACTION [options]

Actions:
  train [train.py options]       Train A1 with paper settings (4096 x 100 x 5000)
  export [--checkpoint PATH]     Export model_N.pt to TorchScript policy.pt
  view [view options]            Real-time GUI playback; no frame capture
  record [record options]        GUI replay and encode playback.mp4
  all                            Train, export model_5000.pt, then record it

Environment overrides:
  TASK_PYTHON, A1_LOG_DIR, A1_OUTPUT_DIR, A1_CHECKPOINT, A1_POLICY, A1_VIDEO_DIR

Examples:
  bash isaaclab_migration/scripts/a1_himloco.sh train
  bash isaaclab_migration/scripts/a1_himloco.sh export
  bash isaaclab_migration/scripts/a1_himloco.sh record --seconds 20
HELP
}

export_policy() {
    local checkpoint="$1"
    mkdir -p "$(dirname -- "$POLICY")"
    "$PYTHON" isaaclab_migration/scripts/export_policy.py \
        --task a1 --checkpoint "$checkpoint" --output "$POLICY"
}

record_policy() {
    local checkpoint="$1"
    shift
    local steps=4000
    local num_envs=128
    while (($#)); do
        case "$1" in
            --seconds)
                steps="$(awk "BEGIN { if ($2 <= 0) exit 1; printf \"%.0f\", $2 * 50 }")" ||
                    { echo 'record: --seconds must be positive' >&2; return 2; }
                shift 2
                ;;
            --steps) steps="$2"; shift 2 ;;
            --num_envs) num_envs="$2"; shift 2 ;;
            *) echo "record: unknown option: $1" >&2; return 2 ;;
        esac
    done
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "record requires ffmpeg to encode PNG frames into MP4" >&2
        return 1
    fi
    export_policy "$checkpoint"
    rm -rf "$VIDEO_DIR"
    mkdir -p "$VIDEO_DIR"
    local terrain_count
    terrain_count="$("$PYTHON" -c \
        'from himloco_lab.config import load_config; print(load_config("a1", 4096, 4)[0].terrain.num_cols)')"
    [[ "$terrain_count" =~ ^[1-9][0-9]*$ ]] || {
        echo "record: invalid terrain count: $terrain_count" >&2
        return 1
    }
    ((num_envs >= terrain_count)) || {
        echo "record: --num_envs must be at least $terrain_count for per-terrain recording" >&2
        return 2
    }
    local base_steps=$((steps / terrain_count))
    local extra_steps=$((steps % terrain_count))
    local col env_index segment_steps segment_dir
    for ((col=0; col<terrain_count; col++)); do
        # terrain_types uses floor(env_index / (num_envs / num_cols)).
        env_index=$((((col + 1) * num_envs + terrain_count - 1) / terrain_count - 1))
        segment_steps=$base_steps
        if ((col < extra_steps)); then
            segment_steps=$((segment_steps + 1))
        fi
        segment_dir="$VIDEO_DIR/terrain_$(printf '%02d' "$col")"
        mkdir -p "$segment_dir"
        printf 'RECORDING_TERRAIN %d/%d env=%d steps=%d\n' \
            "$((col + 1))" "$terrain_count" "$env_index" "$segment_steps"
        "$PYTHON" isaaclab_migration/scripts/play.py \
            --task a1 --stage 4 --policy "$POLICY" --num_envs "$num_envs" \
            --steps "$segment_steps" --capture_dir "$segment_dir" --capture_every 10 \
            --follow_camera --camera_env "$env_index" --no-realtime \
            --viewer_resolution 960 540 --antialiasing FXAA \
            --state_csv "$segment_dir/state.csv"
        ffmpeg -y -loglevel error -framerate 5 \
            -i "$segment_dir/frame_%06d.png" -c:v libx264 -pix_fmt yuv420p \
            "$segment_dir/playback.mp4"
        printf 'VIDEO_SAVED %s\n' "$segment_dir/playback.mp4"
    done
    printf 'TERRAIN_VIDEOS_COMPLETE %s terrains=%s total_steps=%s\n' \
        "$VIDEO_DIR" "$terrain_count" "$steps"
}

view_policy() {
    local checkpoint="$1"
    shift
    local seconds=0
    local num_envs=4
    local command=(1 0 0)
    while (($#)); do
        case "$1" in
            --seconds) seconds="$2"; shift 2 ;;
            --num_envs) num_envs="$2"; shift 2 ;;
            --command) command=("$2" "$3" "$4"); shift 4 ;;
            *) echo "view: unknown option: $1" >&2; return 2 ;;
        esac
    done
    export_policy "$checkpoint"
    "$PYTHON" isaaclab_migration/scripts/play.py \
        --task a1 --stage 4 --policy "$POLICY" --num_envs "$num_envs" \
        --seconds "$seconds" --command "${command[@]}" \
        --viewer_resolution 960 540 --antialiasing FXAA
}

action="${1:-help}"
if (($#)); then shift; fi
case "$action" in
    train)
        mkdir -p "$LOG_DIR"
        "$PYTHON" isaaclab_migration/scripts/train.py \
            --task a1 --stage 4 --num_envs 4096 --paper_a1_rollout 100 \
            --iterations 5000 --log_dir "$LOG_DIR" "$@"
        ;;
    export)
        checkpoint="$CHECKPOINT"
        if (($#)); then
            [[ "$1" == "--checkpoint" && $# -ge 2 ]] || {
                echo 'export accepts only --checkpoint PATH' >&2; exit 2;
            }
            checkpoint="$2"
            shift 2
        fi
        (($# == 0)) || { echo "export: unexpected arguments: $*" >&2; exit 2; }
        export_policy "$checkpoint"
        ;;
    view)
        view_policy "$CHECKPOINT" "$@"
        ;;
    record)
        record_policy "$CHECKPOINT" "$@"
        ;;
    all)
        "$0" train "$@"
        "$0" export
        "$0" record
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown action: $action" >&2
        usage >&2
        exit 2
        ;;
esac
