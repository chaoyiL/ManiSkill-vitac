#!/usr/bin/env bash
# Run eval_scripts/loglike_evaluate.py.
# PyAV/LeRobot requires newer FFmpeg shared libraries; this environment commonly
# needs FFmpeg 7.x (libavformat.so.61), while the system FFmpeg 4.x is incompatible.
# This wrapper prefers FFmpeg from the active conda environment when available.
#
# Usage:
#   ./scripts/loglike_evaluate.sh CHECKPOINT_DIR EPISODE_INDEX [extra args...]
#   ./scripts/loglike_evaluate.sh CONFIG CHECKPOINT_DIR EPISODE_INDEX [extra args...]
#
# Extra args are passed through to eval_scripts/loglike_evaluate.py, for example:
#   --frame 0 --sample-interval 3 --num-steps 10 --finite-difference --fd-eps 1e-3 --remove-modality tactile --blur-sigma 8.0 --max-frames 10

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# If the conda environment provides FFmpeg, expose it so PyAV can find libavformat.so.
if [[ -n "$CONDA_PREFIX" ]]; then
    CONDA_LIB="$CONDA_PREFIX/lib"
    if [[ -d "$CONDA_LIB" ]] && ls "$CONDA_LIB"/libavformat.so* 1>/dev/null 2>&1; then
        export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

CONFIG="pi05_bi_vitac"
if [[ "$#" -lt 2 ]]; then
    echo "Usage: $0 [CONFIG] CHECKPOINT_DIR EPISODE_INDEX [extra args...]" >&2
    exit 2
fi

if [[ "$#" -ge 3 ]]; then
    CONFIG="$1"
    shift
fi

CHECKPOINT_DIR="$1"
EPISODE_INDEX="$2"
shift 2

cd "$PROJECT_ROOT"
uv run python eval_scripts/loglike_evaluate.py \
    --config-name "$CONFIG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --episode-index "$EPISODE_INDEX" \
    "$@"
