#!/usr/bin/env bash
# Run policy/scripts/test_single_inf.py if that debug script exists.
# PyAV/LeRobot requires newer FFmpeg shared libraries; this environment commonly
# needs FFmpeg 7.x (libavformat.so.61), while the system FFmpeg 4.x is incompatible.
# This wrapper prefers FFmpeg from the active conda environment when available.

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

cd "$PROJECT_ROOT"
uv run python policy/scripts/test_single_inf.py "$@"
