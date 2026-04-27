#!/usr/bin/env bash
# Run policy/scripts/compute_norm_stats.py.
# PyAV/LeRobot requires newer FFmpeg shared libraries; this environment commonly
# needs FFmpeg 7.x (libavformat.so.61), while the system FFmpeg 4.x is incompatible.
# This wrapper prefers FFmpeg from the active conda environment when available.
#
# Before running, make sure the conda environment has FFmpeg 7.x and a recent libstdc++:
#     conda install -c conda-forge 'ffmpeg=7' libstdcxx-ng
# If PyAV needs to be built from source, also install:
#     conda install -c conda-forge pkg-config cython c-compiler
#
# Usage: ./compute_norm_stats.sh [CONFIG] [extra args...]
#   CONFIG: Training config name. Prefer passing pi05_bi_vitac or pi05_bi explicitly.
#           If omitted, the script still uses the historical default pi05_chaoyi.

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

# The first argument can override the config; otherwise use the historical default.
if [[ -n "${1:-}" ]]; then
    CONFIG="$1"
    shift
else
    CONFIG="pi05_chaoyi"
fi

cd "$PROJECT_ROOT"
uv run python policy/scripts/compute_norm_stats.py --config-name "$CONFIG" "$@"
