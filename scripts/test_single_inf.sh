#!/usr/bin/env bash
# 运行 inference_singlestep.py
# PyAV/LeRobot 需要 FFmpeg 7.x (libavformat.so.71)，系统默认 FFmpeg 4.x 不兼容。
# 此脚本会优先使用 conda 环境中的 FFmpeg（若已安装）。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 若 conda 环境中有 FFmpeg，让 PyAV 能找到 libavformat.so
if [[ -n "$CONDA_PREFIX" ]]; then
    CONDA_LIB="$CONDA_PREFIX/lib"
    if [[ -d "$CONDA_LIB" ]] && ls "$CONDA_LIB"/libavformat.so* 1>/dev/null 2>&1; then
        export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

cd "$PROJECT_ROOT"
uv run python policy/scripts/test_single_inf.py "$@"
