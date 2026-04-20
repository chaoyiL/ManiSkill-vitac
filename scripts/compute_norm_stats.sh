#!/usr/bin/env bash
# 运行 convert_zarr_to_lerobot.py
# PyAV/LeRobot 需要 FFmpeg 6.x (libavformat.so.61)，系统默认 FFmpeg 4.x 不兼容。
# 此脚本会优先使用 conda 环境中的 FFmpeg（若已安装）。
#
# 使用前请确保：
#   方案 A（推荐）：在 conda 环境中安装 FFmpeg 6.x
#     conda install -c conda-forge ffmpeg
#
# 用法：./compute_norm_stats.sh [CONFIG] [其他参数...]
#   CONFIG: 配置名，默认 pi05_chaoyi

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

# config 可通过第一个参数传入，默认为 pi05_chaoyi
if [[ -n "${1:-}" ]]; then
    CONFIG="$1"
    shift
else
    CONFIG="pi05_chaoyi"
fi

cd "$PROJECT_ROOT"
uv run python policy/scripts/compute_norm_stats.py --config-name "$CONFIG" "$@"
