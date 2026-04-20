#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 若 conda 环境中有 FFmpeg/libstdc++，优先让 PyAV 等依赖从该环境加载动态库，
# 避免系统库版本过低导致 CXXABI 或 av 相关导入失败。
if [[ -n "$CONDA_PREFIX" ]]; then
    CONDA_LIB="$CONDA_PREFIX/lib"
    if [[ -d "$CONDA_LIB" ]]; then
        export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

cd "$PROJECT_ROOT"
uv run python deploy_scripts/infer.py "$@"
