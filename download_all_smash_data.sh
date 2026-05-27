#!/usr/bin/env bash
#
# 下载 EricChen06 名下的 smash 数据集。
#
# 用法:
#   ./download_all_smash_data.sh [HF dataset cache目录]
#
# 示例:
#   ./download_all_smash_data.sh
#   ./download_all_smash_data.sh ~/.cache/huggingface/dataset
#   HF_DATASET_CACHE_DIR=~/.cache/huggingface/dataset ./download_all_smash_data.sh
#
# 首次使用前请确保 uv 环境内可用 Hugging Face CLI:
#   uv add huggingface_hub
#
# 私有数据集或限流场景可先设置 HF_TOKEN，或执行:
#   uv run hf auth login

set -euo pipefail

# ===================== 配置区域 =====================
HF_NAMESPACE="EricChen06"
CONFIG_PATH="${CONFIG_PATH:-policy/src/openpi/training/config.py}"
HF_DATASET_CACHE_DIR="${1:-${HF_DATASET_CACHE_DIR:-${HOME}/.cache/huggingface/dataset}}"
# ====================================================

DATASETS=(
    white_smash_01
    white_smash_02
    white_smash_03
    white_smash_04
    yellow_smash_01
    yellow_smash_02
    yellow_smash_03
    yellow_smash_04
    black_smash_01
    black_smash_02
    black_smash_03
    black_smash_04
    black_smash_05
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

load_lerobot_config() {
    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "=========================================="
        echo "未找到配置文件: $CONFIG_PATH"
        echo "请通过 CONFIG_PATH 指定正确的 config.py 路径"
        echo "=========================================="
        exit 1
    fi

    LEROBOT_NAMESPACE="$(
        python - "$CONFIG_PATH" <<'PY'
import ast
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    tree = ast.parse(f.read(), filename=config_path)

namespace = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "DATASET_REPO_NAMESPACE":
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    namespace = value.value
                elif isinstance(value, ast.Str):
                    namespace = value.s
if not namespace:
    raise SystemExit("ERROR: DATASET_REPO_NAMESPACE 未在 config.py 中找到或不是字符串常量")
print(namespace)
PY
    )"

    LEROBOT_CACHE_DIR="${HOME}/.cache/huggingface/lerobot/${LEROBOT_NAMESPACE}"
}

check_deps() {
    if ! command -v uv &>/dev/null; then
        echo "=========================================="
        echo " 未检测到 uv，请先安装 uv:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo ""
        echo " 然后在项目环境中安装 huggingface_hub:"
        echo "   uv add huggingface_hub"
        echo "=========================================="
        exit 1
    fi

    if ! uv run hf --version &>/dev/null; then
        echo "=========================================="
        echo " uv 环境中未检测到 hf 命令，请执行:"
        echo "   uv add huggingface_hub"
        echo ""
        echo " 安装后如需登录，请执行:"
        echo "   uv run hf auth login"
        echo "=========================================="
        exit 1
    fi
}

create_lerobot_symlink() {
    local dataset_name="$1"
    local source_dir="$2"
    local link_path="${LEROBOT_CACHE_DIR}/${dataset_name}"
    local source_abs
    source_abs="$(realpath "$source_dir")"

    mkdir -p "$LEROBOT_CACHE_DIR"

    if [[ -L "$link_path" ]]; then
        rm -f "$link_path"
    elif [[ -e "$link_path" ]]; then
        log "警告: 目标已存在且不是软链接，跳过: $link_path"
        return 0
    fi

    ln -s "$source_abs" "$link_path"
    log "已创建软链接 ${link_path} -> ${source_abs}"
}

get_snapshot_dir() {
    local dataset_name="$1"
    local repo_cache_dir="${HF_DATASET_CACHE_DIR}/datasets--${HF_NAMESPACE//\//--}--${dataset_name}"
    local snapshots_dir="${repo_cache_dir}/snapshots"
    local latest_snapshot=""

    if [[ ! -d "$snapshots_dir" ]]; then
        echo ""
        return 0
    fi

    latest_snapshot="$(ls -1dt "${snapshots_dir}"/* 2>/dev/null | head -n 1 || true)"
    echo "$latest_snapshot"
}

download_dataset() {
    local dataset_name="$1"
    local repo_id="${HF_NAMESPACE}/${dataset_name}"
    local snapshot_dir=""

    log "开始下载 ${repo_id} (cache: ${HF_DATASET_CACHE_DIR})"

    uv run hf download "$repo_id" \
        --repo-type dataset \
        --cache-dir "$HF_DATASET_CACHE_DIR"

    snapshot_dir="$(get_snapshot_dir "$dataset_name")"
    if [[ -z "$snapshot_dir" ]]; then
        log "错误: 未找到 snapshot 目录，期望路径: ${HF_DATASET_CACHE_DIR}/datasets--${HF_NAMESPACE//\//--}--${dataset_name}/snapshots/*"
        return 1
    fi

    create_lerobot_symlink "$dataset_name" "$snapshot_dir"
    log "完成下载 ${repo_id}"
}

main() {
    check_deps
    load_lerobot_config
    mkdir -p "$HF_DATASET_CACHE_DIR"

    log "============================================"
    log "Hugging Face smash 数据集下载脚本启动"
    log "命名空间: ${HF_NAMESPACE}"
    log "HF dataset cache: ${HF_DATASET_CACHE_DIR}"
    log "LeRobot namespace: ${LEROBOT_NAMESPACE}"
    log "LeRobot 链接目录: ${LEROBOT_CACHE_DIR}"
    log "数据集数量: ${#DATASETS[@]}"
    log "============================================"

    for dataset_name in "${DATASETS[@]}"; do
        download_dataset "$dataset_name"
    done

    log "全部数据集下载完成"
}

main "$@"
