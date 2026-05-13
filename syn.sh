#!/bin/bash
#
# 监控 checkpoint 文件夹，检测到新 checkpoint 后：
#   1. 清空 Google Drive 上的旧内容
#   2. 只上传最新的 checkpoint 子文件夹
#
# 用法:
#   ./sync_to_gdrive.sh [检查间隔秒数] [rclone远端名]
#
# 示例:
#   ./sync_to_gdrive.sh              # 默认每 300 秒检查一次
#   ./sync_to_gdrive.sh 600 gdrive   # 每 600 秒检查，使用 gdrive 远端
#
# 后台运行:
#   nohup ./sync_to_gdrive.sh > sync.out 2>&1 &
#
# 首次使用前需要:
#   1. 安装 rclone:  curl https://rclone.org/install.sh | sudo bash
#   2. 配置远端:     rclone config  (选 Google Drive，取名如 gdrive)

set -euo pipefail

# ===================== 配置区域 =====================
WATCH_DIR="~/ManiSkill-vitac/checkpoints/pi05_bi_vitac/my_experiment"
REMOTE_PATH="ckpt"
# ====================================================

INTERVAL="${1:-300}"
REMOTE_NAME="${2:-gdrive}"
REMOTE_FULL="${REMOTE_NAME}:${REMOTE_PATH}"
LAST_UPLOADED=""
LOG_FILE="sync_gdrive_$(date +%Y%m%d_%H%M%S).log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

check_deps() {
    if ! command -v rclone &>/dev/null; then
        echo "=========================================="
        echo " rclone 未安装，请先执行:"
        echo "   curl https://rclone.org/install.sh | sudo bash"
        echo ""
        echo " 安装后配置 Google Drive:"
        echo "   rclone config"
        echo "   -> New remote -> 取名 '${REMOTE_NAME}'"
        echo "   -> 选择 'Google Drive' -> 按提示完成授权"
        echo "=========================================="
        exit 1
    fi

    if ! rclone listremotes 2>/dev/null | grep -q "^${REMOTE_NAME}:$"; then
        echo "=========================================="
        echo " 远端 '${REMOTE_NAME}' 未配置，请运行:"
        echo "   rclone config"
        echo "=========================================="
        exit 1
    fi
}

get_latest_checkpoint() {
    # 找到最新修改的子文件夹（即最新的 checkpoint）
    local latest
    latest=$(ls -dt "${WATCH_DIR}"/*/ 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        basename "$latest"
    fi
}

delete_local_checkpoint() {
    local ckpt_name="$1"
    local ckpt_path="${WATCH_DIR}/${ckpt_name}"

    # 仅允许删除 WATCH_DIR 下的一级子目录，避免误删
    if [ -z "$ckpt_name" ] || [ "$ckpt_name" = "." ] || [ "$ckpt_name" = ".." ]; then
        log "警告: 跳过删除，checkpoint 名称非法: '${ckpt_name}'"
        return 1
    fi

    if [ ! -d "$ckpt_path" ]; then
        log "警告: 跳过删除，本地目录不存在: ${ckpt_path}"
        return 0
    fi

    case "$ckpt_path" in
        "${WATCH_DIR}/"*)
            ;;
        *)
            log "错误: 目录校验失败，拒绝删除: ${ckpt_path}"
            return 1
            ;;
    esac

    log "删除本地已上传目录: ${ckpt_path}"
    rm -rf "$ckpt_path"
    log "本地目录删除完成: ${ckpt_name}"
}

upload_checkpoint() {
    local ckpt_name="$1"
    local ckpt_path="${WATCH_DIR}/${ckpt_name}"

    if [ ! -d "$ckpt_path" ]; then
        log "错误: ${ckpt_path} 不存在"
        return 1
    fi

    local ckpt_size
    ckpt_size=$(du -sh "$ckpt_path" 2>/dev/null | cut -f1)
    log "最新 checkpoint: ${ckpt_name} (${ckpt_size})"

    log "清空云盘: ${REMOTE_FULL} ..."
    rclone delete "${REMOTE_FULL}" --rmdirs 2>/dev/null || true
    log "清空云盘回收站..."
    rclone cleanup "${REMOTE_NAME}:" 2>/dev/null || true
    log "云盘清理完成"

    log "创建云盘目标目录..."
    rclone mkdir "${REMOTE_FULL}/${ckpt_name}" 2>/dev/null || true

    log "开始上传 ${ckpt_name} ..."
    local start_time
    start_time=$(date +%s)

    rclone copy "${ckpt_path}" "${REMOTE_FULL}/${ckpt_name}" \
        --transfers 8 \
        --checkers 16 \
        --log-file "$LOG_FILE" \
        --log-level INFO \
        --stats 30s \
        --stats-log-level NOTICE
    local rc=$?

    local end_time elapsed
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    if [ $rc -eq 0 ]; then
        log "上传完成! 耗时: ${elapsed} 秒"

        local remote_count
        remote_count=$(rclone size "${REMOTE_FULL}/${ckpt_name}" 2>/dev/null | head -1 || echo "未知")
        log "云盘内容: ${remote_count}"
        delete_local_checkpoint "$ckpt_name"
        return 0
    else
        log "错误: 上传可能失败 (exit code: ${rc}), 耗时: ${elapsed} 秒"
        log "提示: 请到 Google Drive 检查文件是否实际上传成功"
        return 1
    fi
}

cleanup() {
    log "收到终止信号，退出..."
    exit 0
}
trap cleanup SIGINT SIGTERM

main() {
    check_deps

    log "============================================"
    log "Checkpoint 自动同步脚本启动"
    log "  监控目录:   ${WATCH_DIR}"
    log "  云盘目标:   ${REMOTE_FULL}"
    log "  检查间隔:   ${INTERVAL} 秒"
    log "  日志文件:   ${LOG_FILE}"
    log "  Ctrl+C 停止"
    log "============================================"

    while true; do
        latest=$(get_latest_checkpoint)

        if [ -z "$latest" ]; then
            log "监控目录为空，等待 checkpoint 出现..."
        elif [ "$latest" != "$LAST_UPLOADED" ]; then
            # 从 checkpoint 名中解析“代数”（最后一个数字），只在能被 5000 整除时上传
            local gen
            gen=$(echo "$latest" | grep -oE '[0-9]+' | tail -1 || echo "")

            if [ -z "$gen" ]; then
                log "检测到新 checkpoint: ${latest}，但未解析到代数数字，默认跳过上传"
            elif [ $((gen % 5000)) -ne 0 ]; then
                log "检测到新 checkpoint: ${latest} (代数: ${gen})，不满足每 5000 代上传条件，跳过"
            else
                log "检测到新 checkpoint: ${latest} (代数: ${gen})，满足每 5000 代上传条件 (gen % 5000 == 0)"
                if upload_checkpoint "$latest"; then
                    LAST_UPLOADED="$latest"
                    log "已记录上传: ${LAST_UPLOADED}"
                fi
            fi
        else
            log "无变化 (当前最新: ${latest})"
        fi

        sleep "$INTERVAL"
    done
}

main
