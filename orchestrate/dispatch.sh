#!/bin/bash
# 派发一个任务给执行 agent。
# 用法：orchestrate/dispatch.sh <任务编号> [补充说明]
# 模型可用环境变量切换：WORKER_THINKING；WORKER_MODEL 只接受生产模型
set -u
cd "$(dirname "$0")/.." || exit 2

TASK="${1:?需要任务编号}"
EXTRA="${2:-}"
MODEL="${WORKER_MODEL:-opencode-go/deepseek-v4.1-flash}"
[ "$MODEL" = "opencode-go/deepseek-v4.1-flash" ] || { echo "生产模型只允许 opencode-go/deepseek-v4.1-flash"; exit 2; }
THINKING="${WORKER_THINKING:-high}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="orchestrate/logs/${TASK}-${TS}.log"

[ -f "orchestrate/tasks/${TASK}.md" ] || { echo "缺少任务单 orchestrate/tasks/${TASK}.md"; exit 2; }
mkdir -p orchestrate/logs orchestrate/sessions orchestrate/reports

PROMPT="你的任务编号是 ${TASK}。先阅读 orchestrate/tasks/_通用说明.md，再阅读 orchestrate/tasks/${TASK}.md，按要求完成任务，最后写 orchestrate/reports/${TASK}.md。"
[ -n "$EXTRA" ] && PROMPT="${PROMPT}

编排者补充说明：${EXTRA}"

{
  echo "# task=${TASK} model=${MODEL} thinking=${THINKING} start=$(date '+%F %T')"
  pi -p --model "$MODEL" --thinking "$THINKING" \
     --session-dir orchestrate/sessions --name "$TASK-$TS" \
     "$PROMPT"
  echo "# exit=$? end=$(date '+%F %T')"
} > "$LOG" 2>&1

tail -5 "$LOG"
