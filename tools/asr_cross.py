#!/usr/bin/env python3
"""旧双路命令已停用，生产语音入口为 tools/asr.py。"""
import sys

if __name__ == "__main__":
    print("双路 ASR 已停用；请使用 tools/asr.py 的本地 Qwen 单路入口。", file=sys.stderr)
    raise SystemExit(2)
