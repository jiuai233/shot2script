#!/usr/bin/env python3
"""本机跨集声纹入库、比对和人工真值评测。"""

import argparse
import json
from pathlib import Path
import subprocess

from s2s.voiceprint.library import enroll, compare, read


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("入库", "比对"):
        command = commands.add_parser(name)
        command.add_argument("剧目目录", type=Path)
        command.add_argument("集号")
        command.add_argument("--音频", type=Path)
        command.add_argument("--模型", choices=["campp", "resnet34"], default="campp")
    evaluation = commands.add_parser("评测")
    evaluation.add_argument("聚类结果", type=Path)
    evaluation.add_argument("标注", type=Path)
    evaluation.add_argument("--声线角色映射", type=Path)
    evaluation.add_argument("--声音记录", type=Path)
    evaluation.add_argument("--回听结果", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "评测":
            from s2s.voiceprint.evaluation import evaluate, evaluate_listen
            mapping = read(args.声线角色映射) if args.声线角色映射 else None
            result = evaluate(read(args.聚类结果), read(args.标注), mapping)
            if args.回听结果:
                if args.声音记录 is None:
                    raise ValueError("回听评测须同时提供 --声音记录")
                result["回听"] = evaluate_listen(read(args.声音记录), read(args.回听结果))
        else:
            function = enroll if args.command == "入库" else compare
            result = function(args.剧目目录, args.集号, args.音频, args.模型)
        # 入库函数的返回值和比对文件均不含声纹向量。
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, ImportError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"声纹操作失败：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
