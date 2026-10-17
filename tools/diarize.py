#!/usr/bin/env python3
"""按已有句时间重新聚类声线；不修改识别文本与时间。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from s2s.diarize.embeddings import MODEL_DIR, MODEL_FILES
from s2s.diarize.pipeline import run, write_back


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("视频", type=Path)
    parser.add_argument("声音记录", type=Path)
    parser.add_argument("输出", type=Path)
    parser.add_argument("--模型", choices=MODEL_FILES, default="campp")
    parser.add_argument("--去配乐", action="store_true")
    parser.add_argument("--阈值", type=float, default=0.6)
    parser.add_argument("--重叠阈值", type=float, default=0.35)
    parser.add_argument("--分窗秒", type=float, default=3.0)
    parser.add_argument("--线程", type=int, default=2)
    parser.add_argument("--模型目录", type=Path, default=MODEL_DIR)
    parser.add_argument("--约束", nargs="*", choices=["lipsync"], default=["lipsync"])
    parser.add_argument("--口型", type=Path)
    parser.add_argument("--试听目录", type=Path)
    parser.add_argument("--写回", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.输出.exists():
            raise FileExistsError(f"输出已存在，不覆盖：{args.输出}")
        if args.线程 < 1:
            raise ValueError("线程数须为正整数")
        records = json.loads(args.声音记录.read_text())
        if args.写回 and json.loads(args.写回.read_text()) != records:
            raise ValueError("写回目标与声音记录不一致")
        constraints = None
        if args.约束 is not None:
            constraints = {"constraints": args.约束}
        lipsync = json.loads(args.口型.read_text()) if args.口型 else None
        result = run(args.视频, records, model=args.模型, denoise=args.去配乐,
                     threshold=args.阈值, overlap_threshold=args.重叠阈值,
                     window_seconds=args.分窗秒, words=(),
                     model_dir=args.模型目录, threads=args.线程, constraints=constraints, lipsync=lipsync)
        if args.试听目录:
            from s2s.diarize.integration import export_representatives
            export_representatives(args.视频, result, args.试听目录)
        if args.写回:
            if args.写回.resolve() == args.输出.resolve():
                raise ValueError("结果输出和写回目标不能相同")
            result["写回"] = write_back(args.写回, records, result)
        args.输出.parent.mkdir(parents=True, exist_ok=True)
        with args.输出.open("x") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(json.dumps({"输出": str(args.输出), "类数": len(result["类别"]),
                          "待定句数": sum(r["待定"] for r in result["句列表"]),
                          "运行": result["运行"]}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"错误": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
