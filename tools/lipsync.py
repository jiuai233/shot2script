"""用法：uv run --with mediapipe python tools/lipsync.py 视频 本集目录 输出.json。"""

import argparse
from dataclasses import asdict, fields
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2s.lipsync import Parameters
from s2s.lipsync.pipeline import run, review_existing, rescore_existing


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU 嘴部关键点与音量同步检测；输出旁生成独立离线核对页")
    parser.add_argument("video")
    parser.add_argument("episode", metavar="本集目录")
    parser.add_argument("output", metavar="输出.json")
    defaults = asdict(Parameters())
    for field in fields(Parameters):
        parser.add_argument("--" + field.name.replace("_", "-"), type=field.type, default=field.default)
    parser.add_argument("--阈值", dest="threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--复核", action="store_true", dest="review")
    parser.add_argument("--variant", choices=["baseline", "side", "model", "fused"], default="baseline")
    parser.add_argument("--yunet", default="models/lipsync/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--asd-python", default="envs/asd/bin/python")
    parser.add_argument("--asd-weights", default="models/lipsync/Light-ASD/pretrain_AVA_CVPR.model")
    parser.add_argument("--review-source", help="仅复核已有记录，仍需 --复核；输出到新目录")
    parser.add_argument("--reuse-source", help="复用 side 记录的规则证据，补 model 或 fused 分数")
    parser.add_argument("--asd-source", help="复用输入完全一致的 asd输出.json，配合 --reuse-source")
    parser.add_argument("--asd-timeout", type=float, default=1200, help="ASD 子进程上限秒数，默认 1200")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "models/lipsync/face_landmarker.task"))
    args = parser.parse_args(argv)
    try:
        params = Parameters(**{k: getattr(args, k) for k in defaults})
        if args.review_source and args.reuse_source:
            raise ValueError("--review-source 与 --reuse-source 不能同时使用")
        if args.asd_source and not args.reuse_source:
            raise ValueError("--asd-source 须配合 --reuse-source")
        if args.asd_timeout <= 0:
            raise ValueError("--asd-timeout 须为正数")
        if args.review_source:
            if not args.review:
                raise ValueError("--review-source 须同时指定 --复核")
            result = review_existing(args.review_source, args.output, args.video)
        elif args.reuse_source:
            if args.review:
                raise ValueError("复用评分后请用 --review-source 单独复核")
            if params != Parameters():
                raise ValueError("复用评分沿用源记录参数，不接受参数覆写")
            result = rescore_existing(args.video, args.reuse_source, args.output, args.variant,
                                      args.asd_python, args.asd_weights, args.asd_timeout, args.asd_source)
        else:
            result = run(args.video, args.episode, args.output, params, args.review, args.model,
                         args.variant, args.yunet, args.asd_python, args.asd_weights, args.asd_timeout)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"口型检测未完成：{exc}\n")
    print(json.dumps({"输出": args.output, "统计": result["统计"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
