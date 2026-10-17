#!/usr/bin/env python3
"""本地 Qwen 单路识别、按音频切句、声纹及异常回听清单。"""

import argparse
import json
from pathlib import Path
import sys

from s2s.asr.single import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--provider", choices=("local",), help="只允许本地 Qwen3-ASR")
    parser.add_argument("--words", type=Path, action="append", default=[], help="复用真实词级响应，不调用 ASR；可重复")
    parser.add_argument("--人声段", "--recover-from", dest="voices", type=Path, help="复用人声段，不接受字幕")
    parser.add_argument("--原裁定", type=Path, help="按时间、文本及来源严格迁移原回听结果")
    parser.add_argument("--min-pause", type=float)
    parser.add_argument("--max-sentence", type=float)
    args = parser.parse_args(argv)
    try:
        result = run(args.video, args.episode_dir, word_paths=args.words, voice_path=args.voices,
                     decisions_path=args.原裁定, min_pause=args.min_pause, max_sentence=args.max_sentence)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        print(json.dumps({"通过": False, "错误": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
