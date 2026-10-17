#!/usr/bin/env python3
"""本地 Qwen 定向识别，只输出 JSON，不改声音记录。"""

import argparse
import json
from pathlib import Path
import sys

from s2s.asr import get_backend
from s2s.asr.postprocess import seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("start", type=seconds)
    parser.add_argument("end", type=seconds)
    parser.add_argument("--provider", choices=("local",))
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        backend = get_backend(args.provider)
        backend.settings["raw_dir"] = str(args.raw_dir or args.video.parent / "asr_raw")
        result = backend.transcribe(args.video, args.start, args.end)
        print(json.dumps([{"起": s.start, "止": s.end, "文本": s.text,
                           "声线": s.speaker, "听辨": s.clarity} for s in result], ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
