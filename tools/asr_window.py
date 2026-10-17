#!/usr/bin/env python3
"""定向识别时间窗口，仅输出 JSON，不改声音记录。"""

import argparse
import json
from pathlib import Path
import sys

from s2s.asr import get_backend
from s2s.asr.common import ASRError
from s2s.asr.grok import keyterms_from
from s2s.asr.postprocess import seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("start", type=seconds, help="秒数或 MM:SS.s")
    parser.add_argument("end", type=seconds, help="秒数或 MM:SS.s")
    parser.add_argument("--provider", choices=("grok", "doubao", "local"))
    parser.add_argument("--keyterm", action="append", help="Grok 人名提示词，可重复传入")
    parser.add_argument("--raw-dir", type=Path, help="默认存放在输入文件同目录的 asr_raw 下")
    parser.add_argument("--keyterm-from", type=Path, action="append", help="从角色卡或字幕记录提取人名提示词，可重复")
    args = parser.parse_args(argv)
    try:
        backend = get_backend(args.provider)
        if args.keyterm or args.keyterm_from:
            if backend.provider != "grok":
                raise ValueError("--keyterm / --keyterm-from 仅用于 Grok 后端")
            terms = list(backend.settings.get("keyterms", [])) + (args.keyterm or [])
            for path in args.keyterm_from or []:
                terms.extend(keyterms_from(path))
            backend.settings["keyterms"] = list(dict.fromkeys(terms))
        if args.raw_dir:
            backend.settings["raw_dir"] = str(args.raw_dir)
        result = backend.transcribe(args.video, args.start, args.end)
        print(json.dumps([{"起": s.start, "止": s.end, "文本": s.text,
                           "声线": s.speaker, "听辨": s.clarity} for s in result], ensure_ascii=False, indent=2))
        return 0
    except (ASRError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
