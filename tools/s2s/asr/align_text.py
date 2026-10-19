"""独立对齐指定文本；只加载本地强制对齐模型。"""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

from .local_runtime import coalesce_zero_spans, restore_punctuation


def align(audio_path, text, model_path):
    from mlx_audio.stt.utils import load_model

    started = time.monotonic()
    model = load_model(Path(model_path))
    loaded = time.monotonic()
    aligned = model.generate(str(audio_path), text=text, language="Chinese")
    words = restore_punctuation(text, aligned.segments)
    return {"text": text, "words": coalesce_zero_spans(words),
            "zero_span_tokens": sum(w["start"] == w["end"] for w in words),
            "model_path": str(model_path), "load_seconds": loaded - started,
            "alignment_seconds": time.monotonic() - loaded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("text", type=Path, help="UTF-8 转写文本文件")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"输出已存在：{args.output}")
    if not args.audio.is_file() or not args.text.is_file() or not args.model.is_dir():
        parser.error("缺少音频、文本或本地对齐模型")
    with redirect_stdout(sys.stderr):
        result = align(args.audio, args.text.read_text(encoding="utf-8"), args.model)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
