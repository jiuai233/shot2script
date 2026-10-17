#!/usr/bin/env python3
"""视频 → 本集声音记录.json；按本地人声段切句，原始响应保存到素材/asr_raw。"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

from s2s import audio
from s2s.asr import get_backend
from s2s.asr.common import ASRError
from s2s.asr.grok import keyterms_from
from s2s.asr.postprocess import records, normalized, seconds
from s2s.asr.segment import load_words, missing_voice, segment_pair
from s2s.config import load_config
from s2s.vad import detect, waveform


def recover_missing(backend, source, primary, voices, episode, *, secondary):
    """只在人声中两路均无词的窗口回查；primary/secondary 均须为真实词级结果。"""
    checks, words = [], list(primary)
    total = audio.duration(source)
    for index, row in enumerate(missing_voice(voices, primary, secondary)):
        start, end = seconds(row["起"]), seconds(row["止"])
        check = {"人声段": [start, end], "窗口": [max(0, start - 1), min(total, end + 1)], "新增": []}
        checks.append(check)
        old_vad = backend.settings.get("vad_threshold")
        try:
            if backend.provider == "grok":
                backend.settings["vad_threshold"] = 0
            backend.transcribe(source, *check["窗口"])
            candidates = getattr(backend, "last_words", None)
            if candidates is None:
                raise ValueError("补识别后端未保留原始词级时间")
        except (ASRError, ValueError) as exc:
            check["错误"] = str(exc)
            continue
        finally:
            if old_vad is None:
                backend.settings.pop("vad_threshold", None)
            else:
                backend.settings["vad_threshold"] = old_vad
        for item in candidates:
            a, b = seconds(item["start"]), seconds(item["end"])
            if not start <= (a + b) / 2 < end or not normalized(item["text"]):
                continue
            if any(min(b, seconds(w["end"])) > max(a, seconds(w["start"])) for w in [*words, *secondary]):
                continue
            addition = {**item, "speaker": f"recovery{index}:{item.get('speaker')}"}
            words.append(addition)
            check["新增"].append(addition)
    return sorted(words, key=lambda w: seconds(w["start"])), checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--provider", choices=("grok", "doubao", "local"))
    parser.add_argument("--episode", help="集号，默认从本集目录名（如 E01）取得")
    parser.add_argument("--keyterm", action="append", help="Grok 人名提示词，可重复传入角色卡中的姓名")
    parser.add_argument("--keyterm-from", type=Path, action="append", help="只从角色卡姓名、别名提取提示词，可重复")
    parser.add_argument("--recover-from", type=Path, help="已有的人声段.json；不接受字幕。未提供时本地检测")
    parser.add_argument("--secondary-words", type=Path, help="已有副路原始词级响应；未提供时不触发补识别")
    parser.add_argument("--no-recover", action="store_true", help="不回查疑似漏句")
    parser.add_argument("--min-pause", type=float)
    parser.add_argument("--max-sentence", type=float)
    args = parser.parse_args(argv)
    episode = args.episode or args.episode_dir.name
    output = args.episode_dir / "声音记录.json"
    if output.exists():
        parser.error(f"输出已存在，不覆盖：{output}")
    try:
        records([], episode)
        episode = f"E{int(str(episode).lstrip('Ee')):02d}"
        settings = load_config()["asr"].get("segmentation", {})
        # 提示词先核对，传字幕时不能先调用识别服务再报错。
        terms = list(args.keyterm or [])
        for path in args.keyterm_from or []:
            terms.extend(keyterms_from(path))
        secondary = load_words(args.secondary_words) if args.secondary_words else None
        voices = json.loads(args.recover_from.read_text(encoding="utf-8")) if args.recover_from else None
        if voices is not None and (not isinstance(voices, list) or any(
                not isinstance(v, dict) or "起" not in v or "止" not in v or "文字" in v for v in voices)):
            raise ValueError("--recover-from 只接受人声段.json，不接受字幕")
        backend = get_backend(args.provider)
        if terms:
            if backend.provider != "grok":
                raise ValueError("--keyterm / --keyterm-from 仅用于 Grok 后端")
            backend.settings["keyterms"] = list(dict.fromkeys(list(backend.settings.get("keyterms", [])) + terms))
        material = args.episode_dir / "素材"
        material.mkdir(parents=True, exist_ok=True)
        backend.settings["raw_dir"] = str(material / "asr_raw")
        Path(backend.settings["raw_dir"]).mkdir(parents=True, exist_ok=True)
        checks = []
        with tempfile.TemporaryDirectory(prefix="audio-", dir=material) as tmp:
            wav = audio.extract_audio(args.video, Path(tmp) / "source.wav")
            voices = detect(wav, settings.get("vad", {})) if voices is None else voices
            backend.transcribe(wav)
            words = getattr(backend, "last_words", None)
            if words is None:
                raise ValueError("后端未返回词级时间，不能执行音频切句；原始响应保留待核对")
            if not args.no_recover and secondary is not None:
                words, checks = recover_missing(backend, wav, words, voices, episode, secondary=secondary)
            result, _ = segment_pair(words, [], voices, waveform(wav), episode,
                                     min_pause=args.min_pause if args.min_pause is not None else settings.get("min_pause", .5),
                                     max_sentence=args.max_sentence if args.max_sentence is not None else settings.get("max_sentence", 6))
            for row in result:
                row.pop("核对")
                if any(str(w.get("speaker", "")).startswith("recovery") and
                       seconds(row["起"]) <= (w["start"] + w["end"]) / 2 < seconds(row["止"]) for w in words):
                    row["听辨"] = "听不清"
        (material / "人声段.json").write_text(json.dumps(voices, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (material / "asr_raw/recovery.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with output.open("x", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        summary = {"文件": str(output), "句数": len(result), "后端": backend.provider,
                   "补识别窗口数": len(checks), "补识别失败窗口数": sum("错误" in c for c in checks)}
        if secondary is None and not args.no_recover:
            summary["未补识别原因"] = "缺少副路词级结果，不能确认两路均无词；未按字幕触发"
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except (ASRError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
