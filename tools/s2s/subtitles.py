"""批量识别硬字幕，保留原文、左上原点的归一化框和置信度。"""

import argparse
from collections import Counter
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
import subprocess
import time
import unicodedata

from rapidfuzz.distance import Levenshtein

from .timefmt import fmt as timestamp, to_sec as to_seconds

ROOT = Path(__file__).resolve().parents[2]


def normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


@dataclass
class Settings:
    similarity: float = 0.85
    position_tolerance: float = 0.06
    dialogue_top: float = 0.68
    dialogue_bottom: float = 0.90
    center_tolerance: float = 0.18
    dialogue_min_height: float = 0.018
    dialogue_max_height: float = 0.13
    line_max_height: float = 0.065
    short_text_length: int = 20
    card_max_duration: float = 6.0
    location_top: float = 0.25
    location_max_height: float = 0.065
    corner_margin: float = 0.18
    watermark_ratio: float = 0.80
    watermark_min_frames: int = 10
    min_confidence: float = 0.0
    keep_watermarks: bool = False


def ensure_vocr(binary: Path | None = None) -> Path:
    binary = Path(binary) if binary else ROOT / "tools/bin/vocr"
    source = ROOT / "tools/ocr/vocr.swift"
    if not binary.is_file() or binary.stat().st_mtime < source.stat().st_mtime:
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["swiftc", "-O", str(source), "-o", str(binary)],
                       check=True, capture_output=True, text=True, timeout=120)
    return binary


def recognize(paths: list[Path], batch_size: int = 32) -> list[dict]:
    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0")
    binary = ensure_vocr()
    output = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        result = subprocess.run([str(binary), *(str(p.resolve()) for p in batch)],
                                check=True, capture_output=True, text=True, timeout=180)
        rows = json.loads(result.stdout)
        if len(rows) != len(batch):
            raise ValueError("OCR 返回的图片数与输入不一致")
        for row in rows:
            if row.get("error"):
                raise ValueError(f"OCR 图片读取失败：{row['path']}：{row['error']}")
        output.extend(rows)
    return output


def load_frames(episode: Path, frames: Path | None, fps: float) -> list[tuple[Path, float]]:
    if fps <= 0:
        raise ValueError("fps 必须大于 0")
    explicit = frames is not None
    directory = frames if explicit else episode / "素材/frames"
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not paths:
        raise ValueError(f"帧目录没有图片：{directory}")
    table = episode / "镜头表.json"
    if not explicit and table.exists():
        data = json.loads(table.read_text(encoding="utf-8"))
        times = {Path(f["文件"]).name: to_seconds(f["时间"])
                 for shot in data["镜头"] for f in shot["帧"]}
        missing = [p.name for p in paths if p.name not in times]
        if missing:
            raise ValueError(f"帧未登记在镜头表中：{missing[0]}")
        return sorted(((p, times[p.name]) for p in paths), key=lambda pair: pair[1])
    if not explicit and any(re.fullmatch(r"K\d+_\d+", p.stem) for p in paths):
        raise ValueError("镜头帧需要镜头表中的真实时间，不能用 fps 推算")
    return [(p, index / fps) for index, p in enumerate(paths)]


def near(a: dict, b: dict, tolerance: float) -> bool:
    return (abs(a["x"] + a["width"] / 2 - b["x"] - b["width"] / 2) <= tolerance
            and abs(a["y"] - b["y"]) <= tolerance
            and abs(a["height"] - b["height"]) <= tolerance)


def dialogue_box(box: dict, s: Settings) -> bool:
    return (s.dialogue_top <= box["y"] <= s.dialogue_bottom
            and abs(box["x"] + box["width"] / 2 - 0.5) <= s.center_tolerance
            and s.dialogue_min_height <= box["height"] <= s.dialogue_max_height
            and box["y"] + box["height"] <= 0.98)


def combine_lines(results: list[dict], s: Settings) -> list[dict]:
    """同一帧中紧邻的两行居中台词合为一个框；其他文字不强行拼接。"""
    ordered = sorted(results, key=lambda r: (r["box"]["y"], r["box"]["x"]))
    output = []
    for row in ordered:
        row = {**row, "box": dict(row["box"])}
        if (output and dialogue_box(row["box"], s) and dialogue_box(output[-1]["box"], s)
                and row["box"]["height"] <= s.line_max_height
                and output[-1]["box"]["height"] <= s.line_max_height):
            previous = output[-1]
            a, b = previous["box"], row["box"]
            gap = b["y"] - a["y"] - a["height"]
            if ("\n" not in previous["text"] and -0.005 <= gap <= 0.025
                    and abs(a["x"] + a["width"] / 2 - b["x"] - b["width"] / 2) <= s.center_tolerance):
                left, right = min(a["x"], b["x"]), max(a["x"] + a["width"], b["x"] + b["width"])
                previous["text"] += "\n" + row["text"]
                previous["confidence"] = min(previous["confidence"], row["confidence"])
                previous["box"] = {"x": left, "y": a["y"], "width": right - left,
                                   "height": b["y"] + b["height"] - a["y"]}
                continue
        output.append(row)
    return output


def classify(text: str, box: dict, duration: float, s: Settings) -> str:
    if dialogue_box(box, s) and ("\n" in text or box["height"] <= s.line_max_height):
        return "台词字幕"
    short = len(normalize(text)) <= s.short_text_length
    if (short and box["height"] <= s.location_max_height and not re.search(r"\d", text)
            and (box["y"] < s.location_top or box["x"] + box["width"] < s.corner_margin
                 or box["x"] > 1 - s.corner_margin)):
        return "地点字幕"
    if (short and s.location_top <= box["y"] < s.dialogue_top
            and duration <= s.card_max_duration
            and re.search(r"总裁|董事|经理|医生|律师|妻子|丈夫|母亲|父亲|女儿|儿子|千金|少爷|助理|秘书", text)):
        return "人物字卡"
    return "画面文字"


def merge_records(frames: list[dict], settings: Settings | None = None) -> tuple[list[dict], dict]:
    s = settings or Settings()
    tracks = []
    active = []
    for index, frame in enumerate(frames):
        used = set()
        rows = combine_lines([r for r in frame["results"] if r["confidence"] >= s.min_confidence and normalize(r["text"])], s)
        for row in rows:
            text = normalize(row["text"])
            candidates = []
            for track in active:
                if id(track) in used or track["last_frame"] != index - 1:
                    continue
                ratio = Levenshtein.normalized_similarity(text, normalize(track["last"]["text"]))
                if ratio >= s.similarity and near(row["box"], track["last"]["box"], s.position_tolerance):
                    candidates.append((ratio, track))
            if candidates:
                track = max(candidates, key=lambda pair: pair[0])[1]
            else:
                track = {"start": frame["time"], "end": frame["time"], "rows": [], "frames": [], "last_frame": index}
                tracks.append(track)
            track["rows"].append(row)
            track["frames"].append(index)
            track.update(end=frame["time"], last_frame=index, last=row)
            used.add(id(track))
        active = [t for t in tracks if t["last_frame"] == index]

    # 同一固定位置的相同文字即使中途漏识别，也按全片出现帧比例识别水印。
    fixed = []
    for index, frame in enumerate(frames):
        for row in frame["results"]:
            group = next((g for g in fixed if g["text"] == normalize(row["text"])
                          and near(g["box"], row["box"], s.position_tolerance)), None)
            if group is None:
                group = {"text": normalize(row["text"]), "box": row["box"], "frames": set()}
                fixed.append(group)
            group["frames"].add(index)
    watermarks = []
    for group in fixed:
        b = group["box"]
        edge = b["y"] < s.location_top or b["y"] > s.dialogue_bottom or b["x"] + b["width"] < s.corner_margin or b["x"] > 1 - s.corner_margin
        if (not s.keep_watermarks and group["text"] and edge and not dialogue_box(b, s)
                and len(group["frames"]) >= s.watermark_min_frames
                and len(group["frames"]) / len(frames) >= s.watermark_ratio):
            watermarks.append(group)
    output, provenance = [], []
    filtered = 0
    for track in tracks:
        counts = Counter(r["text"] for r in track["rows"])
        best = max(track["rows"], key=lambda r: (counts[r["text"]], r["confidence"], len(r["text"])))
        if any(normalize(best["text"]) == g["text"] and near(best["box"], g["box"], s.position_tolerance) for g in watermarks):
            filtered += 1
            continue
        box = {k: round(v, 6) for k, v in best["box"].items()}
        output.append({"起": timestamp(track["start"]), "止": timestamp(track["end"]),
                       "文字": best["text"], "类型": classify(best["text"], box, track["end"] - track["start"], s),
                       "框": box, "置信度": round(sum(r["confidence"] for r in track["rows"]) / len(track["rows"]), 4)})
        representative = next(i for i in track["frames"] if any(r["text"] == best["text"] for r in frames[i]["results"])) if "\n" not in best["text"] else track["frames"][0]
        provenance.append({"帧": frames[representative]["path"], "时间": timestamp(frames[representative]["time"]), "出现帧数": len(track["frames"])})
    return output, {"阈值": asdict(s), "过滤条数": filtered,
                    "水印": [{"文字": g["text"], "框": g["box"], "出现帧数": len(g["frames"])} for g in watermarks],
                    "来源": provenance}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--frames", type=Path, help="显式指定等间隔帧，按文件名排序、从 0 秒用 fps 推算时间")
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audit", type=Path, help="另存 OCR 原始结果、帧来源及过滤统计")
    for name, field in Settings.__dataclass_fields__.items():
        option = "--" + name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(option, action="store_true", default=field.default)
        else:
            parser.add_argument(option, type=type(field.default), default=field.default)
    args = parser.parse_args(argv)
    try:
        s = Settings(**{name: getattr(args, name) for name in Settings.__dataclass_fields__})
        for name in ("similarity", "watermark_ratio", "min_confidence"):
            if not 0 <= getattr(s, name) <= 1:
                raise ValueError(f"{name} 必须在 0 到 1 之间")
        frames = load_frames(args.episode, args.frames, args.fps)
        start = time.monotonic()
        raw = recognize([p for p, _ in frames], args.batch_size)
        for row, (path, seconds) in zip(raw, frames):
            row.update(path=str(path), time=seconds)
        records, audit = merge_records(raw, s)
        audit.update(耗时秒=round(time.monotonic() - start, 3), 帧数=len(frames), 类型统计=dict(Counter(r["类型"] for r in records)))
        args.episode.mkdir(parents=True, exist_ok=True)
        target = args.episode / "字幕记录.json"
        target.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.audit:
            args.audit.parent.mkdir(parents=True, exist_ok=True)
            args.audit.write_text(json.dumps({**audit, "原始识别": raw}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"输出": str(target), **{k: audit[k] for k in ("耗时秒", "帧数", "类型统计", "过滤条数", "水印")}}, ensure_ascii=False))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"字幕识别失败：{exc}\n")
    return 0
