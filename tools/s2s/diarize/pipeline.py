"""按声音记录重标声线；文本和时间保持不变。"""

from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import re
import resource
import sys
import time
from uuid import uuid4

import numpy as np

from s2s.timefmt import to_sec
from .clustering import assign
from .constraints import constrained_assign
from .embeddings import Embedder, MODEL_DIR, SAMPLE_RATE, decode_audio, separate_vocals, window_embedding


def validate_records(records):
    if not isinstance(records, list):
        raise ValueError("声音记录须为数组")
    seen, episode = set(), None
    for row in records:
        match = re.fullmatch(r"(E\d{2,})-S\d{3,}", row["id"])
        if not match or row["id"] in seen:
            raise ValueError("句编号格式错误或重复")
        if episode is not None and match[1] != episode:
            raise ValueError("一次只处理一集")
        episode = match[1]
        seen.add(row["id"])
        if not 0 <= to_sec(row["起"]) < to_sec(row["止"]):
            raise ValueError(f"句起止时间无效：{row['id']}")
    return episode


def load_words(paths):
    """接受 Grok 原始响应、项目保存的响应封套，或绝对时间词条数组。"""
    words = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        offset = 0
        if isinstance(data, dict):
            offset = data.get("window", [0])[0]
            data = data.get("response", data).get("words", [])
        for word in data:
            if not any(c.isalnum() for c in word.get("text", "")):
                continue
            words.append({**word, "start": float(word["start"]) + offset,
                          "end": float(word["end"]) + offset, "_scope": str(Path(path).resolve())})
    return sorted(words, key=lambda w: w["start"])


def word_switch(words, start, stop):
    inside = [w for w in words if w["start"] < stop and w["end"] > start
              and w.get("speaker") is not None]
    return any(a["speaker"] != b["speaker"] for a, b in zip(inside, inside[1:]))


def analyze(records, samples, embedder, threshold=0.6, overlap_threshold=0.35,
            window_seconds=3.0, words=(), constraints=None, lipsync=None, vectors_out=None):
    episode = validate_records(records)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("聚类相似度阈值须在 0 到 1 之间")
    if not math.isfinite(overlap_threshold) or not -1 <= overlap_threshold <= 1:
        raise ValueError("重叠相似度阈值须在 -1 到 1 之间")
    if not math.isfinite(window_seconds) or window_seconds < 0.5:
        raise ValueError("分窗长度须至少 0.5 秒")
    vectors, eligible, details = [], [], []
    for row in records:
        start, stop = to_sec(row["起"]), to_sec(row["止"])
        if stop > len(samples) / SAMPLE_RATE + 0.05:
            raise ValueError(f"句时间超出音频：{row['id']}")
        clip = samples[round(start * SAMPLE_RATE):round(stop * SAMPLE_RATE)]
        vector = window_embedding(clip, embedder, window_seconds)
        vectors.append(vector)
        eligible.append(stop - start >= 0.5 - 1e-8)
        reasons = []
        half_score = None
        if len(clip) >= SAMPLE_RATE:
            mid = len(clip) // 2
            left = window_embedding(clip[:mid], embedder, window_seconds)
            right = window_embedding(clip[mid:], embedder, window_seconds)
            if left is not None and right is not None:
                half_score = float(np.clip(left @ right, -1, 1))
                if half_score < overlap_threshold:
                    reasons.append("前后半句声纹差异")
        if word_switch(words, start, stop):
            reasons.append("词级声线跳变")
        if row.get("听辨") == "重叠":
            reasons.append("原记录重叠")
        details.append((reasons, half_score))
    if vectors_out is not None:
        vectors_out.extend(vectors)
    return classify(records, vectors, eligible, details, threshold, words, constraints, lipsync)


def classify(records, vectors, eligible, details, threshold=0.6, words=(), constraints=None, lipsync=None):
    """对已有向量分类，便于同一输入比较约束开关，不重复模型推理。"""
    episode = validate_records(records)
    reasons_by_row, metadata = {}, None
    if constraints is None:
        labels, scores, matrix = assign(vectors, eligible, threshold)
    else:
        labels, scores, matrix, reasons_by_row, metadata = constrained_assign(
            vectors, eligible, threshold, records, words, lipsync, constraints)
        eligible = [ok or "lipsync" in reasons_by_row.get(i, ()) for i, ok in enumerate(eligible)]
    result = []
    for i, row in enumerate(records):
        reasons, half_score = details[i]
        label = labels.get(i)
        result.append({
            "句": row["id"], "起": row["起"], "止": row["止"], "文本": row["文本"], "声线": f"{episode}-V{label + 1}" if label is not None else "待定",
            "相似度": round(scores[i], 6) if scores[i] is not None else None,
            "待定": label is None, "疑似重叠": bool(reasons), "重叠依据": reasons,
            "前后半句相似度": round(half_score, 6) if half_score is not None else None,
            "参与聚类": eligible[i] and vectors[i] is not None,
            "置信度": round(max(0.0, scores[i]), 6) if label is not None else None,
            "合并依据": sorted(reasons_by_row.get(i, {"voiceprint"})) if label is not None else [],
        })
    classes = [{"声线": f"{episode}-V{label + 1}",
                "句数": sum(v == label for v in labels.values()),
                "聚类句数": sum(v == label and eligible[i] for i, v in labels.items())}
               for label in range(len(matrix))]
    for category in classes:
        candidates = [(r, source) for r, source in zip(result, records) if r["声线"] == category["声线"]]
        candidates.sort(key=lambda pair: (not pair[0]["疑似重叠"], pair[0]["相似度"],
                                         to_sec(pair[1]["止"]) - to_sec(pair[1]["起"])), reverse=True)
        category["代表句"] = [{"句": r["句"], "文本": source["文本"], "起": source["起"], "止": source["止"]}
                            for r, source in candidates[:3]]
    output = {"句列表": result, "类别": classes,
              "类间相似度矩阵": [[round(x, 6) for x in row] for row in matrix],
              "矩阵声线顺序": [row["声线"] for row in classes],
              "置信度说明": "类中心余弦相似度截断至0–1，未作概率校准；单句类接近1不代表身份可靠"}
    if metadata is not None:
        output["约束"] = metadata
    return output


def run(source, records, model="campp", denoise=False, threshold=0.6,
        overlap_threshold=0.35, window_seconds=3.0, words=(), model_dir=MODEL_DIR, threads=2,
        constraints=None, lipsync=None, vectors_out=None):
    validate_records(records)
    start = time.monotonic()
    samples = (separate_vocals(source, model_dir, threads) if denoise else
               decode_audio(source)[:, 0])
    embedder = Embedder(model, model_dir, threads)
    result = analyze(records, samples, embedder, threshold, overlap_threshold, window_seconds, words,
                     constraints, lipsync, vectors_out)
    divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    result["运行"] = {
        "模型": model, "去配乐": denoise, "阈值": threshold, "重叠阈值": overlap_threshold,
        "分窗秒": window_seconds, "线程": threads, "耗时秒": round(time.monotonic() - start, 3),
        "进程内存峰值MiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor, 2),
        "子进程内存峰值MiB": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / divisor, 2),
    }
    return result


def write_back(path, original, result):
    path = Path(path)
    raw = path.read_bytes()
    if json.loads(raw) != original:
        raise ValueError("写回目标与分析输入不同，停止写回")
    rows = result["句列表"]
    if [r["句"] for r in rows] != [r["id"] for r in original]:
        raise ValueError("写回结果句编号与输入不一致")
    updated, pending = [], []
    for old, new in zip(original, rows):
        row = dict(old)
        if not new["待定"]:
            row["声线"] = new["声线"]
        else:
            # 当前黑板格式只接受 E*-V*，未定声线留给结果侧表，不造身份。
            pending.append(row["id"])
        if new["疑似重叠"]:
            row["听辨"] = "重叠"
        updated.append(row)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%f+0800")
    backup = path.with_name(path.name + f".bak.{stamp}")
    with backup.open("xb") as file:
        file.write(raw)
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as file:
            json.dump(updated, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"备份": str(backup), "保留原声线的待定句": pending}
