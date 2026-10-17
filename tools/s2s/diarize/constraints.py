"""带口型硬约束和 Grok 软证据的平均连接聚类。"""

from collections import defaultdict
import math

import numpy as np

from s2s.timefmt import to_sec
from .embeddings import unit


DEFAULTS = {"constraints": ["grok", "lipsync", "max_speakers"],
            "grok_weight": 0.15, "grok_purity": 0.7, "grok_min_similarity": 0.2,
            "max_speakers": None}


def options(settings):
    result = {**DEFAULTS, **(settings or {})}
    if not isinstance(result["constraints"], list) or set(result["constraints"]) - set(DEFAULTS["constraints"]):
        raise ValueError("未知约束；只支持 grok、lipsync、max_speakers")
    for key in ("grok_weight", "grok_purity", "grok_min_similarity"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} 须在 0 到 1 之间")
    limit = result["max_speakers"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("max_speakers 须为正整数或 null")
    return result


def evidence(records, words, lipsync, settings):
    """Grok 标签只在同一响应窗口内有效；口型须匹配当前句文本和时间。"""
    settings = options(settings)
    enabled = settings["constraints"]
    grok, tracks, conflicts = {}, defaultdict(set), []
    for i, row in enumerate(records):
        if "grok" not in enabled:
            break
        weights = defaultdict(float)
        start, stop = to_sec(row["起"]), to_sec(row["止"])
        for word in words:
            duration = min(stop, word["end"]) - max(start, word["start"])
            if duration > 0 and word.get("speaker") is not None:
                weights[(word.get("_scope", "default"), str(word["speaker"]))] += duration
        if weights:
            label = max(weights, key=weights.get)
            if weights[label] / sum(weights.values()) >= settings["grok_purity"]:
                grok[i] = label
    lookup = {r["id"]: (i, r) for i, r in enumerate(records)}
    if "lipsync" in enabled and lipsync:
        for row in lipsync.get("句", []):
            match = lookup.get(row.get("句"))
            if match is None:
                conflicts.append({"类型": "口型输入不匹配", "句": [row.get("句")]})
                continue
            i, sound = match
            if any(row.get(key) != sound.get(key) for key in ("起", "止", "文本")):
                conflicts.append({"类型": "口型输入不匹配", "句": [sound["id"]]})
                continue
            if row.get("结论") != "画内":
                continue
            active = {t["轨迹组"] for t in row.get("轨迹", [])
                      if t.get("状态") == "开口" and t.get("轨迹组")}
            if len(active) > 1:
                conflicts.append({"类型": "单句多轨迹开口", "句": [sound["id"]], "轨迹组": sorted(active)})
                continue
            for group in active:
                tracks[group].add(i)
    must, cannot = [], set()
    for group, indices in tracks.items():
        ordered = sorted(indices)
        must.extend((ordered[0], j, group) for j in ordered[1:])
    track_items = list(tracks.items())
    for k, (_, left) in enumerate(track_items):
        for _, right in track_items[k + 1:]:
            for i in left:
                for j in right:
                    if min(to_sec(records[i]["止"]), to_sec(records[j]["止"])) > max(to_sec(records[i]["起"]), to_sec(records[j]["起"])):
                        cannot.add(tuple(sorted((i, j))))
    return grok, must, cannot, conflicts


def constrained_assign(vectors, eligible, threshold, records, words=(), lipsync=None, settings=None):
    settings = options(settings)
    grok, must, cannot, conflicts = evidence(records, words, lipsync, settings)
    valid = {i for i, v in enumerate(vectors) if v is not None}
    # 同轨迹口型允许短句建立共同类，不能凭口型为静音伪造声纹。
    seeds = {i for i in valid if eligible[i]}
    for a, b, _ in must:
        if a in valid and b in valid:
            seeds.update((a, b))
    groups = [[i] for i in sorted(seeds)]
    reasons = defaultdict(set)
    matrix = np.zeros((len(vectors), len(vectors)))
    if valid:
        indices = sorted(valid)
        matrix[np.ix_(indices, indices)] = np.clip(np.asarray([vectors[i] for i in indices]) @ np.asarray([vectors[i] for i in indices]).T, -1, 1)

    def blocked(a, b):
        return any(tuple(sorted((i, j))) in cannot for i in a for j in b)

    def join(a, b, why):
        members = groups[a] + groups[b]
        inherited = set(why).union(*(reasons[i] for i in members))
        for i in members:
            reasons[i].update(inherited)
        groups[a] = sorted(members)
        groups.pop(b)

    for a, b, track in must:
        left = next((k for k, g in enumerate(groups) if a in g), None)
        right = next((k for k, g in enumerate(groups) if b in g), None)
        if left is None or right is None or left == right:
            continue
        if blocked(groups[left], groups[right]):
            conflicts.append({"类型": "must-link/cannot-link", "句": [records[a]["id"], records[b]["id"]], "轨迹组": track})
        else:
            join(min(left, right), max(left, right), ["lipsync"])
    limit = settings["max_speakers"] if "max_speakers" in settings["constraints"] else None
    cap = limit + 1 if limit is not None else None
    while len(groups) > 1:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if blocked(groups[a], groups[b]):
                    continue
                acoustic = float(matrix[np.ix_(groups[a], groups[b])].mean())
                same = sum(i in grok and j in grok and grok[i] == grok[j] for i in groups[a] for j in groups[b])
                boost = settings["grok_weight"] * same / (len(groups[a]) * len(groups[b])) if acoustic >= settings["grok_min_similarity"] else 0
                candidate = (acoustic + boost, -a, -b, acoustic, boost)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            if cap and len(groups) > cap:
                conflicts.append({"类型": "人数上限与cannot-link冲突", "目标类数": cap, "实际类数": len(groups)})
            break
        score, na, nb, acoustic, boost = best
        forced = score < threshold and cap is not None and len(groups) > cap
        if score < threshold and not forced:
            break
        why = ["max_speakers"] if forced else ["voiceprint"]
        if boost and score >= threshold and acoustic < threshold:
            why.append("grok")
        join(-na, -nb, why)
    groups.sort(key=lambda g: g[0])
    centers = [unit(np.mean([vectors[i] for i in g], axis=0)) for g in groups]
    # 零中心可由方向相反的句子被人数上限强并产生；保留可计算的代表中心并登记。
    for k, center in enumerate(centers):
        if center is None:
            centers[k] = vectors[groups[k][0]]
            conflicts.append({"类型": "零中心", "句": [records[i]["id"] for i in groups[k]]})
    labels = {i: k for k, g in enumerate(groups) for i in g}
    scores = [None] * len(vectors)
    for i in sorted(valid):
        if not centers:
            continue
        similarities = np.clip(np.asarray(centers) @ vectors[i], -1, 1)
        if i not in labels:
            candidates = [k for k, g in enumerate(groups) if not blocked([i], g)]
            if candidates:
                k = max(candidates, key=lambda k: similarities[k])
                if similarities[k] > threshold:
                    labels[i] = k
                    groups[k].append(i)
                    reasons[i].add("voiceprint")
        scores[i] = float(similarities[labels[i]]) if i in labels else float(max(similarities))
    # 吸附短句后重算中心，统计、代表句、跨集比对使用相同的最终成员。
    centers = [unit(np.mean([vectors[i] for i in g], axis=0)) for g in groups]
    centers = [c if c is not None else vectors[g[0]] for c, g in zip(centers, groups)]
    for i, k in labels.items():
        scores[i] = float(np.clip(vectors[i] @ centers[k], -1, 1))
    inter = np.clip(np.asarray(centers) @ np.asarray(centers).T, -1, 1).tolist() if centers else []
    blocked_pairs = [[records[i]["id"], records[j]["id"]] for i, j in sorted(cannot)]
    metadata = {"参数": settings, "冲突": conflicts, "cannot-link": blocked_pairs,
                "must-link数": len(must), "Grok覆盖句数": len(grok)}
    return labels, scores, inter, reasons, metadata
