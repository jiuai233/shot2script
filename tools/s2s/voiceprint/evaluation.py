"""人工标注与回听决定到位后的声纹候选评测。"""

from collections import Counter, defaultdict

from s2s.sub_align import comparison_normalize
from s2s.timefmt import to_sec


def evaluate(result, annotation, voice_to_speaker=None):
    if not isinstance(annotation, dict) or annotation.get("校对完成") is not True:
        raise ValueError("人工标注尚未校对完成，不能产出准确率")
    truths = annotation.get("台词", [])
    if any(not t.get("说话人") for t in truths):
        raise ValueError("人工标注存在空说话人")
    predictions = result["句列表"]
    candidates = []
    for i, truth in enumerate(truths):
        for j, pred in enumerate(predictions):
            same_id = truth.get("id") == pred["句"]
            iou = 0
            if all("起" in r and "止" in r for r in (truth, pred)):
                a, b, c, d = to_sec(truth["起"]), to_sec(truth["止"]), to_sec(pred["起"]), to_sec(pred["止"])
                iou = max(0, min(b, d) - max(a, c)) / max(1e-8, max(b, d) - min(a, c))
            if same_id or iou >= 0.5:
                candidates.append((2 if same_id else iou, i, j))
    matches, used = {}, set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in matches and j not in used:
            matches[i] = predictions[j]
            used.add(j)
    groups, splits = defaultdict(list), defaultdict(set)
    covered, correct, errors = 0, 0, []
    mapping = voice_to_speaker or {}
    for i, truth in enumerate(truths):
        pred = matches.get(i)
        label = pred["声线"] if pred and not pred["待定"] else None
        if label is not None:
            covered += 1
            groups[label].append(truth["说话人"])
            splits[truth["说话人"]].add(label)
        actual = mapping.get(label)
        correct += actual == truth["说话人"]
        if actual != truth["说话人"]:
            errors.append({"id": truth["id"], "期望": truth["说话人"], "声线": label, "实际人物": actual})
    majority = sum(Counter(people).most_common(1)[0][1] for people in groups.values())
    n = len(truths)
    return {"标注句数": n, "已分配句数": covered, "覆盖率": covered / n if n else None,
            "已分配句纯度": majority / covered if covered else None,
            "全量多数类覆盖率": majority / n if n else None,
            "各类句数": {k: len(v) for k, v in groups.items()},
            "说话人拆分类数": {speaker: len(splits[speaker]) for speaker in sorted({t["说话人"] for t in truths})},
            "说话人正确句数": correct if voice_to_speaker is not None else None,
            "说话人准确率": correct / n if n and voice_to_speaker is not None else None,
            "未匹配预测句数": len(predictions) - len(used),
            "错误": errors if voice_to_speaker is not None else [],
            "说明": "按句号或IoU≥0.5一对一匹配；断句不同需核对匹配。未匹配、待定计入总分母；纯度不能代替准确率。"}


def evaluate_listen(records, listen):
    if listen.get("声音快照") != records:
        raise ValueError("回听声音快照与评测输入不一致")
    lookup = {r["id"]: r for r in records}
    decisions = listen.get("结果", [])
    seen, correct, errors, deleted = set(), 0, [], 0
    for decision in decisions:
        key = decision["句号"]
        if key not in lookup or key in seen:
            raise ValueError("回听结果含未知或重复句号")
        seen.add(key)
        if decision["决定"] == "删除":
            deleted += 1
            continue
        equal = comparison_normalize(lookup[key]["文本"]) == comparison_normalize(decision["最终文本"])
        correct += equal
        if not equal:
            errors.append(key)
    total = len(decisions) - deleted
    return {"已裁定句数": len(decisions), "删除句数": deleted, "非删除裁定句数": total,
            "原文本一致句数": correct, "已裁定非删除句一致率": correct / total if total else None,
            "裁定覆盖率": len(decisions) / len(records) if records else None, "文字变化句": errors,
            "说明": "只评测已人工裁定的文字，不据回听文字决定推断说话人真值。"}
