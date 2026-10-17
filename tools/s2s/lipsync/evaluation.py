"""口型的画内性与说话人候选评测，不猜测画外/心声/设备细类。"""

from ..timefmt import to_sec


def evaluate(record, annotation, group_to_speaker=None):
    if not annotation.get("校对完成"):
        raise ValueError("人工标注尚未校对完成，不能产出准确率")
    mapping = group_to_speaker or {}
    predictions = record["句"]
    truths = annotation.get("台词", [])
    # 标注 Lxxx 与声音记录 Sxxx 不一定同一套编号，按一对一时间重叠对齐。
    candidates = []
    for i, truth in enumerate(truths):
        for j, pred in enumerate(predictions):
            a, b = to_sec(truth["起"]), to_sec(truth["止"])
            c, d = to_sec(pred["起"]), to_sec(pred["止"])
            overlap = max(0, min(b, d) - max(a, c))
            union = max(b, d) - min(a, c)
            iou = overlap / union if union else 0
            same_id = truth.get("id") == pred["句"]
            if same_id or iou >= .5:
                candidates.append((2 if same_id else iou, i, j))
    matches, used = {}, set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in matches and j not in used:
            matches[i] = predictions[j]
            used.add(j)
    counts = {"形态标注句数": 0, "画内性正确句数": 0, "画内性已判断句数": 0,
              "说话人标注句数": 0, "说话人正确句数": 0, "说话人已判断句数": 0}
    errors = []
    for i, truth in enumerate(truths):
        pred = matches.get(i)
        form = truth.get("形态")
        expected = "画内" if form == "画内" else "未开口" if form in ("画外", "心声", "设备") else None
        if expected:
            counts["形态标注句数"] += 1
            actual = pred["结论"] if pred else None
            counts["画内性已判断句数"] += actual in ("画内", "未开口")
            counts["画内性正确句数"] += actual == expected
            if actual != expected:
                errors.append({"id": truth["id"], "项目": "画内性", "期望": expected, "实际": actual})
        if truth.get("说话人"):
            counts["说话人标注句数"] += 1
            opened = [t for t in pred["轨迹"] if t["状态"] == "开口"] if pred else []
            actual_people = {mapping[t["轨迹组"]] for t in opened if t.get("轨迹组") in mapping}
            actual = next(iter(actual_people)) if len(actual_people) == 1 else None
            counts["说话人已判断句数"] += actual is not None
            counts["说话人正确句数"] += actual == truth["说话人"]
            if actual != truth["说话人"]:
                errors.append({"id": truth["id"], "项目": "说话人", "期望": truth["说话人"], "实际": actual})
    def ratio(n, d):
        return counts[n] / counts[d] if counts[d] else None
    return {**counts, "画内性准确率": ratio("画内性正确句数", "形态标注句数"),
            "画内性覆盖率": ratio("画内性已判断句数", "形态标注句数"),
            "说话人准确率": ratio("说话人正确句数", "说话人标注句数"),
            "说话人覆盖率": ratio("说话人已判断句数", "说话人标注句数"),
            "错误": errors, "说明": "未开口只对应非画内候选，不区分画外、心声、设备；弃权和未匹配计入总分母。说话人需人工确认轨迹组映射；不凭声线循环证明身份。"}
