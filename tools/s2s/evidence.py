"""按当前声音记录核对口型、声纹及跨集候选的有效范围。"""


def sound_snapshot(sounds):
    return [{k: row[k] for k in ("id", "起", "止", "文本", "声线")} for row in sounds]


def evidence_problem(kind, evidence, sounds):
    if kind == "声纹候选":
        if evidence.get("声音摘要") != sound_snapshot(sounds):
            return "声音摘要缺失或已过期，须重新比对"
        return None
    key = "句" if kind == "口型记录" else "句列表"
    rows = evidence.get(key, [])
    if [row.get("句") for row in rows] != [row["id"] for row in sounds]:
        return "句号缺失或已变化，须重新生成"
    for row, sound in zip(rows, sounds):
        if any(row.get(k) != sound[k] for k in ("起", "止", "文本")):
            return "文本或起止已变化，须重新生成"
        if kind == "声纹聚类" and not row.get("待定") and row.get("声线") != sound["声线"]:
            return "声线已变化，须重新聚类"
    return None
