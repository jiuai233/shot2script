"""按时间重叠迁移人工裁定；句界变化只展示原裁定，不自动套字。"""

from copy import deepcopy

from ..asr.postprocess import seconds


def migrate(sounds, decisions):
    old = decisions["声音快照"]
    by_id = {r["id"]: r for r in old}
    resolved = {}
    for decision in decisions["结果"]:
        ident = decision["句号"]
        if ident not in by_id or ident in resolved:
            raise ValueError("原裁定包含未知或重复句号")
        if decision["决定"] != "删除" and not decision["最终文本"].strip():
            raise ValueError("原裁定的最终文本为空")
        resolved[ident] = decision
    intersections = {}
    for row in sounds:
        intersections[row["id"]] = [r for r in old if min(seconds(row["止"]), seconds(r["止"])) >
                                      max(seconds(row["起"]), seconds(r["起"]))]
    output, migrations, mapped = deepcopy(sounds), [], set()
    for row in output:
        overlaps = intersections[row["id"]]
        decided = [r for r in overlaps if r["id"] in resolved]
        if not decided:
            continue
        mapped.update(r["id"] for r in decided)
        same = (len(overlaps) == 1 and len(decided) == 1 and
                row["文本"] == decided[0]["文本"] and
                not ("原始文本" in row.get("核对", {}) and resolved[decided[0]["id"]]["决定"] in ("主路", "副路")) and
                row["起"] == decided[0]["起"] and row["止"] == decided[0]["止"] and
                sum(decided[0] in values for values in intersections.values()) == 1 and
                resolved[decided[0]["id"]]["决定"] != "删除")
        if same:
            choice = resolved[decided[0]["id"]]
            row.update({"文本": choice["最终文本"], "听辨": choice["听辨"]})
            row["核对"].update({"状态": "人工确认", "决定来源": choice["决定"], "备注": choice["备注"]})
        migrations.append({"新句号": row["id"], "旧句号": [r["id"] for r in decided],
                           "状态": "已沿用" if same else "待确认",
                           "原裁定": "\n".join(f'{r["id"]}（{r["起"]}—{r["止"]}）：' +
                                               (resolved[r["id"]]["最终文本"] if resolved[r["id"]]["决定"] != "删除" else "删除该句")
                                               for r in decided)})
    for ident in resolved.keys() - mapped:
        migrations.append({"新句号": None, "旧句号": [ident], "状态": "未匹配", "原裁定": resolved[ident]["最终文本"]})
    return output, migrations
