"""将人工回听决定写回声音记录与待核对。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .storage import read, save_board, write
from .snapshot import summary, field_changes

DECISIONS = ("主路", "副路", "字幕", "手改", "删除")


def apply_results(episode, decisions):
    episode = Path(episode)
    original = read(episode / "声音记录.json")
    if not isinstance(decisions, dict) or not isinstance(decisions.get("结果"), list):
        raise ValueError("回听结果须包含声音快照、结果数组")
    snapshot = decisions.get("声音快照")
    if not isinstance(snapshot, list) or any(not isinstance(r, dict) or not all(k in r for k in ("id", "文本", "起", "止")) for r in snapshot):
        raise ValueError("回听结果缺少有效声音快照")
    if summary(snapshot) != summary(original) or ("声音摘要" in decisions and decisions["声音摘要"] != summary(snapshot)):
        raise ValueError("声音记录已变化，请重新生成回听页，避免用旧句号覆盖新内容")
    changes = field_changes(snapshot, original)
    snapshot_by_id = {r["id"]: r for r in snapshot}
    sounds = deepcopy(original)
    pending = read(episode / "待核对.json") if (episode / "待核对.json").exists() else []
    by_id = {s["id"]: s for s in sounds}
    selected = {}
    for decision in decisions["结果"]:
        if not isinstance(decision, dict):
            raise ValueError("每条回听决定须为对象")
        ident = decision.get("句号")
        if ident not in by_id or ident in selected:
            raise ValueError(f"未知或重复句号：{ident}")
        source = decision.get("决定")
        if source not in DECISIONS or decision.get("听辨") not in ("清楚", "重叠", "听不清"):
            raise ValueError(f"决定或听辨无效：{ident}")
        text, note = decision.get("最终文本"), decision.get("备注")
        if not isinstance(text, str) or not isinstance(note, str) or source != "删除" and not text.strip():
            raise ValueError(f"最终文本或备注无效：{ident}")
        row = by_id[ident]
        check = row.setdefault("核对", {"状态": "人工确认", "主路": row["文本"], "副路": "", "字幕": "", "优先级": 2})
        # 采用来源与生成页时的原文比较；后续声线、听辨、核对补充不使页面失效。
        reference = snapshot_by_id[ident].get("核对", {})
        source_text = decisions.get("字幕参考", {}).get(ident, reference.get(source)) if source == "字幕" else reference.get(source)
        if source in ("主路", "副路", "字幕") and (not source_text or text != source_text):
            raise ValueError(f"选定来源与最终文本不一致，请用手改：{ident}")
        selected[ident] = decision
        if source != "删除":
            row.update({"文本": text, "听辨": decision["听辨"]})
            check.update({"状态": "人工确认", "决定来源": source, "备注": note})
    mapping, kept = {}, []
    for row in sounds:
        old = row["id"]
        if old in selected and selected[old]["决定"] == "删除":
            mapping[old] = None
            continue
        row["id"] = f'{old.split("-S")[0]}-S{len(kept) + 1:03d}'
        mapping[old] = row["id"]
        kept.append(row)
    for issue in pending:
        old = issue["位置"].get("句")
        if old in selected and issue["来源"] == "识别" and issue["处理职责"] == "回听":
            decision = selected[old]
            issue.update({"状态": "已裁定", "复核结果":
                          f'原句号：{old}；决定：{decision["决定"]}；最终文本：{decision["最终文本"]}；听辨：{decision["听辨"]}；备注：{decision["备注"]}'})
        if old in mapping:
            if mapping[old] is None:
                del issue["位置"]["句"]
                if old not in selected or not (issue["来源"] == "识别" and issue["处理职责"] == "回听"):
                    issue["依据"] += f"\n原句 {old} 经人工删除；请核对本问题是否仍成立。"
            else:
                issue["位置"]["句"] = mapping[old]
    saved = save_board(episode, kept, pending, "应用")
    target = episode / "素材" / "回听" / "句号对照.json"
    if target.exists():
        read(target)
        # 同名对照始终对应最近一次应用；旧对照随本轮黑板备份保留。
        if saved["备份"]:
            write(Path(saved["备份"]) / target.name, read(target))
    write(target, mapping)
    return {"处理句数": len(selected), "删除句数": sum(v is None for v in mapping.values()),
            "剩余句数": len(kept), "句号对照": str(target), "非摘要字段变化": changes, **saved}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("本集目录", type=Path)
    parser.add_argument("回听结果", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(apply_results(args.本集目录, read(args.回听结果)), ensure_ascii=False))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"应用回听结果失败：{exc}\n")
    return 0
