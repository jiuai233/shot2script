"""按镜头数及动作识别送图帧预算分段，并保留邻段边界镜头引用。"""

import argparse
import json
from pathlib import Path

from .timefmt import to_sec


def per_second(shot: dict) -> list[dict]:
    """动作识别送图：整集时间轴每秒取一点（n+0.5 秒），每点挑本镜头最近的一帧；没有取点的短镜头只留中间帧。"""
    frames = shot.get("帧", [])
    start, end = to_sec(shot["起"]), to_sec(shot["止"])
    picked = []
    for n in range(int(start), int(end) + 1):
        if start <= n + 0.5 < end:
            best = min(frames, key=lambda f: abs(to_sec(f["时间"]) - n - 0.5), default=None)
            if best is not None and best not in picked:
                picked.append(best)
    middle = len(frames) // 2
    return sorted(picked, key=lambda f: to_sec(f["时间"])) or frames[middle:middle + 1]


def partition_sizes(count: int, target: int = 17, soft_limit: int = 20) -> list[int]:
    if not isinstance(count, int) or count < 0:
        raise ValueError("镜头数须为非负整数")
    if not isinstance(target, int) or not isinstance(soft_limit, int) or not 1 <= target <= soft_limit:
        raise ValueError("镜头目标数须为正整数且不大于软上限")
    sizes = [target] * (count // target)
    tail = count % target
    if tail:
        if sizes and sizes[-1] + tail <= soft_limit:
            sizes[-1] += tail
        else:
            sizes.append(tail)
    return sizes


def build_segments(table: dict, *, target: int = 17, soft_limit: int = 20, frame_budget: int = 40, context_budget: int = 48) -> dict:
    partition_sizes(0, target, soft_limit)
    if any(not isinstance(n, int) or n < 1 for n in (frame_budget, context_budget)):
        raise ValueError("帧预算须为正整数")
    shots = table["镜头"]
    episode = table["集"]
    seen = set()
    for index, shot in enumerate(shots):
        if shot["id"] in seen or not shot["id"].startswith(f"E{episode:02d}-K"):
            raise ValueError("镜头编号重复或不属于本集")
        seen.add(shot["id"])
        if to_sec(shot["止"]) <= to_sec(shot["起"]):
            raise ValueError("镜头结束时间须晚于开始时间")
        if index and shots[index - 1]["止"] != shot["起"]:
            raise ValueError("镜头必须按时间首尾相接")
    result = {"集": episode, "片段": []}
    offset = 0
    index = 0
    while offset < len(shots):
        index += 1
        size = partition_sizes(len(shots) - offset, target, soft_limit)[0]
        # 预算按动作识别实际送出的帧计算。
        def fits(n):
            own = sum(len(per_second(shot)) for shot in shots[offset:offset + n])
            context = own + (len(per_second(shots[offset - 1])) if offset else 0)
            if offset + n < len(shots):
                context += len(per_second(shots[offset + n]))
            return own <= frame_budget and context <= context_budget
        while size > 1 and not fits(size):
            size -= 1
        if not fits(size):
            raise ValueError('单镜头及衔接帧已超预算，无法在镜头边界继续缩小：' + shots[offset]['id'])
        group = shots[offset:offset + size]
        result["片段"].append({
            "id": f"E{episode:02d}-P{index:02d}",
            "镜头": [shot["id"] for shot in group],
            "起": group[0]["起"],
            "止": group[-1]["止"],
            "衔接前": shots[offset - 1]["id"] if offset else None,
            "衔接后": shots[offset + size]["id"] if offset + size < len(shots) else None,
        })
        offset += size
    return result


def write_segments(shot_table: str | Path, **kwargs) -> dict:
    path = Path(shot_table)
    result = build_segments(json.loads(path.read_text(encoding="utf-8")), **kwargs)
    path.with_name("片段表.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="默认每段 17 镜头、软上限 20，按送图帧预算缩小片段")
    parser.add_argument("shot_table")
    parser.add_argument("--shots-per-segment", type=int, default=17)
    parser.add_argument("--soft-limit", type=int, default=20)
    parser.add_argument("--frame-budget", type=int, default=40)
    parser.add_argument("--context-frame-budget", type=int, default=48)
    args = parser.parse_args(argv)
    try:
        result = write_segments(args.shot_table, target=args.shots_per_segment, soft_limit=args.soft_limit, frame_budget=args.frame_budget, context_budget=args.context_frame_budget)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(1, f"片段划分失败：{exc}\n")
    print(json.dumps(result, ensure_ascii=False))
