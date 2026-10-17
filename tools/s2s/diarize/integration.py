"""交叉核对的声纹步骤与试听导出。"""

import json
from pathlib import Path
import subprocess

from s2s.audio import FFMPEG
from .pipeline import load_words, run


def discover_words(directories):
    """同窗口只取最近一份成功的 Grok 响应，不混合历史重试标签。"""
    chosen = {}
    for directory in directories:
        for path in sorted(Path(directory).rglob("*.json")) if Path(directory).exists() else []:
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or data.get("provider", "grok") != "grok" or data.get("status", 200) != 200:
                continue
            if not isinstance(data.get("response", data).get("words"), list):
                continue
            window = tuple(data.get("window", [0, None]))
            stamp = (data.get("time", ""), path.name)
            if window not in chosen or stamp > chosen[window][0]:
                chosen[window] = (stamp, path)
    covered, words = [], []
    # 定向补识别响应的标签与整集响应不共用命名空间，重叠区优先整集标签。
    for window, (_, path) in sorted(chosen.items(), key=lambda item: (item[0][1] or float("inf")) - item[0][0], reverse=True):
        current = load_words([path])
        words.extend(word for word in current if not any(a <= (word["start"] + word["end"]) / 2 < b for a, b in covered))
        covered.append((window[0], window[1] or float("inf")))
    return sorted(words, key=lambda word: word["start"])


def apply_labels(sounds, result):
    if [r["id"] for r in sounds] != [r["句"] for r in result["句列表"]]:
        raise ValueError("声纹结果与当前声音记录句号不一致")
    # 交叉核对只新增声线步骤，不改变台词、时间、听辨及回听状态。
    return [{**old, "声线": new["声线"]} if not new["待定"] else dict(old)
            for old, new in zip(sounds, result["句列表"])]


def analyze_board(video, sounds, episode_dir, settings, raw_dirs=()):
    episode_dir = Path(episode_dir)
    constraints = {k: settings[k] for k in ("constraints", "grok_weight", "grok_purity", "grok_min_similarity", "max_speakers") if k in settings}
    roles_path = episode_dir.parent / "角色卡.json"
    if constraints.get("max_speakers") is None and roles_path.exists():
        roles = json.loads(roles_path.read_text())
        if roles:
            constraints["max_speakers"] = len({r["id"] for r in roles})
    lips_path = episode_dir / "口型记录.json"
    lips = json.loads(lips_path.read_text()) if lips_path.exists() else None
    words = discover_words([episode_dir / "素材/asr_raw", *raw_dirs])
    result = run(video, sounds, model=settings.get("model", "campp"),
                 threshold=settings.get("threshold", 0.6), constraints=constraints,
                 lipsync=lips, words=words, threads=settings.get("threads", 2))
    result["输入可用性"] = {"Grok词数": len(words), "口型记录": lips is not None,
                           "人数上限": constraints.get("max_speakers")}
    return apply_labels(sounds, result), result


def export_representatives(source, result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for category in result["类别"]:
        for row in category["代表句"]:
            name = f'{category["声线"]}_{row["句"]}.wav'
            destination = output_dir / name
            from s2s.timefmt import to_sec
            start, stop = to_sec(row["起"]), to_sec(row["止"])
            completed = subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-n", "-i", str(source),
                                        "-ss", str(start), "-t", str(stop - start), "-vn",
                                        "-ar", "16000", "-ac", "1", str(destination)],
                                       capture_output=True, timeout=30)
            if completed.returncode:
                raise ValueError(f"代表句导出失败：{row['句']}")
            index.append({"声线": category["声线"], **row, "文件": name})
    with (output_dir / "试听索引.json").open("x") as file:
        json.dump(index, file, ensure_ascii=False, indent=2)
    return index
