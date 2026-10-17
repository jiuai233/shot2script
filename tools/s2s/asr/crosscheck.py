"""按时间重叠对齐两路识别；字幕只作回听参考。"""

import argparse
from copy import deepcopy
from difflib import SequenceMatcher
import json
from pathlib import Path
import tempfile
import subprocess

from rapidfuzz.distance import Levenshtein

from .. import audio
from ..config import load_config
from ..listen.storage import read, save_board
from ..sub_align import comparison_normalize, indexed_normalize
from . import get_backend
from .common import ASRError
from .postprocess import records, seconds

STATES = ("一致", "不一致", "仅主路", "仅副路", "人工确认")


def overlap(a, b):
    return min(seconds(a["止"]), seconds(b["止"])) > max(seconds(a["起"]), seconds(b["起"]))


def align(primary, secondary):
    """有序动态规划；每条候选必须实际重叠，不用文字相似度拒绝异文。"""
    n, m = len(primary), len(secondary)
    dp = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    back = {}
    dp[0][0] = 0

    def update(i, j, score, previous):
        if score > dp[i][j]:
            dp[i][j], back[i, j] = score, previous

    for i in range(n + 1):
        for j in range(m + 1):
            score = dp[i][j]
            if i < n:
                update(i + 1, j, score, (i, j))
            if j < m:
                update(i, j + 1, score, (i, j))
            if i == n or j == m or not overlap(primary[i], secondary[j]):
                continue
            # 一对多和多对一；输出阶段仍保留每条主路的时间与声线。
            candidates = [(1, 1)]
            for a in range(2, n - i + 1):
                rows = primary[i:i + a]
                if not overlap(rows[-1], secondary[j]):
                    break
                candidates.append((a, 1))
            for b in range(2, m - j + 1):
                if not overlap(primary[i], secondary[j + b - 1]):
                    break
                candidates.append((1, b))
            for a, b in candidates:
                left, right = primary[i:i + a], secondary[j:j + b]
                lt = comparison_normalize("".join(r["文本"] for r in left))
                rt = comparison_normalize("".join(r["文本"] for r in right))
                similarity = Levenshtein.normalized_similarity(lt, rt)
                start = max(seconds(left[0]["起"]), seconds(right[0]["起"]))
                end = min(max(seconds(r["止"]) for r in left), max(seconds(r["止"]) for r in right))
                duration = min(max(seconds(r["止"]) for r in left) - seconds(left[0]["起"]),
                               max(seconds(r["止"]) for r in right) - seconds(right[0]["起"]))
                temporal = max(0, end - start) / max(0.1, duration)
                reward = min(len(lt), len(rt)) * (2 * similarity + 0.4 * temporal) - 0.2 * (a + b - 2)
                update(i + a, j + b, score + reward, (i, j))
    groups = []
    i, j = n, m
    while i or j:
        pi, pj = back[i, j]
        groups.append((primary[pi:i], secondary[pj:j]))
        i, j = pi, pj
    return list(reversed(groups))


def priority(state, primary, secondary, subtitle="", *, duration=1, truncated=False):
    # subtitle 仅保留调用兼容性，不能影响自动优先级。
    a, b = map(comparison_normalize, (primary, secondary))
    similarity = Levenshtein.normalized_similarity(a, b)
    if truncated or state in ("仅主路", "仅副路") or state == "不一致" and similarity < .6:
        return 1
    if duration < .5 or duration > 6:
        return 1
    if state == "不一致" and similarity >= .8:
        return 3
    return 2


def needs_listen(row):
    state = row.get("核对", {}).get("状态")
    if state == "人工确认":
        return False
    duration = seconds(row["止"]) - seconds(row["起"])
    return state != "一致" or row.get("截断", False) or duration < .5 - 1e-8 or duration > 6 + 1e-8


def split_reference(primary, reference):
    """按归一化字符对齐切副路原文，主路断句不随另一后端改变。"""
    if len(primary) == 1:
        return [reference]
    text = "".join(r["文本"] for r in primary)
    left, left_positions = indexed_normalize(text)
    right, right_positions = indexed_normalize(reference)
    operations = SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
    cuts, boundary = [0], 0
    for row in primary[:-1]:
        boundary += len(row["文本"])
        position = next((i for i, span in enumerate(left_positions) if span[0] >= boundary), len(left))
        mapped = len(right)
        for _kind, i, j, a, b in operations:
            if i <= position <= j:
                mapped = a + round((b - a) * (position - i) / (j - i)) if j > i else a
                break
        cut = right_positions[mapped][0] if mapped < len(right_positions) else len(reference)
        # 标点跟随前一句，不把句号切到下一句开头。
        while cut < len(reference) and not comparison_normalize(reference[cut]):
            cut += 1
        cuts.append(max(cuts[-1], cut))
    cuts.append(len(reference))
    return [reference[a:b] for a, b in zip(cuts, cuts[1:])]


def merge(primary, secondary, subtitles=(), episode="E01"):
    primary = sorted(deepcopy(primary), key=lambda r: (seconds(r["起"]), seconds(r["止"])))
    secondary = sorted(deepcopy(secondary), key=lambda r: (seconds(r["起"]), seconds(r["止"])))
    output = []
    # 副路标签只在副路内有效，不能直接冒充同号的主路声线。
    next_voice = max((int(r["声线"].split("-V")[1]) for r in primary), default=0)
    secondary_voices = {}
    for left, right in align(primary, secondary):
        reference = "".join(r["文本"] for r in right)
        parts = split_reference(left, reference) if left else [r["文本"] for r in right]
        for original, part in zip(left or right, parts):
            row = {k: deepcopy(v) for k, v in original.items() if k != "核对"}
            a, b = (row["文本"], part) if left else ("", part)
            state = ("一致" if comparison_normalize(a) == comparison_normalize(b) else "不一致") if left and right else "仅主路" if left else "仅副路"
            if not left:
                voice = row["声线"]
                if voice not in secondary_voices:
                    next_voice += 1
                    secondary_voices[voice] = f"{episode}-V{next_voice}"
                row["声线"] = secondary_voices[voice]
            subtitle = ""
            row["核对"] = {"状态": state, "主路": a, "副路": b, "字幕": subtitle,
                           "优先级": priority(state, a, b, duration=seconds(row["止"]) - seconds(row["起"]), truncated=row.get("截断", False))}
            output.append(row)
    output.sort(key=lambda r: (seconds(r["起"]), seconds(r["止"])))
    for index, row in enumerate(output, 1):
        row["id"] = f"{episode}-S{index:03d}"
    return output


def update_pending(sounds, existing, episode, migrations=()):
    pending = deepcopy(existing)
    number = max((int(r["id"].split("-Q")[1]) for r in pending), default=0)
    used = set()
    migrated = {r["新句号"]: r for r in migrations if r.get("状态") == "待确认"}
    for sound in sounds:
        check = sound["核对"]
        # 同时开始的不同句子各占一个问题；插句后时间与句号一起匹配。
        candidates = [r for r in pending if r["id"] not in used and r["来源"] == "识别"
                      and r["处理职责"] == "回听" and r["类型"] == "全文完整性与一致性"]
        found = next((r for r in candidates if r["位置"].get("句") == sound["id"]
                      and r["位置"].get("时间", sound["起"]) == sound["起"]), None)
        if found is None:
            found = next((r for r in candidates if r["位置"].get("时间") == sound["起"]), None)
        if found:
            used.add(found["id"])
        if not needs_listen(sound) and sound["id"] not in migrated:
            if found:
                found.update({"位置": {**found["位置"], "句": sound["id"], "时间": sound["起"]},
                              "状态": "已裁定", "复核结果": "两路一致或已人工确认"})
            continue
        evidence = f'主路：{check["主路"]}\n副路：{check["副路"]}'
        if sound["id"] in migrated:
            evidence += "\n原裁定：" + migrated[sound["id"]]["原裁定"]
        if found is None:
            number += 1
            found = {"id": f"{episode}-Q{number:03d}"}
            pending.append(found)
        used.add(found["id"])
        found.update({"来源": "识别", "位置": {"句": sound["id"], "时间": sound["起"]},
                      "原文": sound["文本"], "类型": "全文完整性与一致性", "级别": "阻断",
                      "相关人物": [], "依据": evidence, "修正要求": "回听本句及前后上下文，裁定实际台词。",
                      "处理职责": "回听", "状态": "待核对", "复核结果": None})
    for issue in pending:
        if issue["id"] not in used and issue["来源"] == "识别" and issue["处理职责"] == "回听":
            issue.update({"状态": "已裁定", "复核结果": "原切句问题已归档；按当前共享句界问题回听。"})
    return pending


def run(video, episode_dir, primary_path=None, secondary_path=None, *,
        primary_words=None, secondary_words=None, voice_path=None, min_pause=None, max_sentence=None):
    from .segment import load_words, segment_pair
    from ..vad import detect, waveform
    episode_dir = Path(episode_dir)
    configuration = load_config()
    settings = configuration["asr"]
    cross = settings.get("cross_check", {})
    if cross.get("enabled") is not True:
        raise ValueError("须配置 asr.cross_check.enabled=true；单路识别请用 tools/asr.py")
    providers = [cross.get("primary"), cross.get("secondary")]
    if not all(isinstance(p, str) and p for p in providers) or providers[0] == providers[1]:
        raise ValueError("交叉核对须配置两个不同的 primary、secondary 后端")
    episode = episode_dir.name
    records([], episode)
    episode = f"E{int(str(episode).lstrip('Ee')):02d}"
    # 先核对输入；缺词时间时不初始化识别后端，也不制造替代词时间。
    paths = [primary_path, secondary_path]
    word_paths = [primary_words, secondary_words]
    lanes = [None, None]
    for i, (path, word_path) in enumerate(zip(paths, word_paths)):
        if word_path is not None:
            lanes[i] = load_words(word_path)
        elif path is not None:
            raise ValueError("复用句级结果还须提供 --主路词 / --副路词 原始词级响应；不会重新识别补齐")
    episode_dir.mkdir(parents=True, exist_ok=True)
    segmentation = settings.get("segmentation", {})
    with tempfile.TemporaryDirectory(prefix=".asr-cross-", dir=episode_dir) as directory:
        wav = audio.extract_audio(video, Path(directory) / "source.wav")
        voices = read(voice_path) if voice_path else detect(wav, segmentation.get("vad", {}))
        for i in range(2):
            if lanes[i] is not None:
                continue
            backend = get_backend(providers[i])
            raw = episode_dir / "素材/asr_raw" / f"cross-{i + 1}-{providers[i]}"
            raw.mkdir(parents=True, exist_ok=True)
            backend.settings["raw_dir"] = str(raw)
            backend.transcribe(wav)
            lanes[i] = getattr(backend, "last_words", None)
            if lanes[i] is None:
                raise ValueError("后端未提供原始词级响应，不能按音频统一切句")
        sounds, _ = segment_pair(*lanes, voices, waveform(wav), episode,
                                min_pause=min_pause if min_pause is not None else segmentation.get("min_pause", .5),
                                max_sentence=max_sentence if max_sentence is not None else segmentation.get("max_sentence", 6))
    pending_path = episode_dir / "待核对.json"
    existing = read(pending_path) if pending_path.exists() else []
    previous_path = episode_dir / "声音记录.json"
    previous = {r["id"]: r for r in read(previous_path)} if previous_path.exists() else {}
    for issue in existing:
        old = previous.get(issue["位置"].get("句"))
        if old:
            choices = [r for r in sounds if overlap(r, old)]
            if choices:
                target = max(choices, key=lambda r: Levenshtein.normalized_similarity(
                    comparison_normalize(old["文本"]), comparison_normalize(r["文本"])))
                issue["位置"]["句"] = target["id"]
            else:
                del issue["位置"]["句"]
                issue["位置"].setdefault("时间", old["起"])
                issue["依据"] += f'\n原句 {old["id"]} 已不在当前识别结果中，按时间回查。'
    pending = update_pending(sounds, existing, episode)
    diarize_settings = configuration.get("diarize", {})
    voice_result = None
    if diarize_settings.get("enabled") is True:
        from ..diarize.integration import analyze_board
        raw_dirs = [Path(path).parent / "素材/asr_raw" for path, provider in zip(paths, providers)
                    if path is not None and provider == "grok"]
        raw_dirs.extend(Path(path).parent for path, provider in zip(word_paths, providers)
                        if path is not None and provider == "grok")
        # 旧口型不能约束新句界；在不含旧口型文件的暂存目录执行。
        with tempfile.TemporaryDirectory(prefix=".diarize-", dir=episode_dir.parent) as directory:
            sounds, voice_result = analyze_board(video, sounds, directory, diarize_settings,
                                                 [episode_dir / "素材/asr_raw", *raw_dirs])
    saved = save_board(episode_dir, sounds, pending, "合并")
    from ..listen.storage import write
    write(episode_dir / "素材/人声段.json", voices)
    extra = {}
    if voice_result is not None:
        if (episode_dir / "声纹聚类.json").exists():
            read(episode_dir / "声纹聚类.json")
        write(episode_dir / "声纹聚类.json", voice_result)
        extra["声纹"] = {"类数": len(voice_result["类别"]), "输入可用性": voice_result["输入可用性"]}
    return {"主路": providers[0], "副路": providers[1], "统计": {s: sum(r["核对"]["状态"] == s for r in sounds) for s in STATES},
            "待回听": sum(needs_listen(r) for r in sounds), **saved, **extra}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("视频", type=Path)
    parser.add_argument("本集目录", type=Path)
    parser.add_argument("--主路结果", type=Path)
    parser.add_argument("--副路结果", type=Path)
    parser.add_argument("--主路词", type=Path, help="原始词级响应；复用句级结果时必填")
    parser.add_argument("--副路词", type=Path, help="原始词级响应；复用句级结果时必填")
    parser.add_argument("--人声段", type=Path)
    parser.add_argument("--min-pause", type=float)
    parser.add_argument("--max-sentence", type=float)
    args = parser.parse_args(argv)
    try:
        result = run(args.视频, args.本集目录, args.主路结果, args.副路结果,
                     primary_words=args.主路词, secondary_words=args.副路词, voice_path=args.人声段,
                     min_pause=args.min_pause, max_sentence=args.max_sentence)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, KeyError, ASRError, ImportError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"两路核对失败：{exc}\n")
    return 0
