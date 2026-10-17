"""按时间和字符相似度做有序多对多对齐；结果仅供回听核对。"""

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import unicodedata

from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import partial_ratio

from .subtitles import normalize, timestamp, to_seconds


# 时刻须先于普通数字识别，否则 3:40 会在去标点后与 340 混为一谈。
_DIGITS = dict(zip("零〇一二两三四五六七八九", "001223456789"))
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}
_NUMBER = r"[0-9零〇一二两三四五六七八九十百千万亿]+"
_NUMERIC = re.compile(
    rf"(?P<clock>{_NUMBER}(?:时|点){_NUMBER}分|{_NUMBER}点[零〇一二两三四五六七八九]*十[零〇一二两三四五六七八九]?|[0-9]{{1,2}}:[0-9]{{2}})"
    rf"|(?P<decimal>{_NUMBER}[.点][0-9零〇一二两三四五六七八九]+)"
    rf"|(?P<integer>{_NUMBER})"
)


def _integer(token: str) -> str:
    if not any(c in _UNITS for c in token):
        return str(int("".join(_DIGITS.get(c, c) for c in token)))
    total, section, number = 0, 0, ""
    for c in token:
        if c not in _UNITS:
            number += _DIGITS.get(c, c)
        elif _UNITS[c] < 10000:
            section += int(number or "1") * _UNITS[c]
            number = ""
        else:
            section += int(number or "0")
            if c == "亿":
                total = (total + section or 1) * _UNITS[c]
            else:
                total += (section or 1) * _UNITS[c]
            section, number = 0, ""
    return str(total + section + int(number or "0"))


def indexed_normalize(text: str) -> tuple[str, list[tuple[int, int]]]:
    """比较单元映射回原文区间；数字缩写或展开时仍指向完整原始数字。"""
    chars, source = [], []
    for index, char in enumerate(text):
        for c in unicodedata.normalize("NFKC", char):
            chars.append(c)
            source.append((index, index + 1))
    wide = "".join(chars)
    output, positions = [], []

    def append_plain(start, end):
        for i in range(start, end):
            c = wide[i]
            if not c.isspace() and not unicodedata.category(c).startswith("P"):
                output.append(c)
                positions.append(source[i])

    previous = 0
    for match in _NUMERIC.finditer(wide):
        append_plain(previous, match.start())
        token = match.group()
        if match.lastgroup == "clock":
            hour, minute = re.split(r"[:点时]", token.removesuffix("分"))
            h, m = int(_integer(hour)), int(_integer(minute))
            converted = f"{h}:{m:02d}" if 0 <= h <= 23 and 0 <= m < 60 else token
        elif match.lastgroup == "decimal":
            whole, fraction = re.split(r"[.点]", token)
            converted = _integer(whole) + "." + "".join(_DIGITS.get(c, c) for c in fraction)
        else:
            converted = _integer(token)
        span = (source[match.start()][0], source[match.end() - 1][1])
        output.extend(converted)
        # 等长阿拉伯数字保留逐字定位；转换过的数字按整个词定位。
        positions.extend(source[match.start():match.end()] if converted == token else [span] * len(converted))
        previous = match.end()
    append_plain(previous, len(wide))
    return "".join(output), positions


def comparison_normalize(text: str) -> str:
    return indexed_normalize(text)[0]


def differences(subtitle: str, sound: str) -> list[dict]:
    left, left_positions = indexed_normalize(subtitle)
    right, right_positions = indexed_normalize(sound)

    def original_range(positions, start, end, text):
        if start == end:
            point = positions[start][0] if start < len(positions) else len(text)
            return [point, point]
        return [positions[start][0], positions[end - 1][1]]

    output = []
    for kind, i, j, a, b in SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        if kind != "equal":
            output.append({"操作": kind, "字幕位置": [i, j], "声音位置": [a, b],
                           "字幕原文位置": original_range(left_positions, i, j, subtitle),
                           "声音原文位置": original_range(right_positions, a, b, sound),
                           "字幕文字": left[i:j], "声音文字": right[a:b]})
    return output


def _prepare(rows, text_field, subtitle=False):
    output = []
    for index, row in enumerate(rows, 1):
        if subtitle and row["类型"] != "台词字幕":
            continue
        if not normalize(row[text_field]):
            continue
        start, end = to_seconds(row["起"]), to_seconds(row["止"])
        if end < start:
            raise ValueError(f"止时间早于起时间：{row}")
        output.append({"id": index if subtitle else row["id"], "text": row[text_field],
                       "norm": comparison_normalize(row[text_field]), "start": start, "end": end,
                       "confidence": row.get("置信度")})
    return sorted(output, key=lambda row: (row["start"], row["end"]))


def _overlap(a, b):
    # 单帧字幕起止相同；仅在对齐打分时给 0.1 秒宽度，不改写源时间。
    return min(max(a["end"], a["start"] + 0.1), max(b["end"], b["start"] + 0.1)) - max(a["start"], b["start"])


def temporal_score(left, right, slack):
    if (any(not any(_overlap(a, b) >= -slack for b in right) for a in left)
            or any(not any(_overlap(a, b) >= -slack for a in left) for b in right)):
        return None
    start = max(left[0]["start"], right[0]["start"])
    end = min(max(r["end"] for r in left), max(r["end"] for r in right))
    duration = min(max(r["end"] for r in left) - left[0]["start"],
                   max(r["end"] for r in right) - right[0]["start"])
    return max(0, min(1, (end - start) / max(0.1, duration)))


def _uncommon(text):
    """以 GB2312 一级汉字区（3755 字）作为常用字的可复现近似。"""
    han = [c for c in text if unicodedata.name(c, "").startswith("CJK UNIFIED IDEOGRAPH")]
    rare = []
    for c in han:
        try:
            encoded = c.encode("gb2312")
            common = len(encoded) == 2 and 0xB0 <= encoded[0] <= 0xD7
        except UnicodeEncodeError:
            common = False
        if not common:
            rare.append(c)
    return rare, len(rare) / len(han) if han else 0.0


def _entry(ls, rs):
    rows = ls + rs
    return {"起": timestamp(min(r["start"] for r in rows)), "止": timestamp(max(r["end"] for r in rows)),
            "字幕编号": [r["id"] for r in ls], "声音编号": [r["id"] for r in rs],
            "字幕原文": [r["text"] for r in ls], "声音原文": [r["text"] for r in rs]}


def _separate_noise(left, right, *, similarity, confidence, rare_ratio, rare_min_chars):
    clean, noise = [], []
    for row in left:
        # 无实际重叠时不能排除 ASR 漏句，不用邻近但无关的声音给噪声定性。
        overlapping = [r for r in right
                       if min(row["end"], r["end"]) > max(row["start"], r["start"])
                       or (row["start"] == row["end"] and r["start"] <= row["start"] < r["end"])]
        rare, ratio = _uncommon(row["text"])
        low_confidence = row["confidence"] is not None and row["confidence"] <= confidence
        many_rare = len(rare) >= rare_min_chars and ratio >= rare_ratio
        local_similarity = max((partial_ratio(row["norm"], r["norm"]) / 100 for r in overlapping), default=1.0)
        if overlapping and local_similarity < similarity and (low_confidence or many_rare):
            entry = _entry([row], overlapping)
            entry["起"], entry["止"] = timestamp(row["start"]), timestamp(row["end"])
            entry["噪声依据"] = {"局部相似度": round(local_similarity, 4), "字幕置信度": row["confidence"],
                                   "非常用字": rare, "非常用字占比": round(ratio, 4),
                                   "低置信度": low_confidence, "大量非常用字": many_rare}
            noise.append(entry)
        else:
            clean.append(row)
    return clean, noise


def compare(subtitles: list[dict], sounds: list[dict], *, max_span=32, time_slack=0.35, min_similarity=0.35,
            noise_similarity=0.2, noise_confidence=0.4, noise_rare_ratio=0.5, noise_rare_min_chars=2) -> dict:
    thresholds = (min_similarity, noise_similarity, noise_confidence, noise_rare_ratio)
    if max_span < 1 or time_slack < 0 or any(not 0 <= v <= 1 for v in thresholds) or noise_rare_min_chars < 1:
        raise ValueError("合并上限和非常用字最少字数须大于 0，时间容差不得为负，相似度、置信度和比例须在 0 到 1 之间")
    left, right = _prepare(subtitles, "文字", True), _prepare(sounds, "文本")
    subtitle_count = len(left)
    left, noise = _separate_noise(left, right, similarity=noise_similarity, confidence=noise_confidence,
                                  rare_ratio=noise_rare_ratio, rare_min_chars=noise_rare_min_chars)
    n, m = len(left), len(right)
    dp = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    def span_text(rows):
        return {(i, count): comparison_normalize("".join(r["text"] for r in rows[i:i + count]))
                for i in range(len(rows)) for count in range(1, min(max_span, len(rows) - i) + 1)}

    left_text, right_text = span_text(left), span_text(right)

    def update(i, j, score, previous):
        if score > dp[i][j]:
            dp[i][j], back[i][j] = score, previous

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == float("-inf"):
                continue
            if i < n:
                update(i + 1, j, dp[i][j], (i, j))
            if j < m:
                update(i, j + 1, dp[i][j], (i, j))
            if i == n or j == m:
                continue
            right_limit = min(max_span, m - j)
            latest_right = max(max(r["end"], r["start"] + 0.1) for r in right[j:j + right_limit])
            for a in range(1, min(max_span, n - i) + 1):
                ls = left[i:i + a]
                if ls[-1]["start"] > latest_right + time_slack:
                    break
                latest_left = max(max(r["end"], r["start"] + 0.1) for r in ls)
                for b in range(1, right_limit + 1):
                    rs = right[j:j + b]
                    # 只扩展到当前时间窗口内的行，长句自然获得更多候选。
                    if rs[-1]["start"] > latest_left + time_slack:
                        break
                    temporal = temporal_score(ls, rs, time_slack)
                    if temporal is None:
                        continue
                    lt, rt = left_text[i, a], right_text[j, b]
                    similarity = Levenshtein.normalized_similarity(lt, rt)
                    if similarity < min_similarity and temporal < 0.6:
                        continue
                    reward = similarity * (len(lt) + len(rt)) + 0.3 * temporal * min(len(lt), len(rt)) - 0.15 * (a + b - 2)
                    update(i + a, j + b, dp[i][j] + reward, (i, j))
    groups = []
    i, j = n, m
    while i or j:
        pi, pj = back[i][j]
        groups.append((left[pi:i], right[pj:j]))
        i, j = pi, pj
    result = {"用字不一致": [], "只有字幕": [], "只有声音": [], "断句不一致": [], "疑似识别噪声": noise}
    matched = 0
    for ls, rs in reversed(groups):
        entry = _entry(ls, rs)
        if not ls:
            result["只有声音"].append(entry)
        elif not rs:
            result["只有字幕"].append(entry)
        else:
            matched += 1
            changes = differences("".join(entry["字幕原文"]), "".join(entry["声音原文"]))
            entry["差异位置"] = changes
            if changes:
                result["用字不一致"].append(entry)
            if ((len(ls) > 1 or len(rs) > 1)
                    and [len(r["norm"]) for r in ls] != [len(r["norm"]) for r in rs]):
                result["断句不一致"].append(entry)
    result["统计"] = {"字幕条数": subtitle_count, "声音条数": m, "配对组数": matched,
                       **{k: len(v) for k, v in result.items() if k != "疑似识别噪声" or v}}
    result["位置说明"] = "字幕编号为字幕记录.json 中从 1 开始的条目序号；差异位置为两边各自拼接文本的从 0 开始、左闭右开区间，数字、时刻仅在比较时归一，原文位置包含标点，转换数字的差异映射到整个原始数字。断句不一致仅为分组提示，不代表用字错误。疑似识别噪声中的声音为重叠参照，不消耗声音配对。"
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--max-span", type=int, default=32, help="每侧合并条数上限；实际按时间重叠扩展（默认 32）")
    parser.add_argument("--time-slack", type=float, default=0.35)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--noise-similarity", type=float, default=0.2, help="疑似噪声局部相似度上限（严格小于）")
    parser.add_argument("--noise-confidence", type=float, default=0.4, help="低置信度上限（含边界）")
    parser.add_argument("--noise-rare-ratio", type=float, default=0.5, help="非常用汉字占比下限（含边界）")
    parser.add_argument("--noise-rare-min-chars", type=int, default=2, help="非常用汉字最少字数")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        subtitles = json.loads((args.episode / "字幕记录.json").read_text(encoding="utf-8"))
        sounds = json.loads((args.episode / "声音记录.json").read_text(encoding="utf-8"))
        result = compare(subtitles, sounds, max_span=args.max_span, time_slack=args.time_slack,
                         min_similarity=args.min_similarity, noise_similarity=args.noise_similarity,
                         noise_confidence=args.noise_confidence, noise_rare_ratio=args.noise_rare_ratio,
                         noise_rare_min_chars=args.noise_rare_min_chars)
        target = args.output or args.episode / "字幕声音对照.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"输出": str(target), **result["统计"]}, ensure_ascii=False))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"字幕声音对照失败：{exc}\n")
    return 0
