"""以人声及短时能量建立共享句界，两路词按中点归句；不接受字幕。"""

from copy import deepcopy
import json
import math
from pathlib import Path
import re

import numpy as np

from .postprocess import join_text, seconds, timestamp


def prepare_words(words, duration):
    result = []
    prefix = ""
    for raw in deepcopy(words):
        text = raw["text"].strip()
        a, b = seconds(raw["start"]), seconds(raw["end"])
        if not text:
            continue
        if not 0 <= a <= b <= duration + .15 or a >= duration:
            raise ValueError("词时间戳越界")
        if not any(c.isalnum() for c in text):
            if result:
                result[-1]["text"] += text
            else:
                prefix += text
            continue
        if a == b:
            raise ValueError("零长实字不能作为切句依据")
        result.append({**raw, "text": prefix + text, "start": a, "end": min(duration, b)})
        prefix = ""
    return result


def load_words(path):
    """读取单份原始响应；窗口词时间转成整集绝对时间。"""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, list):
        words, offset = value, 0
    else:
        if value.get("status", 200) != 200:
            raise ValueError("不能使用失败的识别响应")
        response = value.get("response", value)
        if "words" not in response:
            raise ValueError(f"缺少原始词级响应：{path}；不能从句级时间推造词时间")
        words, offset = response["words"], value.get("window", [0])[0]
    return [{**w, "start": seconds(w["start"]) + offset,
             "end": seconds(w["end"]) + offset} for w in words]


def union(intervals):
    result = []
    for a, b in sorted(intervals, key=lambda span: (span[0], span[1])):
        if b <= a:
            continue
        if result and a <= result[-1][1] + 1e-9:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def energies(samples, rate=16000):
    """20 ms 窗、10 ms 步长，边界选取仍受两路词的完整跨度约束。"""
    size, hop = round(rate * .02), round(rate * .01)
    squared = np.asarray(samples, dtype=np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    starts = np.arange(0, len(samples), hop)
    stops = np.minimum(starts + size, len(samples))
    return (starts + stops) / (2 * rate), (cumulative[stops] - cumulative[starts]) / (stops - starts)


def voice_intervals(voices, total):
    spans = []
    for row in voices:
        if "文字" in row or "类型" in row:
            raise ValueError("音频切句只接受人声段，不接受字幕")
        a, b = seconds(row["起"]), seconds(row["止"])
        if not 0 <= a < b <= total + .02:
            raise ValueError("人声段时间越界")
        spans.append((a, min(b, total)))
    return union(spans)


def resolve_timing(lanes, voices, samples, rate=16000):
    """保留原词顺序；VAD 长停顿优先于错时词，不修改原始响应。"""
    spans = voice_intervals(voices, len(samples) / rate)
    result = deepcopy(lanes)
    if not spans:
        return result
    times, energy = energies(samples, rate)
    pauses = [(left[1], right[0]) for left, right in zip(spans, spans[1:])
              if right[0] - left[1] >= .5 - 1e-8]
    from ..sub_align import comparison_normalize
    for index, lane in enumerate(result):
        for word in lane:
            a, b = word["start"], word["end"]
            if not any(min(b, end) > max(a, start) for start, end in spans):
                token = comparison_normalize(word["text"])
                supported = any(min(b, other["end"]) > max(a, other["start"]) and
                                (token in comparison_normalize(other["text"]) or
                                 comparison_normalize(other["text"]) in token)
                                for other in result[1 - index])
                # 两路同词同时是 VAD 漏检证据，不能把真实台词挪进邻句。
                if not supported:
                    word["_untimed"] = True
                continue
            for left, right in pauses:
                if a <= left and b >= right:
                    mask = (times >= left) & (times <= right)
                    points = times[mask]
                    levels = energy[mask]
                    quiet = points[levels == levels.min()]
                    cut = min(quiet, key=lambda t: abs(t - (left + right) / 2))
                    if cut - a >= b - cut:
                        b = left
                    else:
                        a = right
            word.update(start=a, end=b)
    return result


def boundaries(voices, lanes, samples, *, rate=16000, min_pause=.5, max_sentence=6):
    if any(isinstance(n, bool) or not math.isfinite(n) or n <= 0 for n in (min_pause, max_sentence)):
        raise ValueError("min_pause、max_sentence 须为有限正数")
    total = len(samples) / rate
    words = [w for lane in lanes for w in lane if not w.get("_untimed")]
    voice_spans = voice_intervals(voices, total)
    # 可靠词保护完整跨度；已确认的长停顿及无声区错时词不受此约束。
    spans = union([*voice_spans, *((w["start"], w["end"]) for w in words)])
    if not spans:
        return []
    times, energy = energies(samples, rate)

    def legal(point):
        return not any(w["start"] + 1e-8 < point < w["end"] - 1e-8 for w in words)

    def quiet(a, b):
        # 输出以 0.1 秒存储，直接选合法网格，避免四舍五入后又切进字里。
        ticks = [i / 10 for i in range(math.ceil((a - 1e-8) * 10), math.floor((b + 1e-8) * 10) + 1)
                 if legal(i / 10)]
        if not ticks:
            return None
        return min(ticks, key=lambda t: (float(energy[np.argmin(abs(times - t))]), abs(t - (a + b) / 2)))

    pauses = [(left[1], right[0]) for left, right in zip(spans, spans[1:])]
    groups = [[spans[0][0], spans[0][1]]]
    for a, b in spans[1:]:
        if a - groups[-1][1] >= min_pause - 1e-8:
            groups.append([a, b])
        else:
            groups[-1][1] = max(b, groups[-1][1])
    # 对话换人常不足停顿阈值；句末标点后有真实词间空隙时另起一句。
    ordered = sorted(words, key=lambda w: w["start"])
    terminal = [(w["end"], nxt["start"]) for w, nxt in zip(ordered, ordered[1:])
                if re.search(r"[。！？!?]$", w["text"]) and nxt["start"] - w["end"] >= .05 - 1e-8]
    split = []
    for a, b in groups:
        for p, q in terminal:
            if a < p and q < b:
                split.append([a, p])
                a = q
        split.append([a, b])
    groups = split
    result = []
    for index, (a, b) in enumerate(groups):
        # 句外最多保留 0.2 秒静音；长空镜不塞进一条台词里。
        previous_end = groups[index - 1][1] if index else 0
        next_start = groups[index + 1][0] if index + 1 < len(groups) else total
        start = quiet(max(previous_end, a - .2), a)
        end = quiet(b, min(next_start, b + .2, math.floor(total * 10) / 10))
        start = math.floor(a * 10) / 10 if start is None else start
        # 片尾用最近的 0.1 秒，误差不超过 0.05 秒，兼容声纹和口型的音频边界。
        end = (round(total * 10) / 10 if b > math.floor(total * 10) / 10
               else math.ceil(b * 10) / 10) if end is None else end
        pending = [(start, end)]
        while pending:
            left, right = pending.pop(0)
            if right - left <= max_sentence + 1e-8:
                result.append((left, right))
                continue
            candidates = []
            for p, q in pauses:
                if left + .3 <= p < q <= right - .3:
                    cut = quiet(p, q)
                    if cut is not None:
                        terminal = any(abs(w["end"] - p) < .05 and re.search(r"[。！？!?]$", w["text"]) for w in words)
                        candidates.append((q - p, terminal, cut))
            # VAD 连续段内只接受确实低能量的 >=80 ms 停顿，不能只凭标点硬切。
            mask = (times >= left + .3) & (times <= right - .3)
            if mask.any():
                threshold = np.quantile(energy[mask], .2)
                low = times[mask & (energy <= threshold)]
                runs = []
                for t in low:
                    if runs and t - runs[-1][-1] <= .011:
                        runs[-1].append(float(t))
                    else:
                        runs.append([float(t)])
                for run in runs:
                    if run[-1] - run[0] < .08:
                        continue
                    cut = quiet(run[0], run[-1])
                    if cut is not None:
                        candidates.append((run[-1] - run[0], False, cut))
            if not candidates:
                # 没有停顿就保留超长句，不伪造字内切点。
                result.append((left, right))
                continue
            cut = max(candidates)[2]
            pending[:0] = [(left, cut), (cut, right)]
    return sorted(result)


def assign(words, spans, episode):
    buckets = [[] for _ in spans]
    positions = []
    for word in words:
        if word.get("_untimed"):
            positions.append(None)
            continue
        midpoint = (word["start"] + word["end"]) / 2
        matches = [i for i, (a, b) in enumerate(spans) if a <= midpoint < b]
        if len(matches) != 1:
            raise ValueError(f"词没有唯一归句：{word['text']} @ {midpoint:g}")
        positions.append(matches[0])
    for index, word in enumerate(words):
        position = positions[index]
        if position is None:
            previous = next((p for p in reversed(positions[:index]) if p is not None), None)
            following = next((p for p in positions[index + 1:] if p is not None), None)
            # 尾词跟随前文；句首无前文时跟随后文，不按错误时间再造一句。
            position = previous if previous is not None else following
            if position is None:
                midpoint = (word["start"] + word["end"]) / 2
                position = min(range(len(spans)), key=lambda i: abs(sum(spans[i]) / 2 - midpoint))
        buckets[position].append(word)
    rows = []
    labels = {}
    for i, ((a, b), bucket) in enumerate(zip(spans, buckets), 1):
        if not bucket:
            continue
        text = ""
        for w in bucket:
            text = join_text(text, w["text"], 0)
        speaker = max(bucket, key=lambda w: w["end"] - w["start"]).get("speaker")
        label = labels.setdefault(speaker, f"{episode}-V{len(labels) + 1}")
        rows.append({"id": f"{episode}-S{i:03d}", "起": timestamp(a), "止": timestamp(b),
                     "文本": text, "声线": label, "听辨": "清楚"})
    return rows


def segment_pair(primary, secondary, voices, samples, episode="E01", *, min_pause=.5, max_sentence=6):
    from .crosscheck import priority
    duration = len(samples) / 16000
    lanes = resolve_timing([prepare_words(lane, duration) for lane in (primary, secondary)], voices, samples)
    spans = boundaries(voices, lanes, samples, min_pause=min_pause, max_sentence=max_sentence)
    primary_rows, secondary_rows = [assign(lane, spans, episode) for lane in lanes]
    by_lane = [{(r["起"], r["止"]): r for r in rows} for rows in (primary_rows, secondary_rows)]
    output = []
    for start, end in spans:
        key = (timestamp(start), timestamp(end))
        a, b = [lane.get(key) for lane in by_lane]
        if not a and not b:
            continue
        row = deepcopy(a or b)
        left, right = a["文本"] if a else "", b["文本"] if b else ""
        from ..sub_align import comparison_normalize
        state = ("一致" if comparison_normalize(left) == comparison_normalize(right) else "不一致") if a and b else "仅主路" if a else "仅副路"
        truncated = (start == 0 and any(seconds(v["起"]) <= .01 for v in voices) or
                     end >= math.floor(duration * 10) / 10 and any(duration - seconds(v["止"]) <= .02 for v in voices))
        if truncated:
            row["截断"] = True
        if not a:
            row["声线"] = f"{episode}-V{len(primary_rows) + int(row['声线'].split('-V')[1])}"
        row["id"] = f"{episode}-S{len(output) + 1:03d}"
        row["核对"] = {"状态": state, "主路": left, "副路": right, "字幕": "",
                       "优先级": priority(state, left, right, duration=end - start, truncated=truncated)}
        output.append(row)
    return output, spans


def missing_voice(voices, primary, secondary):
    """人声段中两路都没有词的部分；仅一路漏字不触发补识别。"""
    coverage = union((seconds(w["start"]), seconds(w["end"])) for w in [*primary, *secondary]
                     if any(c.isalnum() for c in w["text"]))
    result = []
    for voice in voices:
        if "文字" in voice or "类型" in voice:
            raise ValueError("漏句检测只接受人声段，不接受字幕")
        a, b = seconds(voice["起"]), seconds(voice["止"])
        if not a < b:
            raise ValueError("人声段起止无效")
        cursor = a
        for left, right in coverage:
            if right <= cursor or left >= b:
                continue
            if left - cursor >= .5 - 1e-8:
                result.append({"起": cursor, "止": left})
            cursor = min(b, max(cursor, right))
        if b - cursor >= .5 - 1e-8:
            result.append({"起": cursor, "止": b})
    return result
