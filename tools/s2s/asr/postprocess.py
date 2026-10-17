"""时间、句编号和分段边界合并；不推断跨段人物身份。"""

from dataclasses import replace
import math
import re

from .base import Sentence


def seconds(value: str | float) -> float:
    if isinstance(value, bool):
        raise ValueError("时间不能为布尔值")
    if isinstance(value, str) and ":" in value:
        parts = value.split(":")
        if len(parts) not in (2, 3) or any(not re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
            raise ValueError("时间格式无效")
        if any(float(p) >= 60 for p in parts[1:]):
            raise ValueError("时间的秒或小时内分钟越界")
        number = 0.0
        for part in parts:
            number = number * 60 + float(part)
    else:
        number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("时间必须为非负有限数")
    return number


def timestamp(value: float) -> str:
    ticks = round(seconds(value) * 10)
    minutes, rest = divmod(ticks, 600)
    return f"{minutes:02d}:{rest // 10:02d}.{rest % 10}"


def sentence_times(start: float, end: float) -> tuple[str, str]:
    """保留十分之一秒格式，极短应答不因四舍五入变成零长度。"""
    a, b = timestamp(start), timestamp(end)
    if a == b:
        tick = math.floor(start * 10)
        a, b = timestamp(tick / 10), timestamp((tick + 1) / 10)
    return a, b


def words_to_sentences(words: list[dict], offset: float, duration: float, namespace: str,
                       *, pause_seconds=0.35, max_sentence_seconds=6.0,
                       min_speaker_seconds=0.3) -> list[Sentence]:
    if (not math.isfinite(pause_seconds) or pause_seconds <= 0 or
            not math.isfinite(max_sentence_seconds) or max_sentence_seconds <= 0 or
            not math.isfinite(min_speaker_seconds) or min_speaker_seconds < 0):
        raise ValueError("切句参数须为有限正数，min_speaker_seconds 可为零")
    prepared = []
    terminal = r"[。！？!?；;][\"'”’）)]*$"
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("text"), str):
            raise ValueError("词条缺少文本")
        speaker = word.get("speaker")
        if speaker is not None and (isinstance(speaker, bool) or not isinstance(speaker, int) or speaker < 0):
            raise ValueError("词条 speaker 须为非负整数或 null")
        if not word["text"].strip():
            continue
        a, b = seconds(word["start"]), seconds(word["end"])
        punctuation = not any(char.isalnum() for char in word["text"])
        if b < a or b == a and not punctuation or a >= duration or b > duration + 0.15:
            raise ValueError("词时间戳越界")
        prepared.append({**word, "start": a, "end": min(b, duration), "suspect": False})
    prepared.sort(key=lambda w: w["start"])
    # 标点标签可能提前换人，只作为前一实词的标点。
    previous = None
    for word in prepared:
        if not any(c.isalnum() for c in word["text"]):
            if previous:
                word["speaker"] = previous.get("speaker")
        else:
            previous = word

    # 按声线和句末组成短片段。只吸收未结束句中的短跳变，保留独立应答。
    runs = []
    for word in prepared:
        if (not runs or runs[-1][-1].get("speaker") != word.get("speaker") or
                re.search(terminal, runs[-1][-1]["text"])):
            runs.append([])
        runs[-1].append(word)
    for i, run in enumerate(runs):
        span = run[-1]["end"] - run[0]["start"]
        if span >= min_speaker_seconds or run[0].get("speaker") is None:
            continue
        left = runs[i - 1] if i else None
        right = runs[i + 1] if i + 1 < len(runs) else None
        target = None
        if (left and not re.search(terminal, left[-1]["text"]) and
                (run[0]["start"] - left[-1]["end"] <= pause_seconds or
                 re.fullmatch(r"儿[，,。！？!?；;]*", run[0]["text"].strip()))):
            target = left[-1]
        elif (right and not re.search(terminal, run[-1]["text"]) and
              right[0]["start"] - run[-1]["end"] <= pause_seconds):
            target = right[0]
        if target is not None and target.get("speaker") != run[0].get("speaker"):
            for word in run:
                word["speaker"] = target.get("speaker")
                word["suspect"] = True

    # 先识别整句是否过长，再按停顿切段，避免先切停顿后漏掉长句中的逗号。
    phrase = []

    def mark_long():
        if phrase and phrase[-1]["end"] - phrase[0]["start"] > max_sentence_seconds:
            for item in phrase:
                item["long"] = True
        phrase.clear()

    for word in prepared:
        if phrase and word.get("speaker") != phrase[-1].get("speaker"):
            mark_long()
        phrase.append(word)
        if re.search(terminal, word["text"]):
            mark_long()
    mark_long()
    result, buffer = [], []

    def emit(group):
        text = ""
        for word in group:
            text = join_text(text, word["text"].strip(), 0)
        a, b = sentence_times(offset + min(w["start"] for w in group),
                              offset + max(w["end"] for w in group))
        speaker = group[0].get("speaker")
        label = str(speaker) if speaker is not None else f"unknown-{len(result)}"
        clarity = "听不清" if "听不清" in text or any(w["suspect"] for w in group) else "清楚"
        result.append(Sentence(a, b, text, f"{namespace}:{label}", clarity))

    def flush():
        if not buffer:
            return
        # 长句先按模型标点拆分，未带标点时才在词边界限制时长。
        long = any(w.get("long") for w in buffer)
        group = []
        for word in buffer:
            spoken = any(c.isalnum() for c in word["text"])
            suffix = bool(re.fullmatch(r"儿[，,。！？!?；;]*", word["text"].strip()))
            if (group and spoken and not suffix and
                    word["end"] - group[0]["start"] > max_sentence_seconds):
                emit(group)
                group = []
            group.append(word)
            if long and re.search(r"[，,：:][\"'”’）)]*$", word["text"]):
                emit(group)
                group = []
        if group:
            emit(group)
        buffer.clear()

    for word in prepared:
        spoken = any(c.isalnum() for c in word["text"])
        if not spoken and not buffer:
            continue
        # 儿化音不能单独成为下一句，原时间不修造，异常合并标为听不清。
        suffix = bool(re.fullmatch(r"儿[，,。！？!?；;]*", word["text"].strip()))
        if buffer and spoken and (word.get("speaker") != buffer[-1].get("speaker") or
                not suffix and word["start"] - max(w["end"] for w in buffer) >= pause_seconds):
            flush()
        buffer.append(word)
        if re.search(terminal, word["text"]):
            flush()
    flush()
    return result


def normalized(text: str) -> str:
    return "".join(c for c in text if c.isalnum()).casefold()


def text_overlap(left: str, right: str) -> int:
    a, b = normalized(left), normalized(right)
    for size in range(min(len(a), len(b)), 0, -1):
        if a[-size:] == b[:size]:
            return size
    return 0


def join_text(left: str, right: str, overlap: int) -> str:
    if not overlap:
        separator = " " if left and right and left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum() else ""
        return left + separator + right
    count = 0
    for index, char in enumerate(right):
        if char.isalnum():
            count += 1
        if count == overlap:
            return left.rstrip("，,。.!！?？;； ") + right[index + 1:]
    return left


def merge_chunks(previous: list[Sentence], current: list[Sentence], seam_start: float, previous_end: float) -> list[Sentence]:
    """只在重叠时间内对齐文本；没有身份锚点的声线保留分段命名空间。"""
    previous = [replace(s) for s in previous]
    current = [replace(s) for s in current]
    mapping = {}
    candidates = [s for s in previous if seconds(s.end) > seam_start]
    for new in current:
        if seconds(new.start) >= previous_end:
            continue
        matches = []
        for old in candidates:
            if min(seconds(old.end), seconds(new.end)) <= max(seconds(old.start), seconds(new.start)):
                continue
            a, b = normalized(old.text), normalized(new.text)
            overlap = text_overlap(old.text, new.text)
            if a and b and (a == b or overlap >= 2 or min(len(a), len(b)) >= 2 and (a in b or b in a)):
                matches.append(old.speaker)
        if len(set(matches)) == 1:
            mapping.setdefault(new.speaker, set()).add(matches[0])
    reverse = {}
    for speaker, matches in mapping.items():
        for match in matches:
            reverse.setdefault(match, set()).add(speaker)
    for new in current:
        anchored = mapping.get(new.speaker, set())
        if len(anchored) == 1 and len(reverse[next(iter(anchored))]) == 1:
            new.speaker = next(iter(anchored))
        merged = False
        for index in range(len(previous) - 1, -1, -1):
            old = previous[index]
            if seconds(old.end) <= seam_start:
                continue
            temporal_overlap = min(seconds(old.end), seconds(new.end)) - max(seconds(old.start), seconds(new.start))
            if temporal_overlap <= 0 or new.speaker != old.speaker:
                continue
            a, b = normalized(old.text), normalized(new.text)
            overlap = text_overlap(old.text, new.text)
            if a == b or min(len(a), len(b)) >= 2 and (a in b or b in a):
                text = new.text if len(b) > len(a) else old.text
            elif overlap >= 2:
                text = join_text(old.text, new.text, overlap)
            else:
                continue
            previous[index] = Sentence(timestamp(min(seconds(old.start), seconds(new.start))),
                                       timestamp(max(seconds(old.end), seconds(new.end))), text,
                                       old.speaker, combine_clarity(old.clarity, new.clarity))
            merged = True
            break
        if not merged:
            previous.append(new)
    # 若边界落在句中，合并已经通过重叠台词确认同一声线的连续残句。
    ordered = sorted(previous, key=lambda s: (seconds(s.start), seconds(s.end)))
    joined = []
    for item in ordered:
        old = joined[-1] if joined else None
        if (old and old.speaker == item.speaker and
                seam_start <= seconds(old.end) <= previous_end + 0.1 and
                seam_start <= seconds(item.start) <= previous_end + 0.2 and
                -0.1 <= seconds(item.start) - seconds(old.end) <= 0.2 and
                not re.search(r"[。！？.!?；;][\"'”’）)]*$", old.text)):
            joined[-1] = Sentence(old.start, item.end, join_text(old.text, item.text, 0), old.speaker,
                                  combine_clarity(old.clarity, item.clarity))
        else:
            joined.append(item)
    return joined


def combine_clarity(a: str, b: str) -> str:
    return "重叠" if "重叠" in (a, b) else "听不清" if "听不清" in (a, b) else "清楚"


def records(sentences: list[Sentence], episode: str | int) -> list[dict]:
    match = re.fullmatch(r"E?(\d+)", str(episode), re.IGNORECASE)
    if not match or int(match[1]) < 1:
        raise ValueError("集号须为 E01 或正整数")
    prefix = f"E{int(match[1]):02d}"
    result, speakers = [], {}
    for s in sorted(sentences, key=lambda s: (seconds(s.start), seconds(s.end))):
        if not s.text.strip():
            continue
        start, end = seconds(s.start), seconds(s.end)
        if end <= start or timestamp(start) == timestamp(end) or s.clarity not in ("清楚", "重叠", "听不清"):
            raise ValueError("声音记录的时间或听辨标签无效")
        label = speakers.setdefault(s.speaker, f"{prefix}-V{len(speakers) + 1}")
        result.append({"id": f"{prefix}-S{len(result) + 1:03d}", "起": timestamp(start),
                       "止": timestamp(end), "文本": s.text.strip(), "声线": label, "听辨": s.clarity})
    # 不同声线时间重叠只能标为待核对的重叠候选，不能据此确认真人同时说话。
    for i, a in enumerate(result):
        for b in result[i + 1:]:
            if seconds(b["起"]) >= seconds(a["止"]):
                break
            if a["声线"] != b["声线"]:
                a["听辨"] = b["听辨"] = "重叠"
    return result
