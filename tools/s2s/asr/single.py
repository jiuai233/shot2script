"""Qwen 单路：真实词时间切句、音频异常、声纹候选与回听清单。"""

from copy import deepcopy
from pathlib import Path
import tempfile

from .. import audio
from ..config import load_config
from ..listen.storage import read, save_board, write
from ..listen.snapshot import summary
from ..timefmt import fmt
from ..vad import detect, waveform
from . import get_backend
from .postprocess import records, seconds
from .segment import assign, boundaries, load_words, missing_voice, prepare_words, voice_intervals

STATES = ("单路识别", "待核对", "人工确认")


def needs_listen(row):
    check = row.get("核对", {})
    if check.get("状态") == "人工确认":
        return False
    return (check.get("状态") != "单路识别" or bool(check.get("异常")) or
            row.get("听辨") != "清楚" or bool(row.get("截断")))


def flag(row, reason, priority=2):
    check = row["核对"]
    if reason not in check["异常"]:
        check["异常"].append(reason)
    check.update({"状态": "待核对", "优先级": min(priority, check["优先级"])})


def segment(words, voices, samples, episode="E01", *, min_pause=.5, max_sentence=6):
    duration = len(samples) / 16000
    words = prepare_words(words, duration)
    # 单路缺少可区分 ASR 错时与 VAD 漏检的证据，保留真实词时间，不挪词。
    spans = boundaries(voices, [words], samples, min_pause=min_pause, max_sentence=max_sentence)
    rows = assign(words, spans, episode)
    voice_spans = voice_intervals(voices, duration)
    for i, row in enumerate(rows, 1):
        row.update(id=f"{episode}-S{i:03d}", 声线=f"{episode}-V{i}")
        row["核对"] = {"状态": "单路识别", "原始文本": row["文本"], "异常": [], "优先级": 3}
        a, b = seconds(row["起"]), seconds(row["止"])
        inside = [w for w in words if a <= (w["start"] + w["end"]) / 2 < b]
        if b - a < .5 - 1e-8 or b - a > max_sentence + 1e-8:
            flag(row, "异常句长", 1)
        if (a == 0 and any(left <= .01 for left, _ in voice_spans) or
                b >= duration - .1 and any(duration - right <= .02 for _, right in voice_spans)):
            row["截断"] = True
            flag(row, "截断", 1)
        if any(not any(min(w["end"], right) > max(w["start"], left) for left, right in voice_spans)
               for w in inside):
            flag(row, "词时间与人声不匹配", 1)
        if any(min(left["end"], right["end"]) > max(left["start"], right["start"]) + .05
               for left, right in zip(inside, inside[1:])):
            flag(row, "词时间重叠", 1)
    return rows, missing_voice(voices, words, [])


def update_pending(sounds, existing, episode, missing=(), migrations=()):
    # 识别问题重建，其他职责问题继续保留；旧句号按时间重定向，不能静默指向新句。
    pending = deepcopy(existing)
    number = max((int(r["id"].split("-Q")[1]) for r in pending), default=0)
    for issue in pending:
        if issue["来源"] == "识别" and issue["处理职责"] == "回听":
            issue.update(状态="已裁定", 复核结果="旧切句问题已归档，按当前单路问题回听。")
    prior = {m["新句号"]: m for m in migrations if m.get("新句号") and m["状态"] == "待确认"}
    for row in sounds:
        if row["id"] in prior:
            flag(row, "原裁定待确认")
        if not needs_listen(row):
            continue
        number += 1
        evidence = "；".join(row["核对"]["异常"]) or "听辨或身份存疑"
        if row["id"] in prior:
            evidence += "\n原裁定：" + prior[row["id"]]["原裁定"]
        pending.append(_issue(episode, number, {"句": row["id"], "时间": row["起"]}, row["文本"], evidence))
    for span in missing:
        number += 1
        pending.append(_issue(episode, number, {"时间": fmt(seconds(span["起"]))}, "",
                              f'疑似漏段：{fmt(seconds(span["起"]))}—{fmt(seconds(span["止"]))} 有人声但无识别词；不按字幕补字。'))
    return pending


def _issue(episode, number, location, text, evidence):
    return {"id": f"{episode}-Q{number:03d}", "来源": "识别", "位置": location,
            "原文": text, "类型": "全文完整性与一致性", "级别": "阻断", "相关人物": [],
            "依据": evidence, "修正要求": "回听对应时间及上下文，核对真实台词或身份；无文本时人工打点补录。",
            "处理职责": "回听", "状态": "待核对", "复核结果": None}


def run(video, episode_dir, *, word_paths=(), voice_path=None, decisions_path=None,
        min_pause=None, max_sentence=None):
    directory = Path(episode_dir)
    records([], directory.name)
    configuration = load_config()
    backend = get_backend()  # 即使离线重切，也拒绝旧生产配置。
    if (directory / "声音记录.json").exists():
        previous = read(directory / "声音记录.json")
    else:
        previous = []
    existing = read(directory / "待核对.json") if (directory / "待核对.json").exists() else []
    decisions = read(decisions_path) if decisions_path else None
    if decisions and summary(decisions.get("声音快照", [])) != summary(previous):
        raise ValueError("原裁定快照不匹配当前声音记录，停止迁移")
    for path in word_paths:
        raw = read(path)
        if isinstance(raw, dict) and (raw.get("provider", "local") != "local" or
                                     raw.get("model", "qwen3-asr-0.6b") != "qwen3-asr-0.6b"):
            raise ValueError("只接受 Qwen 单路真实词级响应，不接受其他后端结果")
    words = [w for p in word_paths for w in load_words(p)] if word_paths else None
    voices = read(voice_path) if voice_path else None
    settings = configuration["asr"].get("segmentation", {})
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".asr-single-", dir=directory) as folder:
        wav = audio.extract_audio(video, Path(folder) / "source.wav")
        samples = waveform(wav)
        duration = len(samples) / 16000
        voices = detect(wav, settings.get("vad", {})) if voices is None else voices
        voice_intervals(voices, duration)  # 输入无效时不得先调用 ASR。
        if words is None:
            backend.settings["raw_dir"] = str(directory / "素材/asr_raw")
            words = []
            # Qwen 强制对齐单次至多 300 秒；窗口不重叠，避免造重复词。
            for a, b in audio.chunks(0, duration, 300, overlap=0):
                backend.transcribe(wav, a, b)
                if getattr(backend, "last_words", None) is None:
                    raise ValueError("Qwen 未返回真实词级时间，不能从句级时间推造")
                words.extend(backend.last_words)
        sounds, missing = segment(words, voices, samples, directory.name,
                                  min_pause=min_pause if min_pause is not None else settings.get("min_pause", .5),
                                  max_sentence=max_sentence if max_sentence is not None else settings.get("max_sentence", 6))
    migrations = []
    if decisions:
        from ..listen.migrate import migrate
        sounds, migrations = migrate(sounds, decisions)
    prior_ids = {m["新句号"] for m in migrations if m.get("新句号") and m["状态"] == "待确认"}
    for row in sounds:
        if row["id"] in prior_ids:
            flag(row, "原裁定待确认")
    clustering, candidates = None, None
    diarize_settings = configuration.get("diarize", {})
    if diarize_settings.get("enabled") is True:
        from ..diarize.integration import analyze_board
        from ..voiceprint.library import compare, model_info
        vectors = []
        sounds, clustering = analyze_board(video, sounds, directory, diarize_settings, vectors_out=vectors)
        for row, prediction in zip(sounds, clustering["句列表"]):
            if prediction["待定"]:
                flag(row, "声纹待定")
            if prediction["疑似重叠"]:
                flag(row, "句内换人或重叠", 1)
            for conflict in clustering.get("约束", {}).get("冲突", []):
                if row["id"] in conflict.get("句", []):
                    flag(row, conflict["类型"], 1)
        candidates = compare(directory.parent, directory.name,
                             extracted=(sounds, vectors, model_info(diarize_settings.get("model", "campp"))),
                             clustering=clustering, persist=False)
    # 输入改变后，其他职责旧问题保留时间定位，不能沿用可能已指向另一句话的句号。
    old_by_id = {row["id"]: row for row in previous}
    for issue in existing:
        old = old_by_id.get(issue["位置"].get("句"))
        if old:
            exact = next((r for r in sounds if all(r[k] == old[k] for k in ("文本", "起", "止"))), None)
            if exact:
                issue["位置"]["句"] = exact["id"]
            else:
                issue["位置"].pop("句", None)
                issue["位置"]["时间"] = old["起"]
                issue["依据"] += f'\n原句 {old["id"]} 已变更，请按时间重新核对。'
    pending = update_pending(sounds, existing, directory.name, missing, migrations)
    for name in ("素材/人声段.json", "素材/回听/裁定迁移.json", "素材/回听/疑似漏段.json",
                 "声纹聚类.json", "声纹候选.json"):
        if (directory / name).exists():
            read(directory / name)
    saved = save_board(directory, sounds, pending, "单路识别")
    write(directory / "素材/人声段.json", voices)
    write(directory / "素材/回听/裁定迁移.json", migrations)
    write(directory / "素材/回听/疑似漏段.json", missing)
    if clustering is not None:
        write(directory / "声纹聚类.json", clustering)
        write(directory / "声纹候选.json", candidates)
    return {"后端": "local", "模型": "qwen3-asr-0.6b", "句数": len(sounds),
            "待回听": sum(needs_listen(r) for r in sounds), "疑似漏段": missing,
            "统计": {s: sum(r["核对"]["状态"] == s for r in sounds) for s in STATES}, **saved}
