"""单路语音的切句、异常、证据衔接与人工裁定，不调用识别模型。"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from lxml import html
import numpy as np
import pytest

from s2s import config
from s2s.asr import get_backend
from s2s.asr import single
from s2s.diarize import integration
from s2s.diarize.embeddings import unit
from s2s.diarize.pipeline import analyze
from s2s.evidence import evidence_problem, sound_snapshot
from s2s.listen.apply import apply_results
from s2s.listen.migrate import migrate
from s2s.listen import sheet
from s2s.listen.storage import read, write
from s2s.validate import validate_file
from test_t8_dispatch import board, action_result, save
from test_t9b_materials import authorized_data
from s2s.dispatch.materials import assemble


def word(text="台词。", start=.5, end=1.5):
    return {"text": text, "start": start, "end": end}


def sounds():
    return single.segment([word()], [{"起": .4, "止": 1.6}], np.ones(64000))[0]


@pytest.mark.parametrize("provider", ["grok", "doubao", "other"])
def test_remote_backend_unreachable(provider, monkeypatch):
    monkeypatch.setattr(config, "get_environment", lambda: {})
    with pytest.raises(config.ConfigError, match="只允许本地"):
        get_backend(provider)


@pytest.mark.parametrize("change", ["model", "cross_check", "provider"])
def test_old_config_refused(tmp_path, monkeypatch, change):
    data = json.loads(config.DEFAULT_CONFIG.read_text())
    if change == "model":
        data["asr"]["backends"]["local"]["model"] = "paraformer-zh"
    elif change == "cross_check":
        data["asr"]["cross_check"] = {"enabled": True, "primary": "local", "secondary": "grok"}
    else:
        data["asr"]["provider"] = "grok"
    path = tmp_path / "config.json"
    write(path, data)
    monkeypatch.setattr(config, "get_environment", lambda: {"S2S_CONFIG": str(path)})
    with pytest.raises(config.ConfigError):
        get_backend()


def test_normal_single_is_not_agreement_or_human_confirmation():
    row = sounds()[0]
    assert row["文本"] == "台词。"
    assert row["核对"] == {"状态": "单路识别", "原始文本": "台词。", "异常": [], "优先级": 3}
    assert not single.needs_listen(row)
    assert single.update_pending([row], [], "E01") == []


def test_single_keeps_words_outside_vad_in_place_and_flags_time():
    words = [word("甲", .5, 1), word("乙", 2.5, 3)]
    before = deepcopy(words)
    rows, _ = single.segment(words, [{"起": .4, "止": 1.1}], np.ones(64000))
    assert "".join(r["文本"] for r in rows) == "甲乙"
    assert rows[-1]["起"] >= "00:02.3"
    assert "词时间与人声不匹配" in rows[-1]["核对"]["异常"]
    assert rows[-1]["核对"]["状态"] == "待核对"
    assert words == before


def test_truncated_and_overlong_words_never_split_inside_word():
    rows, _ = single.segment([word("没有内部时间的长词", 0, 9)], [{"起": 0, "止": 9}], np.ones(144000))
    assert len(rows) == 1 and rows[0]["截断"]
    assert set(rows[0]["核对"]["异常"]) == {"截断", "异常句长"}
    assert rows[0]["核对"]["优先级"] == 1


def test_missing_voice_is_pending_without_fabricated_text():
    rows, missing = single.segment([word()], [{"起": .4, "止": 1.6}, {"起": 2.5, "止": 3.5}], np.ones(64000))
    assert len(rows) == 1 and missing == [{"起": 2.5, "止": 3.5}]
    pending = single.update_pending(rows, [], "E01", missing)
    assert len(pending) == 1 and "句" not in pending[0]["位置"]
    assert pending[0]["原文"] == "" and "疑似漏段" in pending[0]["依据"]


def test_empty_vad_does_not_discard_asr_words():
    rows, _ = single.segment([word()], [], np.ones(64000))
    assert rows[0]["文本"] == "台词。"
    assert rows[0]["核对"]["状态"] == "待核对"


def test_silence_produces_no_sentence_and_no_identity():
    rows, missing = single.segment([], [], np.zeros(64000))
    assert rows == missing == []


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    episode = tmp_path / "剧名/E01"
    settings = json.loads(config.DEFAULT_CONFIG.read_text())
    backend = SimpleNamespace(settings={}, transcribe=lambda *a: pytest.fail("复用词时不调用 ASR"))
    monkeypatch.setattr(single, "load_config", lambda: settings)
    monkeypatch.setattr(single, "get_backend", lambda: backend)
    monkeypatch.setattr(single.audio, "extract_audio", lambda source, target: target)
    monkeypatch.setattr(single, "waveform", lambda _: np.ones(64000))
    monkeypatch.setattr(single, "detect", lambda *a: [{"起": .4, "止": 1.6}])
    raw = tmp_path / "words.json"
    write(raw, {"provider": "local", "model": "qwen3-asr-0.6b", "response": {"words": [word()]}})
    def fake_run(video, records, **kwargs):
        vectors = kwargs.get("vectors_out")
        if vectors is not None:
            vectors.extend([unit([1, 0])] * len(records))
        return analyze(records, np.ones(64000), lambda _: unit([1, 0]), constraints=kwargs["constraints"], lipsync=kwargs["lipsync"])
    monkeypatch.setattr(integration, "run", fake_run)
    return episode, raw, settings, backend


def test_offline_pipeline_outputs_current_voice_evidence_and_pending(pipeline):
    episode, raw, _, _ = pipeline
    # 字幕故意不可解析：生产识别不得读取。
    write(episode / "待核对.json", [])
    (episode / "字幕记录.json").write_text("无效字幕，不得自动读取")
    result = single.run("video.mp4", episode, word_paths=[raw])
    rows = read(episode / "声音记录.json")
    clustering = read(episode / "声纹聚类.json")
    candidates = read(episode / "声纹候选.json")
    assert result["句数"] == 1 and result["待回听"] == 0
    assert result["统计"]["单路识别"] == 1
    assert clustering["输入可用性"] == {"口型记录": False, "口型问题": "缺失"}
    assert candidates["声音摘要"] == sound_snapshot(rows)
    assert candidates["仅供参考"] and candidates["声线"][0]["候选"] == []
    assert evidence_problem("声纹聚类", clustering, rows) is None
    assert evidence_problem("声纹候选", candidates, rows) is None
    for name in ("声音记录", "声纹聚类", "声纹候选", "待核对"):
        assert validate_file(episode / (name + ".json"))["通过"]
    # 重切保留备份，不修改旧数据后再分析。
    second = single.run("video.mp4", episode, word_paths=[raw])
    assert second["备份"] and read(Path(second["备份"]) / "声音记录.json") == rows


def test_analysis_failure_leaves_board_unchanged(pipeline, monkeypatch):
    episode, raw, _, _ = pipeline
    old = sounds()
    write(episode / "声音记录.json", old)
    def fail(*a, **k):
        raise ValueError("声纹模型缺失")
    monkeypatch.setattr(integration, "analyze_board", fail)
    with pytest.raises(ValueError, match="模型缺失"):
        single.run("video.mp4", episode, word_paths=[raw])
    assert read(episode / "声音记录.json") == old


def test_long_episode_uses_qwen_windows_without_second_backend(pipeline, monkeypatch):
    episode, _, settings, backend = pipeline
    settings["diarize"]["enabled"] = False
    monkeypatch.setattr(single, "waveform", lambda _: np.zeros(601 * 16000))
    monkeypatch.setattr(single, "detect", lambda *a: [])
    calls = []
    def transcribe(path, a, b):
        calls.append((a, b))
        backend.last_words = []
    backend.transcribe = transcribe
    single.run("video.mp4", episode)
    assert calls == [(0, 300), (300, 600), (600, 601)]


def test_non_qwen_raw_refused_before_audio(pipeline, monkeypatch):
    episode, raw, _, _ = pipeline
    write(raw, {"provider": "grok", "words": [word()]})
    monkeypatch.setattr(single.audio, "extract_audio", lambda *a: pytest.fail("无效输入不读音频"))
    with pytest.raises(ValueError, match="只接受 Qwen"):
        single.run("video.mp4", episode, word_paths=[raw])


def test_missing_word_timestamps_never_invented(pipeline):
    episode, raw, _, _ = pipeline
    write(raw, {"response": {"sentences": [{"文本": "只有句级时间"}]}})
    with pytest.raises(ValueError, match="不能从句级"):
        single.run("video.mp4", episode, word_paths=[raw])


@pytest.mark.parametrize("kind", ["口型记录", "声纹聚类", "声纹候选"])
@pytest.mark.parametrize("field", ["文本", "起", "止", "id"])
def test_role_packet_excludes_stale_evidence(board, action_result, kind, field):
    save(board / "派发结果/动作识别/E01-P01.json", action_result)
    rows = read(board / "声音记录.json")
    old = deepcopy(rows)
    old[0][field] = {"文本": "过期内容", "起": "00:00.2", "止": "00:00.8", "id": "E01-S099"}[field]
    value = {"声音摘要": sound_snapshot(old)} if kind == "声纹候选" else {
        "句" if kind == "口型记录" else "句列表": [{"句": r["id"], **{k: r[k] for k in ("文本", "起", "止", "声线")}} for r in old]}
    save(board / (kind + ".json"), value)
    data = authorized_data(assemble("角色卡", board, {}))
    assert kind not in data and data["证据状态"][kind]["状态"] == "过期"
    assert "过期内容" not in data.get(kind, {})


def test_stale_lips_not_used_and_role_count_not_cap(pipeline):
    episode, raw, _, _ = pipeline
    write(episode.parent / "角色卡.json", [{"id": "C001"}])
    write(episode / "口型记录.json", {"句": [{"句": "E01-S001", "文本": "过期"}]})
    single.run("video.mp4", episode, word_paths=[raw])
    result = read(episode / "声纹聚类.json")
    assert not result["输入可用性"]["口型记录"]
    assert result["约束"]["参数"]["constraints"] == ["lipsync"]
    assert "max_speakers" not in result["约束"]["参数"]


@pytest.mark.parametrize("choice", ["主路", "副路"])
def test_old_dual_decision_never_auto_confirms_single(choice):
    rows = sounds()
    old = deepcopy(rows)
    old[0].pop("核对")
    decision = {"句号": old[0]["id"], "决定": choice, "最终文本": old[0]["文本"], "听辨": "清楚", "备注": ""}
    migrated, report = migrate(rows, {"声音快照": old, "结果": [decision]})
    assert migrated[0]["核对"]["状态"] == "单路识别"
    assert report[0]["状态"] == "待确认"


def test_changed_text_never_inherits_human_confirmation():
    old = sounds()
    new = deepcopy(old)
    new[0]["文本"] = "改了原文"
    decision = {"句号": old[0]["id"], "决定": "手改", "最终文本": "人工听到", "听辨": "清楚", "备注": ""}
    migrated, report = migrate(new, {"声音快照": old, "结果": [decision]})
    assert migrated[0]["文本"] == "改了原文" and report[0]["状态"] == "待确认"


def test_single_page_has_no_secondary_and_applies_original(tmp_path, monkeypatch):
    episode = tmp_path / "E01"
    rows = sounds()
    single.flag(rows[0], "口型冲突", 1)
    pending = single.update_pending(rows, [], "E01")
    write(episode / "声音记录.json", rows)
    write(episode / "待核对.json", pending)
    write(episode / "镜头表.json", {"视频": "fixture.mp4", "镜头": []})
    write(episode / "素材/回听/疑似漏段.json", [{"起": 2.5, "止": 3.5}])
    monkeypatch.setattr(sheet.audio, "duration", lambda _: 4)
    monkeypatch.setattr(sheet, "clip", lambda *a: None)
    path = tmp_path / "回听.html"
    assert sheet.build_sheet(episode, path)["句数"] == 1
    tree = html.fromstring(path.read_text())
    assert tree.xpath('//button[@data-choice="原文"]')
    assert not tree.xpath('//button[@data-choice="副路" or @data-choice="主路"]')
    assert "口型冲突" in tree.text_content() and "疑似漏段 1" in tree.text_content()
    data = json.loads(tree.xpath('//script[@id="listen-data"]')[0].text)
    assert "副路" not in data["items"][0]
    decision = {"句号": rows[0]["id"], "决定": "原文", "最终文本": rows[0]["文本"], "听辨": "清楚", "备注": "听过"}
    assert apply_results(episode, {"声音快照": rows, "结果": [decision]})["处理句数"] == 1
    updated = read(episode / "声音记录.json")
    assert updated[0]["核对"]["状态"] == "人工确认"
    assert read(episode / "待核对.json")[0]["状态"] == "已裁定"


def test_single_javascript_original_choice_and_unavailable_secondary():
    path = Path("tools/s2s/listen/sheet.js").resolve()
    code = f'''const assert=require('node:assert/strict'), api=require({json.dumps(str(path))});
const item={{句号:'E01-S001',原文:'原句',字幕:''}};
const d={{句号:item.句号,决定:'原文',最终文本:'原句',听辨:'清楚',备注:''}};
assert(api.valid(item,d));
assert(!api.valid(item,{{...d,决定:'副路'}}));
assert(!api.valid(item,{{...d,最终文本:'改过'}}));'''
    result = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_cross_cli_fails_without_loading_audio():
    result = subprocess.run([".venv/bin/python", "tools/asr_cross.py", "missing.mp4", "missing/E01"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "双路 ASR 已停用" in result.stderr


def test_role_packet_rebinds_lipsync_pairs_after_voice_reclustering(board, action_result):
    save(board / "派发结果/动作识别/E01-P01.json", action_result)
    sound = read(board / "声音记录.json")[0]
    row = {"句": sound["id"], **{k: sound[k] for k in ("文本", "起", "止")},
           "声线": "E01-V99", "结论": "画内", "轨迹": [{"轨迹组": "G1", "状态": "开口"}]}
    old_pairs = [{"声线": "E01-V99", "轨迹组": "G1", "支持句数": 1, "反例句数": 0}]
    save(board / "口型记录.json", {"句": [row], "声线配对": old_pairs})
    packet = authorized_data(assemble("角色卡", board, {}))
    assert packet["口型记录"]["句"][0]["声线"] == sound["声线"]
    assert packet["口型记录"]["声线配对"] == [{**old_pairs[0], "声线": sound["声线"]}]
    assert read(board / "口型记录.json")["声线配对"] == old_pairs


def test_lipsync_uncertainty_generates_pending_and_excludes_candidates(pipeline):
    episode, raw, _, _ = pipeline
    rows = sounds()
    lips = [{"句": r["id"], **{k: r[k] for k in ("文本", "起", "止")},
             "结论": "存疑", "轨迹": []} for r in rows]
    write(episode / "口型记录.json", {"句": lips})
    result = single.run("video.mp4", episode, word_paths=[raw])
    assert result["待回听"] == 1
    updated = read(episode / "声音记录.json")
    assert "口型冲突或存疑" in updated[0]["核对"]["异常"]
    assert read(episode / "声纹候选.json")["声线"][0]["句数"] == 0
    assert "Grok" not in json.dumps(read(episode / "声纹聚类.json"), ensure_ascii=False)


def test_migrated_correction_is_used_by_cluster_and_candidate_snapshot(pipeline):
    episode, raw, _, _ = pipeline
    old = sounds()
    write(episode / "声音记录.json", old)
    decision = {"句号": old[0]["id"], "决定": "手改", "最终文本": "亲耳听到的原声", "听辨": "清楚", "备注": ""}
    path = episode / "回听结果.json"
    write(path, {"声音快照": old, "结果": [decision]})
    single.run("video.mp4", episode, word_paths=[raw], decisions_path=path)
    updated = read(episode / "声音记录.json")
    assert updated[0]["文本"] == "亲耳听到的原声" and updated[0]["核对"]["状态"] == "人工确认"
    assert evidence_problem("声纹聚类", read(episode / "声纹聚类.json"), updated) is None
    assert evidence_problem("声纹候选", read(episode / "声纹候选.json"), updated) is None


@pytest.mark.parametrize("uncertainty", ["声纹待定", "句内换人或重叠"])
def test_uncertain_voice_never_becomes_identity_or_candidate(pipeline, monkeypatch, uncertainty):
    episode, raw, _, _ = pipeline
    def fake_run(video, records, **kwargs):
        marked = deepcopy(records)
        if uncertainty == "句内换人或重叠":
            marked[0]["听辨"] = "重叠"
        vector = None if uncertainty == "声纹待定" else unit([1, 0])
        kwargs["vectors_out"].extend([vector] * len(records))
        return analyze(marked, np.ones(64000), lambda _: vector, constraints=kwargs["constraints"])
    monkeypatch.setattr(integration, "run", fake_run)
    result = single.run("video.mp4", episode, word_paths=[raw])
    assert result["待回听"] == 1
    row = read(episode / "声音记录.json")[0]
    assert row["核对"]["状态"] == "待核对" and uncertainty in row["核对"]["异常"]
    assert read(episode / "待核对.json")[0]["级别"] == "阻断"
    assert read(episode / "声纹候选.json")["声线"][0]["候选"] == []
    assert not (episode / "归属表.json").exists()
    assert validate_file(episode / "声纹聚类.json")["通过"]
