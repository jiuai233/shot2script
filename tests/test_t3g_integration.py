import json

import numpy as np
import pytest

from s2s.asr import crosscheck
from s2s.diarize import integration
from s2s.diarize.embeddings import unit
from s2s.diarize.pipeline import analyze
from s2s.listen.storage import read, write
from s2s.validate import validate_file


def setup(tmp_path, monkeypatch, enabled=True):
    settings = {"asr":{"cross_check":{"enabled":True,"primary":"local","secondary":"grok"}},
                "diarize":{"enabled":enabled,"constraints":[]}}
    monkeypatch.setattr(crosscheck,"load_config",lambda: settings)
    monkeypatch.setattr(crosscheck,"get_backend",lambda *_: pytest.fail("不得新增识别调用"))
    row = {"id":"E01-S001","起":"00:00.1","止":"00:00.9","文本":"台词。","声线":"E01-V9","听辨":"清楚"}
    write(tmp_path/"a.json",[row]);write(tmp_path/"b.json",[row])
    write(tmp_path/"words.json",{"words":[{"text":"台词。","start":.1,"end":.9}]})
    from s2s import vad
    monkeypatch.setattr(crosscheck.audio,"extract_audio",lambda _source,target:target)
    monkeypatch.setattr(vad,"detect",lambda *args:[])
    monkeypatch.setattr(vad,"waveform",lambda _:np.ones(16000))
    return tmp_path/"E01", settings


def test_cross_automatic_step_keeps_existing_fields(tmp_path,monkeypatch):
    episode, settings = setup(tmp_path,monkeypatch)
    calls = []
    def fake_run(video, sounds, **kwargs):
        calls.append(kwargs)
        return analyze(sounds,np.ones(16000),lambda _:unit([1,0]),constraints=kwargs["constraints"])
    monkeypatch.setattr(integration,"run",fake_run)
    result = crosscheck.run("not-required-by-stub.wav",episode,tmp_path/"a.json",tmp_path/"b.json", primary_words=tmp_path/"words.json", secondary_words=tmp_path/"words.json")
    row = read(episode/"声音记录.json")[0]
    from s2s.asr.segment import segment_pair
    expected = segment_pair([{"text":"台词。","start":.1,"end":.9}],
                            [{"text":"台词。","start":.1,"end":.9}], [], np.ones(16000))[0][0]
    assert row == {**expected,"声线":"E01-V1"}
    assert result["声纹"]["类数"] == 1
    assert "max_speakers" not in calls[0]["constraints"]
    assert (episode/"声纹聚类.json").exists()
    assert validate_file(episode/"声音记录.json")["通过"]


def test_disabled_diarize_still_uses_audio_segmentation(tmp_path,monkeypatch):
    episode,_ = setup(tmp_path,monkeypatch,False)
    monkeypatch.setattr(integration,"analyze_board",lambda *a:pytest.fail("禁用时不得运行声纹"))
    result = crosscheck.run("missing",episode,tmp_path/"a.json",tmp_path/"b.json", primary_words=tmp_path/"words.json", secondary_words=tmp_path/"words.json")
    assert "声纹" not in result and not (episode/"声纹聚类.json").exists()


def test_failed_analysis_does_not_write_board(tmp_path,monkeypatch):
    episode,_ = setup(tmp_path,monkeypatch)
    def fail(*args):raise ValueError("声纹模型不存在")
    monkeypatch.setattr(integration,"analyze_board",fail)
    with pytest.raises(ValueError,match="模型不存在"):
        crosscheck.run("missing",episode,tmp_path/"a.json",tmp_path/"b.json", primary_words=tmp_path/"words.json", secondary_words=tmp_path/"words.json")
    assert not (episode/"声音记录.json").exists()


def test_role_count_does_not_cap_new_episode(tmp_path, monkeypatch):
    episode, _ = setup(tmp_path, monkeypatch)
    write(tmp_path / "角色卡.json", [{"id": "C001"}, {"id": "C002"}])
    sounds = read(tmp_path / "a.json")
    write(episode / "口型记录.json", {"句": [{"句": r["id"], **{k: r[k] for k in ("起", "止", "文本")}} for r in sounds]})
    def fake_run(video, records, **kwargs):
        assert "max_speakers" not in kwargs["constraints"]
        assert kwargs["words"] == ()
        assert kwargs["lipsync"] is not None
        return analyze(records, np.ones(16000), lambda _: unit([1, 0]))
    monkeypatch.setattr(integration, "run", fake_run)
    _, result = integration.analyze_board("missing", sounds, episode, {})
    assert result["输入可用性"]["口型记录"]
