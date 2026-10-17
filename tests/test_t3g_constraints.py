import json
import math

import numpy as np
import pytest

from s2s.diarize.constraints import constrained_assign, evidence, options
from s2s.diarize.embeddings import unit
from s2s.diarize.integration import apply_labels, discover_words
from s2s.diarize.pipeline import analyze


def records(n=3):
    return [{"id": f"E01-S{i+1:03d}", "起": f"00:0{i}.0", "止": f"00:0{i+1}.0", "文本": f"第{i}句。", "声线": "E01-V9", "听辨": "清楚"} for i in range(n)]


def lips(rows, groups):
    return {"句": [{"句": r["id"], **{k: r[k] for k in ("起", "止", "文本")}, "结论": "画内",
                    "轨迹": [{"状态": "开口", "轨迹组": g} for g in gs]} for r, gs in zip(rows, groups)]}


def test_soft_grok_merge_can_be_disabled():
    rows = records(2)
    vectors = [unit([1, 0]), unit([0.55, math.sqrt(1-.55**2)])]
    words = [{"start": i, "end": i+1, "text": "词", "speaker": 1} for i in range(2)]
    enabled = constrained_assign(vectors, [True]*2, .6, rows, words, settings={"constraints": ["grok"]})
    disabled = constrained_assign(vectors, [True]*2, .6, rows, words, settings={"constraints": []})
    assert len(enabled[2]) == 1
    assert "grok" in enabled[3][0]
    assert len(disabled[2]) == 2


def test_grok_not_hard_link_and_scope_independent():
    rows = records(2)
    words = [{"start": i, "end": i+1, "speaker": 1, "_scope": str(i)} for i in range(2)]
    grok, *_ = evidence(rows, words, None, {})
    assert grok[0] != grok[1]
    output = constrained_assign([unit([1, 0]), unit([0, 1])], [True]*2, .6, rows, words)
    assert len(output[2]) == 2


def test_low_purity_grok_sentence_abstains():
    words = [{"start": 0, "end": .5, "speaker": 1}, {"start": .5, "end": 1, "speaker": 2}]
    assert evidence(records(1), words, None, {})[0] == {}


def test_must_link_and_short_seed():
    rows = records(2)
    output = constrained_assign([unit([1, 0]), unit([0, 1])], [True, False], .6, rows,
                                lipsync=lips(rows, [["G1"], ["G1"]]), settings={"constraints": ["lipsync"]})
    assert output[0] == {0: 0, 1: 0}
    assert "lipsync" in output[3][0]


def test_short_must_links_form_class_without_long_seed():
    rows = records(2)
    output = constrained_assign([unit([1, 0]), unit([0, 1])], [False]*2, .6, rows,
                                lipsync=lips(rows, [["G1"], ["G1"]]), settings={"constraints": ["lipsync"]})
    assert output[0] == {0: 0, 1: 0}


def test_cannot_link_survives_cap_and_short_attach():
    rows = records(3)
    rows[1]["起"], rows[1]["止"] = "00:00.0", "00:01.0"
    rows[2]["起"], rows[2]["止"] = "00:00.0", "00:01.0"
    output = constrained_assign([unit([1, 0])]*3, [True]*3, .6, rows,
                                lipsync=lips(rows, [["A"], ["B"], ["C"]]), settings={"max_speakers": 1})
    assert len(output[2]) == 3
    assert output[4]["冲突"][-1]["类型"] == "人数上限与cannot-link冲突"
    output = constrained_assign([unit([1, 0])]*3, [True, False, True], .6, rows,
                                lipsync=lips(rows, [["A"], ["B"], ["C"]]))
    assert 1 not in output[0]


def test_conflicting_hard_links_recorded(monkeypatch):
    monkeypatch.setattr("s2s.diarize.constraints.evidence", lambda *a: ({}, [(0, 1, "A"), (1, 2, "A")], {(0, 2)}, []))
    output = constrained_assign([unit([1, 0])]*3, [True]*3, .6, records())
    assert output[0][0] != output[0][2]
    assert output[4]["冲突"][0]["类型"] == "must-link/cannot-link"


def test_upper_bound_plus_unknown():
    vectors = [unit([1, 0, 0]), unit([0, 1, 0]), unit([0, 0, 1])]
    output = constrained_assign(vectors, [True]*3, .6, records(), settings={"constraints": ["max_speakers"], "max_speakers": 1})
    assert len(output[2]) == 2
    assert any("max_speakers" in r for r in output[3].values())


def test_stale_and_ambiguous_lipsync_not_merged():
    rows = records(2)
    stale = lips(rows, [["G1"], ["G1"]])
    stale["句"][1]["文本"] = "旧句"
    output = constrained_assign([unit([1, 0]), unit([0, 1])], [True]*2, .6, rows, lipsync=stale)
    assert len(output[2]) == 2
    assert output[4]["冲突"][0]["类型"] == "口型输入不匹配"
    ambiguous = lips(rows, [["G1", "G2"], ["G1"]])
    assert evidence(rows, [], ambiguous, {})[-1][0]["类型"] == "单句多轨迹开口"


@pytest.mark.parametrize("settings", [{"constraints": ["unknown"]}, {"grok_weight": -1}, {"grok_purity": float("nan")}, {"grok_min_similarity": 2}, {"max_speakers": 0}, {"max_speakers": True}])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        options(settings)


def test_result_metadata_representatives_and_labels_only():
    rows = records(2)
    result = analyze(rows, np.ones(32000), lambda _: unit([1, 0]), constraints={"constraints": []})
    assert len(result["类别"][0]["代表句"]) == 2
    assert result["句列表"][0]["置信度"] == 1
    updated = apply_labels(rows, result)
    for old, new in zip(rows, updated):
        assert {k:v for k,v in old.items() if k != "声线"} == {k:v for k,v in new.items() if k != "声线"}
    result["句列表"][0]["待定"] = True
    assert apply_labels(rows, result)[0] == rows[0]
    with pytest.raises(ValueError):
        apply_labels(rows[:1], result)


def test_discovery_latest_full_not_retry_or_window(tmp_path):
    for name, stamp, window, speaker, status in [("old", "1", [0, 10], 1, 200), ("new", "2", [0, 10], 2, 200), ("failed", "3", [0,10], 3, 500), ("small", "4", [2,4], 4, 200)]:
        data = {"provider": "grok", "time": stamp, "window": window, "status": status,
                "response": {"words": [{"text": "你好", "start": 2.5 if name != "small" else .5, "end": 3 if name != "small" else 1, "speaker": speaker}]}}
        (tmp_path/(name+".json")).write_text(json.dumps(data))
    (tmp_path/"recovery.json").write_text("[]")
    words = discover_words([tmp_path])
    assert len(words) == 1 and words[0]["speaker"] == 2
