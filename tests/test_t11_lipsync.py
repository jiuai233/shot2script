import copy
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from s2s.lipsync import Parameters, evaluate
from s2s.lipsync.core import (attach_detections, frame_indices, group_tracks,
                             pair_voices, score_samples, sentence_conclusion)
from s2s.lipsync.pipeline import Envelope, load_inputs, review_sentence, timecode


def face(x=.1, valid=True, feature=0, color=0):
    hist = np.zeros(48)
    hist[color] = 1
    return {"box": [x, .2, .2, .3], "mar": .1, "valid": valid, "reason": [],
            "feature": np.full(20, feature), "appearance": hist}


def sample(time, mar, valid=True):
    return {**face(valid=valid), "time": time, "mar": mar, "path": f"{time}.jpg"}


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"threshold": 1}, {"margin": .5},
                                   {"sync_weight": -1}, {"motion_scale": 0}, {"fps": float('nan')},
                                   {"confidence": 2}, {"max_yaw": 90}, {"max_faces": 0}])
def test_parameters_reject_invalid(kwargs):
    with pytest.raises(ValueError):
        Parameters(**kwargs)


def test_frame_sampling_keeps_12fps_without_crossing_cut():
    first = frame_indices(0, 1, 12, 30, 90)
    second = frame_indices(1, 2, 12, 30, 90)
    assert len(first) == len(second) == 12
    assert set(first).isdisjoint(second)
    assert max(first) < 30 <= min(second)
    indices = frame_indices(1.02, 1.19, 12, 30, 90)
    assert all(1.02 <= i / 30 < 1.19 for i in indices)
    assert timecode(1 / 30) == "00:00.033"


def test_track_one_to_one_and_shot_reset():
    params = Parameters()
    tracks = []
    attach_detections(tracks, [face(.1), face(.6)], 0, "a.jpg", "E01-K001", params)
    attach_detections(tracks, [face(.61), face(.11)], .1, "b.jpg", "E01-K001", params)
    assert len(tracks) == 2
    assert tracks[0]["samples"][-1]["box"][0] == .11
    assert tracks[1]["samples"][-1]["box"][0] == .61
    attach_detections(tracks, [face(.1)], 1, "c.jpg", "E01-K001", params)
    assert len(tracks) == 3
    other = []
    attach_detections(other, [face(.1)], 1.1, "d.jpg", "E01-K002", params)
    assert other[0]["id"] == "E01-K002-F1"


def test_group_uses_landmarks_and_color_without_merging_same_shot():
    tracks = [{"id": str(i), "shot": shot, "samples": [sample(0, .1) | face(feature=feat, color=color)]}
              for i, (shot, feat, color) in enumerate([("K1", 0, 0), ("K1", 0, 0), ("K2", .01, 0), ("K3", .8, 0), ("K4", 0, 3)])]
    group_tracks(tracks, Parameters())
    assert tracks[0]["group"] != tracks[1]["group"]
    assert tracks[2]["group"] == tracks[0]["group"]
    assert tracks[3]["group"] not in [tracks[0]["group"], tracks[1]["group"]]
    assert tracks[4]["group"] != tracks[0]["group"]


def test_score_speech_static_chewing_noise_and_short_samples():
    p = Parameters()
    samples = [sample(i / 12, .03 + .25 * (i % 2)) for i in range(12)]
    envelope = lambda t: .03 + .25 * (round(t * 12) % 2)
    high, status, evidence = score_samples(samples, [.03] * 8, envelope, p)
    assert high > .9 and status == "开口" and evidence["静音基线帧"] == 8
    static = [sample(i / 12, .03) for i in range(12)]
    assert score_samples(static, [.03] * 8, envelope, p)[:2] == (0, "未开口")
    baseline = [.03, .28] * 4
    assert score_samples(samples, baseline, envelope, p)[1] == "未开口"
    invisible = [sample(i / 12, .1, False) for i in range(6)]
    assert score_samples(invisible, [], envelope, p)[1] == "看不到"
    assert score_samples(samples[:3], [], envelope, p)[1] == "存疑"
    assert score_samples(samples, [], envelope, p)[2]["基线来源"].startswith("固定噪声")
    assert score_samples(samples, [.03] * 8, lambda _: 0, p)[1] == "存疑"


@pytest.mark.parametrize("states,expected", [([], "看不到"), (["看不到"], "看不到"),
    (["未开口"], "未开口"), (["未开口", "存疑"], "存疑"), (["开口", "未开口"], "画内")])
def test_four_conclusions(states, expected):
    assert sentence_conclusion([{"状态": x} for x in states]) == expected


def test_voice_pairs_count_once_per_sentence_and_dont_count_offscreen_as_negative():
    def row(conclusion, states):
        return {"声线": "E01-V1", "结论": conclusion,
                "轨迹": [{"轨迹组": group, "状态": state} for group, state in states]}
    rows = [row("画内", [("G1", "开口"), ("G1", "开口"), ("G2", "未开口")]),
            row("画内", [("G2", "开口"), ("G1", "未开口")]),
            row("未开口", [("G1", "未开口")])]
    assert pair_voices(rows) == [{"声线": "E01-V1", "轨迹组": "G1", "支持句数": 1, "反例句数": 1},
                                {"声线": "E01-V1", "轨迹组": "G2", "支持句数": 1, "反例句数": 1}]


def test_review_default_configuration_and_no_override():
    calls = []
    def ask(images, prompt, **kwargs):
        calls.append((images, prompt, kwargs))
        return "左侧人物嘴部在动"
    row = {"句": "E01-S001", "起": "00:01.0", "止": "00:02.0", "文本": "你好", "结论": "存疑"}
    original = copy.deepcopy(row)
    result = review_sentence(row, [str(i) for i in range(12)], ask)
    assert len(calls[0][0]) == 6 and calls[0][2] == {}
    assert result["结果"] and row == original
    assert review_sentence(row, ["a", "b"], ask)["状态"] == "未调用"
    row["结论"] = "画内"
    assert review_sentence(row, [str(i) for i in range(12)], ask) is None
    assert len(calls) == 1


def test_input_snapshot_does_not_modify_sources(tmp_path):
    board, snapshot = tmp_path / "board", tmp_path / "snapshot"
    board.mkdir(); snapshot.mkdir()
    shots = {"集": 1, "时长": "00:02.0", "镜头": [{"id": "E01-K001", "起": "00:00.0", "止": "00:02.0"}]}
    lines = [{"id": "E01-S001", "起": "00:00.2", "止": "00:01.0", "文本": "你好"}]
    for name, value in [("镜头表.json", shots), ("声音记录.json", lines)]:
        (board / name).write_text(json.dumps(value))
    actual_shots, actual_lines, versions = load_inputs(board, snapshot)
    assert actual_shots == shots and actual_lines == lines and len(versions) == 2
    assert (board / "声音记录.json").read_bytes() == (snapshot / "声音记录.json").read_bytes()
    lines[0]["止"] = "00:00.2"
    (board / "声音记录.json").write_text(json.dumps(lines))
    with pytest.raises(ValueError):
        load_inputs(board, snapshot)


def test_audio_envelope(tmp_path):
    path = tmp_path / "a.wav"
    samples = np.r_[np.zeros(1600), np.full(1600, 16000)].astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(samples.tobytes())
    envelope = Envelope(path)
    assert envelope(.05) == 0 and .48 < envelope(.15) < .50
    assert envelope(-1) == envelope(3) == 0


def test_evaluation_perfect_and_known_errors():
    annotation = {"校对完成": True, "台词": [
        {"id": "L1", "起": "00:00.0", "止": "00:01.0", "形态": "画内", "说话人": "C001"},
        {"id": "L2", "起": "00:01.0", "止": "00:02.0", "形态": "画外", "说话人": None},
        {"id": "L3", "起": "00:02.0", "止": "00:03.0", "形态": "心声", "说话人": None}]}
    prediction = {"句": [
        {"句": "E01-S001", "起": "00:00.0", "止": "00:01.0", "结论": "画内", "轨迹": [{"状态": "开口", "轨迹组": "G1"}]},
        {"句": "E01-S002", "起": "00:01.0", "止": "00:02.0", "结论": "未开口", "轨迹": []},
        {"句": "E01-S003", "起": "00:02.0", "止": "00:03.0", "结论": "未开口", "轨迹": []}]}
    result = evaluate(prediction, annotation, {"G1": "C001"})
    assert result["画内性准确率"] == result["说话人准确率"] == 1
    prediction["句"][0]["结论"] = "存疑"
    prediction["句"][1]["结论"] = "画内"
    wrong = evaluate(prediction, annotation, {"G1": "C002"})
    assert wrong["画内性准确率"] == 1 / 3
    assert wrong["画内性覆盖率"] == 2 / 3
    assert wrong["说话人准确率"] == 0 and len(wrong["错误"]) == 3
    annotation["校对完成"] = False
    with pytest.raises(ValueError, match="校对"):
        evaluate(prediction, annotation)


def test_evaluation_missing_predictions_and_missing_mapping_count_as_failures():
    annotation = {"校对完成": True, "台词": [{"id": "L1", "起": "00:00.0", "止": "00:01.0", "形态": "画内", "说话人": "C001"}]}
    result = evaluate({"句": []}, annotation)
    assert result["画内性准确率"] == result["说话人准确率"] == 0
    assert result["画内性覆盖率"] == result["说话人覆盖率"] == 0
