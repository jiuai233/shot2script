import copy
import json
from pathlib import Path

import pytest

from s2s.sub_align import compare, comparison_normalize, differences, main


CASES = json.loads((Path(__file__).parent / "fixtures/t4/t4b_cases.json").read_text())


def subtitle(text, start="00:01.0", end="00:03.0", confidence=None):
    row = {"起": start, "止": end, "文字": text, "类型": "台词字幕"}
    if confidence is not None:
        row["置信度"] = confidence
    return row


def sound(text, start="00:01.0", end="00:03.0", id="E01-S001"):
    return {"id": id, "起": start, "止": end, "文本": text, "声线": "E01-V1", "听辨": "清楚"}


@pytest.mark.parametrize("left,right", [
    ("你一天八百个电话", "你一天800个电话"),
    ("看清楚三点四十", "看清楚3:40"),
    ("四点十分我去接孙女", "4:10我去接孙女"),
    ("现在还有二十多分钟", "现在还有20多分钟"),
    ("一千零二十三元", "1023元"),
    ("两万三千元", "23000元"),
    ("一亿零三万五千元", "100035000元"),
    ("十万亿", "10000000000000"),
    ("三点四十分", "０３：４０"),
    ("二十三时五十九分", "23:59"),
    ("三点一四元", "3.14元"),
    ("零点五元", "0.5元"),
    ("二〇二六年", "2026年"),
    ("ABC１２，今天好", "ABC12今天好。"),
])
def test_numeric_equivalence_and_input_preservation(left, right):
    subtitles, sounds = [subtitle(left)], [sound(right)]
    originals = copy.deepcopy((subtitles, sounds))
    result = compare(subtitles, sounds)
    assert not result["用字不一致"]
    assert result["统计"]["配对组数"] == 1
    assert (subtitles, sounds) == originals
    assert differences(left, right) == []


@pytest.mark.parametrize("left,right", [
    ("八百", "900"), ("3:40", "340"), ("3.4元", "34元"), ("4:10", "四点二十分"),
])
def test_different_numbers_remain_different(left, right):
    result = compare([subtitle(left)], [sound(right)])
    assert result["用字不一致"]
    assert result["用字不一致"][0]["字幕原文"] == [left]
    assert result["用字不一致"][0]["声音原文"] == [right]


def test_normalized_positions_map_back_to_original_after_expansion():
    left, right = "花八百元，拿走。", "花800元，带走。"
    change = differences(left, right)[0]
    assert change["字幕位置"] == change["声音位置"] == [5, 6]
    assert change["字幕原文位置"] == [5, 6]
    assert change["声音原文位置"] == [6, 7]
    for label, text in [("字幕", left), ("声音", right)]:
        a, b = change[label + "原文位置"]
        assert text[a:b] == change[label + "文字"]
    change = differences("八百元", "900元")[0]
    assert change["字幕原文位置"] == [0, 2]
    assert "八百元"[slice(*change["字幕原文位置"])] == "八百"


def test_long_sound_pairs_all_seven_subtitles():
    case = CASES["长句"]
    result = compare(case["字幕"], case["声音"])
    assert not result["只有字幕"] and not result["只有声音"]
    entry = result["用字不一致"][0]
    assert entry["字幕编号"] == list(range(1, 8))
    assert entry["声音编号"] == ["E01-S023"]
    assert entry["字幕原文"] == [r["文字"] for r in case["字幕"]]
    assert entry["声音原文"] == [case["声音"][0]["文本"]]
    assert len(entry["差异位置"]) == 4
    assert {c["字幕文字"] for c in entry["差异位置"]} == {"吗", "朵朵", "也", ""}


def test_span_limit_is_effective_and_does_not_join_unrelated_times():
    case = CASES["长句"]
    limited = compare(case["字幕"], case["声音"], max_span=3)
    assert sum(len(r["字幕编号"]) for r in limited["只有字幕"]) == 4
    subtitles = case["字幕"] + [subtitle("无声字幕", "02:00.0", "02:01.0")]
    result = compare(subtitles, case["声音"], max_span=100)
    assert len(result["只有字幕"]) == 1
    assert result["只有字幕"][0]["字幕原文"] == ["无声字幕"]


def test_reverse_long_span_and_split_number():
    sounds = [sound(c, f"00:0{i}.0", f"00:0{i}.5", f"E01-S{i:03}")
              for i, c in enumerate("今天必须把话说清楚", 1)]
    result = compare([subtitle("今天必须把话说清楚", "00:01.0", "00:09.5")], sounds)
    assert result["统计"]["配对组数"] == 1
    assert not result["只有声音"] and not result["用字不一致"]
    result = compare([subtitle("八", end="00:02.0"), subtitle("百元", start="00:02.0")], [sound("800元")])
    assert result["统计"]["配对组数"] == 1
    assert not result["用字不一致"]


def test_single_frame_overlap_respects_zero_slack():
    result = compare([subtitle("好", end="00:01.0")], [sound("好", "00:01.05")], time_slack=0)
    assert result["统计"]["配对组数"] == 1


def test_noise_is_separate_and_does_not_steal_real_subtitle_or_sound():
    case = CASES["噪声"]
    result = compare(case["字幕"], case["声音"])
    assert not result["只有字幕"] and not result["只有声音"]
    assert result["统计"]["疑似识别噪声"] == 1
    noise = result["疑似识别噪声"][0]
    assert noise["字幕编号"] == [3]
    assert noise["字幕原文"] == ["18克0日・黑税二目"]
    assert noise["声音编号"] == ["E01-S018"]
    assert noise["噪声依据"]["低置信度"]
    entry = result["用字不一致"][0]
    assert entry["字幕编号"] == [1, 2, 4]
    assert entry["声音编号"] == ["E01-S018"]
    assert [d["声音文字"] for d in entry["差异位置"]] == ["儿"]


def test_very_low_local_similarity_and_quality_signal_both_required():
    result = compare([subtitle("老师看着呢", confidence=0.1)], [sound("能出什么事老师看着呢")])
    assert not result["疑似识别噪声"]
    result = compare([subtitle("窗外下雨", confidence=0.9)], [sound("马上走吧")])
    assert not result["疑似识别噪声"]
    assert result["用字不一致"]
    result = compare([subtitle("窗外下雨")], [sound("马上走吧")])
    assert not result["疑似识别噪声"]
    result = compare([subtitle("龘靐齉", confidence=0.9)], [sound("马上走吧")])
    assert result["疑似识别噪声"][0]["噪声依据"]["大量非常用字"]
    result = compare([subtitle("龘今天走", confidence=0.9)], [sound("马上离开")])
    assert not result["疑似识别噪声"]


def test_any_overlapping_sound_that_matches_prevents_noise():
    sounds = [sound("马上走吧"), sound("老师看着呢", id="E01-S002")]
    result = compare([subtitle("老师看着呢", confidence=0.1)], sounds)
    assert not result["疑似识别噪声"]


def test_no_overlap_is_not_evidence_of_noise():
    case = CASES["无重叠"]
    result = compare(case["字幕"], case["声音"])
    assert not result["疑似识别噪声"]
    assert [r["字幕原文"] for r in result["只有字幕"]] == [["当丫孤梨"], ["桂芝人呢"], ["三缺一就差你一个"]]
    result = compare([subtitle("龘靐齉", end="00:01.0", confidence=0.1)],
                     [sound("马上走吧", "00:01.05")])
    assert not result["疑似识别噪声"]


def test_noise_thresholds_are_configurable():
    subs, sounds = [subtitle("窗外下雨", confidence=0.3)], [sound("马上走吧")]
    assert compare(subs, sounds)["疑似识别噪声"]
    assert not compare(subs, sounds, noise_confidence=0.2)["疑似识别噪声"]
    assert not compare(subs, sounds, noise_similarity=0)["疑似识别噪声"]
    subs = [subtitle("龘靐齉")]
    assert compare(subs, sounds)["疑似识别噪声"]
    assert not compare(subs, sounds, noise_rare_min_chars=4)["疑似识别噪声"]


@pytest.mark.parametrize("kwargs", [
    {"noise_similarity": -1}, {"noise_confidence": 2},
    {"noise_rare_ratio": 1.1}, {"noise_rare_min_chars": 0},
])
def test_invalid_noise_parameters(kwargs):
    with pytest.raises(ValueError):
        compare([], [], **kwargs)


def test_cli_accepts_new_parameters_and_preserves_originals(tmp_path):
    case = CASES["噪声"]
    for name, rows in [("字幕记录", case["字幕"]), ("声音记录", case["声音"])]:
        (tmp_path / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False))
    assert main([str(tmp_path), "--max-span", "12", "--noise-similarity", "0.2",
                 "--noise-confidence", "0.4", "--noise-rare-ratio", "0.5", "--noise-rare-min-chars", "2"]) == 0
    result = json.loads((tmp_path / "字幕声音对照.json").read_text())
    assert len(result["疑似识别噪声"]) == 1
    assert result["统计"]["字幕条数"] == 4
    for name, rows in [("字幕记录", case["字幕"]), ("声音记录", case["声音"])]:
        assert json.loads((tmp_path / f"{name}.json").read_text()) == rows
