import json
from pathlib import Path

import pytest

from s2s.sub_align import compare, differences, main


def subtitle(text, start="00:01.0", end="00:03.0", kind="台词字幕"):
    return {"起": start, "止": end, "文字": text, "类型": kind}


def sound(text, start="00:01.0", end="00:03.0", id="E01-S001"):
    return {"id": id, "起": start, "止": end, "文本": text, "声线": "E01-V1", "听辨": "清楚"}


def test_punctuation_space_and_width_no_false_positive():
    result = compare([subtitle("ＡＢＣ　１２，今天：好！")], [sound("ABC12今天好。")])
    assert not result["用字不一致"] and not result["只有声音"] and not result["只有字幕"]


@pytest.mark.parametrize("left,right", [
    ([subtitle("今天必须", end="00:02.0"), subtitle("说清楚", start="00:02.0")], [sound("今天必须说清楚")]),
    ([subtitle("今天必须说清楚")], [sound("今天必须", end="00:02.0"), sound("说清楚", start="00:02.0", id="E01-S002")]),
    ([subtitle("今天", end="00:02.0"), subtitle("必须说清楚", start="00:02.0")],
     [sound("今天必须", end="00:02.0"), sound("说清楚", start="00:02.0", id="E01-S002")]),
])
def test_multispan_alignment(left, right):
    result = compare(left, right)
    assert not result["用字不一致"] and not result["只有字幕"] and not result["只有声音"]
    assert result["断句不一致"]
    assert result["统计"]["配对组数"] == 1


def test_character_difference_and_original_positions():
    result = compare([subtitle("你，在门回等。")], [sound("你在门口等")])
    entry = result["用字不一致"][0]
    assert entry["字幕编号"] == [1] and entry["声音编号"] == ["E01-S001"]
    change = entry["差异位置"][0]
    assert change["字幕位置"] == [3, 4] and change["字幕原文位置"] == [4, 5]
    assert change["字幕文字"] == "回" and change["声音文字"] == "口"


def test_insertion_difference():
    changes = differences("今天走", "今天就走")
    assert changes[0]["字幕位置"] == [2, 2]
    assert changes[0]["声音文字"] == "就"


def test_only_one_side_and_exclude_non_dialogue():
    result = compare([subtitle("独有字幕", end="00:02.0"), subtitle("医院", kind="地点字幕")],
                     [sound("独有声音", start="00:20.0", end="00:21.0")])
    assert len(result["只有字幕"]) == 1 and len(result["只有声音"]) == 1
    assert result["统计"]["字幕条数"] == 1


def test_identical_text_cannot_cross_time():
    result = compare([subtitle("好")], [sound("好", start="01:00.0", end="01:01.0")])
    assert result["只有字幕"] and result["只有声音"]


def test_totally_different_but_same_time_is_mismatch():
    result = compare([subtitle("甲乙")], [sound("丙丁")])
    assert len(result["用字不一致"]) == 1


def test_zero_duration_subtitle():
    result = compare([subtitle("好", end="00:01.0")], [sound("好")])
    assert not result["只有字幕"] and result["统计"]["配对组数"] == 1


def test_empty_and_ordered_repetition():
    assert compare([], [sound("好")])["只有声音"]
    assert compare([], [])["统计"]["配对组数"] == 0
    result = compare([subtitle("好"), subtitle("好", "00:10.0", "00:11.0")],
                     [sound("好"), sound("好", "00:10.0", "00:11.0", "E01-S002")])
    assert result["统计"]["配对组数"] == 2
    assert not result["用字不一致"]


def test_cli_fixture(tmp_path):
    fixture = Path(__file__).parent / "fixtures/t4"
    for name in ("字幕记录.json", "声音记录.json"):
        (tmp_path / name).write_text((fixture / name).read_text())
    assert main([str(tmp_path)]) == 0
    result = json.loads((tmp_path / "字幕声音对照.json").read_text())
    assert result["统计"] == {"字幕条数": 6, "声音条数": 6, "配对组数": 4,
                              "用字不一致": 1, "只有字幕": 1, "只有声音": 1, "断句不一致": 2}


@pytest.mark.parametrize("kwargs", [{"max_span": 0}, {"time_slack": -1}, {"min_similarity": 2}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        compare([], [], **kwargs)
