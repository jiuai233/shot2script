"""片段大小、完整覆盖和相邻片段的镜头引用。"""

import copy
import json

import pytest

from s2s.segments import build_segments, partition_sizes, write_segments
from s2s.timefmt import fmt


def table(count):
    return {"集": 1, "镜头": [{"id": f"E01-K{i + 1:03d}", "起": fmt(i), "止": fmt(i + 1)} for i in range(count)]}


@pytest.mark.parametrize("count", range(151))
def test_partition_sizes(count):
    sizes = partition_sizes(count)
    assert sum(sizes) == count
    assert all(1 <= size <= 20 for size in sizes)
    assert all(size == 17 for size in sizes[:-1])
    if count <= 20:
        assert sizes == ([count] if count else [])



@pytest.mark.parametrize("count", [0, 1, 6, 10, 11, 12, 17, 21, 38, 41, 100])
def test_segment_coverage(count):
    source = table(count)
    original = copy.deepcopy(source)
    result = build_segments(source)
    assert source == original
    segments = result["片段"]
    assert [item for segment in segments for item in segment["镜头"]] == [shot["id"] for shot in source["镜头"]]
    for index, segment in enumerate(segments):
        assert segment["id"] == f"E01-P{index + 1:02d}"
        assert segment["衔接前"] == (segments[index - 1]["镜头"][-1] if index else None)
        assert segment["衔接后"] == (segments[index + 1]["镜头"][0] if index + 1 < len(segments) else None)
        if index:
            assert segments[index - 1]["止"] == segment["起"]


@pytest.mark.parametrize("problem", ["gap", "overlap", "duplicate", "wrong_episode", "empty_interval"])
def test_bad_shots(problem):
    source = table(2)
    if problem == "gap":
        source["镜头"][1]["起"] = "00:01.1"
    elif problem == "overlap":
        source["镜头"][1]["起"] = "00:00.9"
    elif problem == "duplicate":
        source["镜头"][1]["id"] = "E01-K001"
    elif problem == "wrong_episode":
        source["镜头"][1]["id"] = "E02-K002"
    else:
        source["镜头"][1]["止"] = "00:01.0"
    with pytest.raises(ValueError):
        build_segments(source)


def test_write_segments(tmp_path):
    path = tmp_path / "镜头表.json"
    path.write_text(json.dumps(table(21)), encoding="utf-8")
    result = write_segments(path)
    assert json.loads((tmp_path / "片段表.json").read_text(encoding="utf-8")) == result
    assert [len(segment["镜头"]) for segment in result["片段"]] == [17, 4]


@pytest.mark.parametrize("count", [-1, 1.5])
def test_bad_count(count):
    with pytest.raises(ValueError):
        partition_sizes(count)
