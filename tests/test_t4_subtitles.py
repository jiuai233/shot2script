import json
from pathlib import Path
import subprocess

import pytest

from s2s import subtitles as sub


def row(text="今天必须说清楚", x=.3, y=.76, width=.4, height=.04, confidence=.9):
    return {"text": text, "confidence": confidence,
            "box": {"x": x, "y": y, "width": width, "height": height}}


def frame(seconds, *rows):
    return {"time": seconds, "path": f"{seconds}.jpg", "results": list(rows)}


def test_merge_and_representative():
    frames = [frame(0, row()), frame(.5, row()), frame(1, row("今天必须说清楚!")), frame(1.5)]
    records, audit = sub.merge_records(frames)
    assert len(records) == 1
    assert records[0]["起"] == "00:00.0" and records[0]["止"] == "00:01.0"
    assert records[0]["文字"] == "今天必须说清楚"
    assert audit["来源"][0]["出现帧数"] == 3


def test_similar_text_and_gap():
    records, _ = sub.merge_records([frame(0, row("今天必须把事情说清楚")),
                                    frame(.5, row("今天必须把事倩说清楚")),
                                    frame(1), frame(1.5, row("今天必须把事情说清楚"))])
    assert len(records) == 2
    assert records[0]["止"] == "00:00.5"


def test_distinct_position_and_repeated_same_frame():
    records, _ = sub.merge_records([frame(0, row(), row(x=.1, y=.3)), frame(.5, row())])
    assert len(records) == 2
    by_type = {r["类型"]: r for r in records}
    assert by_type["台词字幕"]["止"] == "00:00.5"
    assert by_type["画面文字"]["止"] == "00:00.0"


def test_two_line_dialogue():
    records, _ = sub.merge_records([frame(0, row("第一行", y=.70), row("第二行", y=.75))])
    assert len(records) == 1
    assert records[0]["文字"] == "第一行\n第二行"
    assert records[0]["类型"] == "台词字幕"
    assert records[0]["框"]["height"] == .09


def test_clock_not_merged_into_dialogue():
    records, _ = sub.merge_records([frame(0, row(), row("15:40", x=.48, width=.27, y=.829, height=.082))])
    assert len(records) == 2
    assert [r["类型"] for r in records] == ["台词字幕", "画面文字"]


@pytest.mark.parametrize("text,box,duration,expected", [
    ("对白", row()["box"], 1, "台词字幕"),
    ("林舟 集团总裁", row(y=.4)["box"], 2, "人物字卡"),
    ("林舟 集团总裁", row(y=.4)["box"], 9, "画面文字"),
    ("医院", row(y=.1)["box"], 1, "地点字幕"),
    ("阳光幼儿园", row(y=.1, height=.15)["box"], 1, "画面文字"),
    ("15:40", row(y=.1)["box"], 1, "画面文字"),
    ("合同", row(y=.4)["box"], 1, "画面文字"),
])
def test_classification(text, box, duration, expected):
    assert sub.classify(text, box, duration, sub.Settings()) == expected


def test_watermark_filter_and_keep():
    frames = [frame(i / 2, row("平台角标", x=.8, width=.15, y=.05), row()) for i in range(20)]
    records, audit = sub.merge_records(frames)
    assert len(records) == 1 and records[0]["类型"] == "台词字幕"
    assert audit["过滤条数"] == 1 and audit["水印"][0]["出现帧数"] == 20
    records, _ = sub.merge_records(frames, sub.Settings(keep_watermarks=True))
    assert len(records) == 2


def test_brief_location_not_filtered():
    frames = [frame(i, row("北京", y=.1)) if i < 3 else frame(i) for i in range(20)]
    records, audit = sub.merge_records(frames)
    assert len(records) == 1 and not audit["水印"]


def test_no_confidence_threshold_by_default():
    assert len(sub.merge_records([frame(0, row(confidence=.3))])[0]) == 1
    assert not sub.merge_records([frame(0, row(confidence=.3))], sub.Settings(min_confidence=.5))[0]


def test_frame_time_from_shot_table(tmp_path):
    folder = tmp_path / "素材/frames"
    folder.mkdir(parents=True)
    (folder / "K001_01.jpg").touch()
    (folder / "K001_02.jpg").touch()
    (tmp_path / "镜头表.json").write_text(json.dumps({"镜头": [{"帧": [
        {"文件": "K001_01.jpg", "时间": "00:01.2"}, {"文件": "K001_02.jpg", "时间": "00:09.1"}]}]}))
    assert [t for _, t in sub.load_frames(tmp_path, None, 2)] == [1.2, 9.1]
    with pytest.raises(ValueError, match="fps"):
        sub.load_frames(tmp_path, None, 0)


def test_frame_time_no_silent_guess(tmp_path):
    folder = tmp_path / "素材/frames"
    folder.mkdir(parents=True)
    (folder / "K001_01.jpg").touch()
    with pytest.raises(ValueError, match="镜头表"):
        sub.load_frames(tmp_path, None, 2)
    assert sub.load_frames(tmp_path, folder, 2)[0][1] == 0


def test_ensure_compile_if_missing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sub.subprocess, "run", lambda *args, **kwargs: calls.append(args[0]))
    binary = tmp_path / "bin/vocr"
    assert sub.ensure_vocr(binary) == binary
    assert calls[0][:2] == ["swiftc", "-O"]


def test_recognize_reports_image_failure(monkeypatch):
    monkeypatch.setattr(sub, "ensure_vocr", lambda: Path("vocr"))
    monkeypatch.setattr(sub.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, json.dumps([
        {"path": "x.jpg", "error": "无法读取图片", "results": []}])))
    with pytest.raises(ValueError, match="无法读取图片"):
        sub.recognize([Path("x.jpg")])


def test_cli_writes_schema(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "F000001.jpg").touch()
    monkeypatch.setattr(sub, "recognize", lambda paths, batch_size: [{"path": str(paths[0]), "results": [row()]}])
    episode = tmp_path / "E01"
    assert sub.main([str(episode), "--frames", str(frames), "--audit", str(tmp_path / "audit.json")]) == 0
    records = json.loads((episode / "字幕记录.json").read_text())
    assert set(records[0]) == {"起", "止", "文字", "类型", "框", "置信度"}


def test_time_rounding():
    assert sub.timestamp(59.99) == "01:00.0"
    assert sub.to_seconds("01:00.5") == 60.5
