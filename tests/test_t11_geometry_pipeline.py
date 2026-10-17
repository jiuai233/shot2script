import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from s2s.lipsync import Parameters
from s2s.lipsync.core import face_measurement
from s2s.lipsync.pipeline import render_html, run


def landmarks():
    angles = np.linspace(0, 2 * np.pi, 478)
    xy = np.column_stack([.5 + .15 * np.cos(angles), .5 + .20 * np.sin(angles)])
    for i, point in {33: (.4, .45), 263: (.6, .45), 78: (.43, .6), 308: (.57, .6),
                     13: (.5, .59), 14: (.5, .61), 82: (.46, .59), 87: (.46, .61),
                     312: (.54, .59), 317: (.54, .61)}.items():
        xy[i] = point
    return [SimpleNamespace(x=x, y=y, z=0) for x, y in xy]


def test_geometry_front_small_profile_and_mouth_outside():
    points = landmarks()
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    m = face_measurement(points, np.eye(4), image, Parameters())
    assert m["valid"] and m["mar"] > 0 and len(m["feature"]) == 20
    small = np.full((60, 80, 3), 128, dtype=np.uint8)
    assert "人脸过小" in face_measurement(points, np.eye(4), small, Parameters())["reason"]
    angle = np.radians(65)
    matrix = np.eye(4)
    matrix[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    assert "侧脸或俯仰过大" in face_measurement(points, matrix, image, Parameters())["reason"]
    points[13].y = 1.1
    assert "嘴部出框或不可分辨" in face_measurement(points, np.eye(4), image, Parameters())["reason"]


def test_html_is_self_contained_escapes_text_and_marks_faces(tmp_path):
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((100, 100, 3), dtype=np.uint8))
    row = {"句": "E01-S001", "结论": "画内", "起": "00:00.0", "止": "00:01.0", "声线": "V1",
           "文本": "<script>alert(1)</script>", "代表帧": str(image), "代表时间秒": 0, "复核": None,
           "轨迹": [{"id": "F1", "轨迹组": "G1", "状态": "开口", "说话分数": .9, "评分依据": {}}]}
    tracks = {"F1": {"samples": [{"time": 0, "box": [.1, .1, .4, .5]}]}}
    document = render_html({"句": [row], "统计": {}}, tracks)
    assert "data:image/jpeg;base64," in document
    assert "<script>" not in document and "&lt;script&gt;" in document
    assert "http://" not in document and "https://" not in document
    assert "F1" in document and "<!--S2S_STATS-->" in document


def test_real_cpu_pipeline_with_synthetic_no_face_video(tmp_path, monkeypatch):
    pytest.importorskip("mediapipe")
    from s2s.lipsync import pipeline
    import wave
    source = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12, (120, 160))
    assert writer.isOpened()
    for _ in range(24):
        writer.write(np.zeros((160, 120, 3), dtype=np.uint8))
    writer.release()
    board = tmp_path / "board"
    board.mkdir()
    (board / "镜头表.json").write_text(json.dumps({"集": 1, "时长": "00:02.0", "镜头": [
        {"id": "E01-K001", "起": "00:00.0", "止": "00:01.0"},
        {"id": "E01-K002", "起": "00:01.0", "止": "00:02.0"}]}))
    (board / "声音记录.json").write_text(json.dumps([
        {"id": "E01-S001", "起": "00:00.5", "止": "00:01.5", "文本": "跨镜头对白", "声线": "E01-V1"}]))
    def silent_audio(source, target):
        with wave.open(str(target), "wb") as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
            audio.writeframes(np.zeros(32000, dtype="<i2").tobytes())
        return target
    monkeypatch.setattr(pipeline, "extract_audio", silent_audio)
    output = tmp_path / "results" / "口型记录.json"
    model = Path(__file__).resolve().parents[1] / "models/lipsync/face_landmarker.task"
    record = run(source, board, output, Parameters(), model=model)
    assert record["句"][0]["结论"] == "看不到"
    assert record["句"][0]["镜头"] == ["E01-K001", "E01-K002"]
    assert record["句"][0]["轨迹"] == []
    assert record["统计"]["取帧数"] > 12
    assert output.is_file() and (output.parent / "口型核对.html").is_file()
    assert "耗时秒" in (output.parent / "口型核对.html").read_text()
    assert "<!--S2S_STATS-->" not in (output.parent / "口型核对.html").read_text()
    with pytest.raises(FileExistsError):
        run(source, board, output, model=model)
