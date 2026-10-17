"""镜头合并、采样与真实视频处理。

样本人工验收：tmp/T2/镜头边界核对.jpg 包含全部候选切点。
漏切、误切的 5% 门槛须由总工逐镜头核对，模型抽查不能替代人工标注；
未收到人工结果前，不声明通过该项验收。实际统计见 orchestrate/reports/T2.md。
"""

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from s2s.frames import FrameReader, dense_frames, sample_times
from s2s.shots import build_shot_table, detect_shots, episode_number, merge_short_shots
from s2s.timefmt import to_sec


@pytest.mark.parametrize(("shots", "expected"), [
    ([], []),
    ([(0, 0.1)], [(0, 0.1)]),
    ([(0, 0.1), (0.1, 1)], [(0, 1)]),
    ([(0, 1), (1, 1.1)], [(0, 1.1)]),
    ([(0, 1), (1, 1.1), (1.1, 3)], [(0, 1.1), (1.1, 3)]),
    ([(0, 2), (2, 2.1), (2.1, 3)], [(0, 2), (2, 3)]),
    ([(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1)], [(0, 0.3), (0.3, 1)]),
    ([(0, 0.3), (0.3, 0.6)], [(0, 0.3), (0.3, 0.6)]),
])
def test_merge(shots, expected):
    original = list(shots)
    assert merge_short_shots(shots) == expected
    assert shots == original


@pytest.mark.parametrize("shots", [[(0, 1), (1.1, 2)], [(0, 1), (0.9, 2)], [(1, 1)], [(-1, 0)]])
def test_merge_rejects_bad_intervals(shots):
    with pytest.raises(ValueError):
        merge_short_shots(shots)


@pytest.mark.parametrize(("start", "end", "fps", "cap"), [(0, 0.3, 2, 12), (0, 1, 2, 12), (0, 90, 2, 12), (2.9666667, 3.7, 2, 12), (0, 12, 30, 3), (3.3, 10.1, 7, 8)])
def test_sampling_bounds_and_spacing(start, end, fps, cap):
    times = sample_times(start, end, fps, cap)
    assert 1 <= len(times) <= cap
    assert times == sorted(set(times))
    assert all(start <= sec < end for sec in times)
    assert all(b - a >= 0.2 - 1e-8 for a, b in zip(times, times[1:]))


def test_sampling_anchors_and_uniform_fill():
    assert sample_times(0, 4) == [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.9]
    capped = sample_times(0, 90)
    assert {0.1, 45.0, 89.9} <= set(capped)
    assert len(capped) == 12
    assert sample_times(0, 10, max_frames=3) == [0.1, 5.0, 9.9]


@pytest.mark.parametrize(("start", "end", "fps", "cap"), [(1, 1, 2, 12), (-1, 1, 2, 12), (0, 1, 0, 12), (0, 1, float("inf"), 12), (0, 1, 2, 2)])
def test_sampling_rejects_bad_parameters(start, end, fps, cap):
    with pytest.raises(ValueError):
        sample_times(start, end, fps, cap)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "E01.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 96))
    assert writer.isOpened()
    for value in (0, 255, 0):
        for _ in range(30):
            writer.write(np.full((96, 64, 3), value, dtype=np.uint8))
    writer.release()
    return path


def test_real_detector_and_frames(video, tmp_path):
    result = build_shot_table(video, tmp_path / "E01")
    assert result["时长"] == "00:03.0"
    assert [(shot["起"], shot["止"]) for shot in result["镜头"]] == [("00:00.0", "00:01.0"), ("00:01.0", "00:02.0"), ("00:02.0", "00:03.0")]
    for shot in result["镜头"]:
        for frame in shot["帧"]:
            image = cv2.imread(str(tmp_path / "E01" / "素材" / "frames" / frame["文件"]))
            assert max(image.shape[:2]) == 960
            value = float(image.mean())
            assert value > 240 if shot["id"] == "E01-K002" else value < 15
            assert to_sec(shot["起"]) <= to_sec(frame["时间"]) < to_sec(shot["止"])


def test_no_cut_is_one_shot(video):
    assert detect_shots(video, threshold=1000) == [(0, 3)]


def test_adaptive_detector(video):
    assert detect_shots(video, detector="adaptive") == [(0, 1), (1, 2), (2, 3)]


def test_dense_frames(video, tmp_path):
    result = dense_frames(video, 0.2, 1.2, 5, tmp_path / "dense")
    assert [frame["时间"] for frame in result["帧"]] == ["00:00.2", "00:00.4", "00:00.6", "00:00.8", "00:01.0"]
    folder = tmp_path / "dense" / "00-00.2-00-01.2"
    assert json.loads((folder / "帧.json").read_text(encoding="utf-8")) == result
    assert all((folder / frame["文件"]).is_file() for frame in result["帧"])
    assert max(cv2.imread(str(folder / "t00-00.2.jpg")).shape[:2]) == 960


@pytest.mark.parametrize(("start", "end", "fps"), [(0, 3.1, 5), (0, 1, 0), (0, 1, 11), (1, 0, 5)])
def test_dense_rejects_bad_window(video, tmp_path, start, end, fps):
    with pytest.raises(ValueError):
        dense_frames(video, start, end, fps, tmp_path / "dense")
    assert not list((tmp_path / "dense").glob("**/*.jpg"))


def test_dense_accepts_rounded_episode_end(tmp_path):
    path = tmp_path / "rounded.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 96))
    assert writer.isOpened()
    for _ in range(38):
        writer.write(np.zeros((96, 64, 3), dtype=np.uint8))
    writer.release()
    result = dense_frames(path, 1, 1.3, 10, tmp_path / "dense")
    assert [frame["时间"] for frame in result["帧"]] == ["00:01.0", "00:01.1", "00:01.2"]
    with pytest.raises(ValueError):
        dense_frames(path, 1, 1.4, 10, tmp_path / "dense")


def test_reader_rejects_missing_video(tmp_path):
    with pytest.raises(ValueError):
        FrameReader(tmp_path / "missing.mp4")


def test_episode_number():
    assert episode_number("clip.mp4", "E02") == 2
    assert episode_number("E03.mp4", "out") == 3
    assert episode_number("clip.mp4", "out", 7) == 7
    with pytest.raises(ValueError):
        episode_number("clip.mp4", "out")
    with pytest.raises(ValueError):
        episode_number("E01.mp4", "out", 0)


@pytest.mark.parametrize(("kwargs"), [{"detector": "unknown"}, {"threshold": 0}, {"min_duration": 0.01}])
def test_bad_detector_parameters(kwargs):
    with pytest.raises(ValueError):
        detect_shots("unused.mp4", **kwargs)


def test_cli_entrypoints(video, tmp_path):
    root = Path(__file__).resolve().parents[1]
    episode_dir = tmp_path / "cli" / "E01"
    commands = [
        ["tools/shots.py", str(video), str(episode_dir), "--fps", "2", "--max-frames", "4"],
        ["tools/segments.py", str(episode_dir / "镜头表.json")],
        ["tools/dense_frames.py", str(video), "00:00.0", "00:01.0", "5", str(episode_dir / "素材" / "frames_dense")],
    ]
    for command in commands:
        result = subprocess.run([sys.executable, *command], cwd=root, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert isinstance(json.loads(result.stdout), dict)
