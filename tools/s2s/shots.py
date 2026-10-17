"""PySceneDetect 镜头检测、极短镜头合并与逐镜头取帧。"""

import argparse
import json
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np

from scenedetect import detect
from scenedetect.detectors import AdaptiveDetector, ContentDetector

from .frames import FrameReader, sample_times
from .timefmt import fmt, to_sec


def merge_short_shots(shots: list[tuple[float, float]], min_duration: float = 0.3) -> list[tuple[float, float]]:
    """极短段并入较短的邻段；邻段等长时并入前段。"""
    if not math.isfinite(min_duration) or min_duration < 0:
        raise ValueError("最短镜头时长须为非负有限数")
    merged = list(shots)
    for i, (start, end) in enumerate(merged):
        if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
            raise ValueError("镜头区间无效")
        if i and not math.isclose(start, merged[i - 1][1], abs_tol=1e-8):
            raise ValueError("镜头必须首尾相接")
    while len(merged) > 1:
        index = next((i for i, (start, end) in enumerate(merged) if end - start < min_duration - 1e-8), None)
        if index is None:
            break
        if index == 0:
            merged[:2] = [(merged[0][0], merged[1][1])]
        elif index == len(merged) - 1 or merged[index - 1][1] - merged[index - 1][0] <= merged[index + 1][1] - merged[index + 1][0]:
            merged[index - 1:index + 1] = [(merged[index - 1][0], merged[index][1])]
        else:
            merged[index:index + 2] = [(merged[index][0], merged[index + 1][1])]
    return merged


def flash_cuts(video: str | Path) -> list[float]:
    """补充短暂近白转场的峰值切点，不把持续白色场景拆开。"""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames, peaks = [], []
    if not capture.isOpened() or fps <= 0:
        capture.release()
        raise ValueError(f"无法读取视频：{video}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (48, 80)), cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    finally:
        capture.release()
    start = None
    for i in range(len(frames) + 1):
        white = i < len(frames) and float((frames[i] > 220).mean()) >= 0.8
        if white and start is None:
            start = i
        if not white and start is not None:
            end, begin = i, start
            start = None
            if (end - begin) / fps > 0.5:
                continue
            peak = max(range(begin, end), key=lambda n: float(frames[n].mean()))
            before, after = peak - round(0.4 * fps), peak + round(0.4 * fps)
            if before < 0 or after >= len(frames):
                continue
            top = float(frames[peak].mean())
            # 两侧均恢复正常曝光，并有不同画面，排除持续白景或原地曝光闪动。
            difference = float(np.abs(frames[before].astype(float) - frames[after].astype(float)).mean())
            if top - max(float(frames[before].mean()), float(frames[after].mean())) >= 45 and difference >= 20:
                peaks.append(peak / fps)
    return peaks


def detect_shots(video: str | Path, *, detector: str = "content", threshold: float | None = None, min_duration: float = 0.3) -> list[tuple[float, float]]:
    if detector not in {"content", "adaptive"}:
        raise ValueError("detector 须为 content 或 adaptive")
    if not math.isfinite(min_duration) or min_duration < 0.1:
        raise ValueError("最短镜头时长至少为 0.1 秒，与时间码精度一致")
    threshold = (27.0 if detector == "content" else 3.0) if threshold is None else threshold
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("检测阈值须大于 0 且有限")
    # 先保留所有候选切点，再统一合并，避免检测器直接吞掉闪白附近的切换。
    algorithm = ContentDetector(threshold=threshold, min_scene_len=1) if detector == "content" else AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=1)
    scenes = detect(str(video), algorithm, start_in_scene=True)
    if not scenes:
        raise ValueError(f"未能解码有效视频帧：{video}")
    intervals = merge_short_shots([(start.get_seconds(), end.get_seconds()) for start, end in scenes], min_duration)
    # 只在既有镜头内部补点，不移动原检测器已保留的边界。
    for cut in flash_cuts(video):
        index = next((i for i, (start, end) in enumerate(intervals)
                      if start + max(0.5, min_duration) <= cut <= end - max(0.5, min_duration)), None)
        if index is not None:
            start, end = intervals[index]
            intervals[index:index + 1] = [(start, cut), (cut, end)]
    return intervals


def episode_number(video: str | Path, episode_dir: str | Path, episode: int | None = None) -> int:
    if episode is not None:
        if episode < 1:
            raise ValueError("集号须大于 0")
        return episode
    for name in (Path(episode_dir).name, Path(video).stem):
        match = re.fullmatch(r"E(\d+)", name, re.IGNORECASE)
        if match and int(match[1]) > 0:
            return int(match[1])
    raise ValueError("无法从 E01 形式的目录或视频名推断集号，请传 --episode")


def build_shot_table(video: str | Path, episode_dir: str | Path, *, episode: int | None = None, detector: str = "content", threshold: float | None = None, min_duration: float = 0.3, fps: float = 2, max_frames: int = 12) -> dict:
    number = episode_number(video, episode_dir, episode)
    # 在较慢的检测之前检查采样参数。
    sample_times(0, 1, fps, max_frames)
    shots = detect_shots(video, detector=detector, threshold=threshold, min_duration=min_duration)
    directory = Path(episode_dir)
    result = {"集": number, "视频": str(video), "时长": fmt(shots[-1][1]), "镜头": []}
    with FrameReader(video) as reader:
        for index, (start, end) in enumerate(shots, 1):
            shot = {"id": f"E{number:02d}-K{index:03d}", "起": fmt(start), "止": fmt(end), "帧": []}
            for frame_index, sec in enumerate(sample_times(start, end, fps, max_frames), 1):
                filename = f"K{index:03d}_{frame_index:02d}.jpg"
                reader.save(sec, directory / "素材" / "frames" / filename)
                shot["帧"].append({"文件": filename, "时间": fmt(sec)})
            result["镜头"].append(shot)
    (directory / "镜头表.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="检测镜头并提取长边 960 像素的帧图片")
    parser.add_argument("video")
    parser.add_argument("episode_dir")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--detector", choices=("content", "adaptive"), default="content")
    parser.add_argument("--threshold", type=float, help="ContentDetector 默认 27，AdaptiveDetector 默认 3")
    parser.add_argument("--min-shot-seconds", type=float, default=0.3)
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        result = build_shot_table(args.video, args.episode_dir, episode=args.episode, detector=args.detector, threshold=args.threshold, min_duration=args.min_shot_seconds, fps=args.fps, max_frames=args.max_frames)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"镜头处理失败：{exc}\n")
    durations = [to_sec(shot["止"]) - to_sec(shot["起"]) for shot in result["镜头"]]
    print(json.dumps({"镜头数": len(durations), "平均镜头秒数": round(to_sec(result["时长"]) / len(durations), 3), "最短镜头秒数": round(min(durations), 3), "最长镜头秒数": round(max(durations), 3), "帧总数": sum(len(shot["帧"]) for shot in result["镜头"]), "耗时秒": round(time.monotonic() - started, 3)}, ensure_ascii=False))
