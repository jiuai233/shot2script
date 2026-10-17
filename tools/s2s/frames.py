"""固定取帧流程；输出时间精度与镜头表一致，均为 0.1 秒。"""

import argparse
import json
import math
from pathlib import Path

import cv2

from .timefmt import fmt, to_sec


def _ticks(sec: float) -> int:
    return round(to_sec(fmt(sec)) * 10)


def _window(start: float, end: float) -> tuple[int, int]:
    if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
        raise ValueError("取帧区间须满足 0 <= 起 < 止，且时间有限")
    lo = math.ceil(start * 10 - 1e-8)
    hi = math.ceil(end * 10 - 1e-8) - 1
    if hi < lo:
        raise ValueError("取帧区间内没有可用的 0.1 秒时间点")
    return lo, hi


def sample_times(start: float, end: float, fps: float = 2, max_frames: int = 12) -> list[float]:
    """首、中、尾优先；补帧不足 0.2 秒的去重，超限时均匀保留补帧。"""
    lo, hi = _window(start, end)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps 须大于 0 且有限")
    if not isinstance(max_frames, int) or max_frames < 3:
        raise ValueError("max_frames 至少为 3，以保留首、中、尾")
    anchors = []
    for sec in (min(start + 0.1, end), (start + end) / 2, max(start, end - 0.1)):
        tick = min(hi, max(lo, _ticks(sec)))
        if all(abs(tick - old) >= 2 for old in anchors):
            anchors.append(tick)
    selected = list(anchors)
    extras = []
    # 输出只精确到 0.1 秒；更高的补帧率不会增加有效候选点。
    step = 1 / min(fps, 10)
    for i in range(math.ceil((end - start) / step)):
        tick = min(hi, max(lo, _ticks(start + i * step)))
        if all(abs(tick - old) >= 2 for old in selected):
            selected.append(tick)
            extras.append(tick)
    slots = max_frames - len(anchors)
    if len(extras) > slots:
        extras = [extras[round((i + 0.5) * len(extras) / slots - 0.5)] for i in range(slots)] if slots else []
    return [tick / 10 for tick in sorted(anchors + extras)]


class FrameReader:
    """复用同一视频句柄，按时间定位并将长边统一为 960 像素。"""

    def __init__(self, video: str | Path):
        self.path = Path(video)
        self.capture = cv2.VideoCapture(str(self.path))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.capture.isOpened() or self.fps <= 0 or self.frame_count <= 0:
            self.capture.release()
            raise ValueError(f"无法读取视频或视频元数据无效：{self.path}")
        self.duration = self.frame_count / self.fps

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.capture.release()

    def read_index(self, index: int):
        if not 0 <= index < self.frame_count:
            raise ValueError(f"帧号越界：{index}")
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self.capture.read()
        if not ok:
            raise RuntimeError(f"读取视频帧失败：{self.path}，帧号 {index}")
        height, width = image.shape[:2]
        scale = 960 / max(height, width)
        return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    def save(self, sec: float, path: str | Path) -> None:
        if not math.isfinite(sec) or sec < 0 or sec >= self.duration:
            raise ValueError(f"取帧时间越界：{sec}")
        index = min(self.frame_count - 1, math.floor(sec * self.fps + 1e-7))
        image = self.read_index(index)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"保存帧图片失败：{path}")


def dense_frames(video: str | Path, start: float, end: float, fps: float, output_dir: str | Path) -> dict:
    """output_dir 为 frames_dense 根目录，函数在其下建立时间区间目录。"""
    _window(start, end)
    if not math.isfinite(fps) or not 0 < fps <= 10:
        raise ValueError("加密取帧 fps 须在 (0, 10] 内，时间码精度为 0.1 秒")
    folder = Path(output_dir) / f"{fmt(start).replace(':', '-')}-{fmt(end).replace(':', '-')}"
    result = {"视频": str(video), "起": fmt(start), "止": fmt(end), "帧率": fps, "帧": []}
    with FrameReader(video) as reader:
        if end > max(reader.duration, to_sec(fmt(reader.duration))) + 1e-8:
            raise ValueError(f"结束时间超过视频时长 {fmt(reader.duration)}")
        # 镜头表末尾可能向上舍入不足 0.05 秒，仍只提取真实存在的帧。
        actual_end = min(end, reader.duration)
        lo, hi = _window(start, actual_end)
        times = sorted({min(hi, max(lo, _ticks(start + i / fps))) for i in range(math.ceil((actual_end - start) * fps))})
        for tick in times:
            sec = tick / 10
            filename = f"t{fmt(sec).replace(':', '-')}.jpg"
            reader.save(sec, folder / filename)
            result["帧"].append({"文件": filename, "时间": fmt(sec)})
    (folder / "帧.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="对指定区间加密取帧，在输出目录下建立时间区间子目录")
    parser.add_argument("video")
    parser.add_argument("start", type=to_sec, metavar="起")
    parser.add_argument("end", type=to_sec, metavar="止")
    parser.add_argument("fps", type=float)
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    try:
        result = dense_frames(args.video, args.start, args.end, args.fps, args.output_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"加密取帧失败：{exc}\n")
    print(json.dumps(result, ensure_ascii=False))
