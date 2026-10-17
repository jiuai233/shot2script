"""固定音频处理：统一为 16 kHz、单声道、16-bit PCM WAV。"""

import json
import math
from pathlib import Path
import subprocess

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"


def duration(path: str | Path) -> float:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"音视频不存在：{path}")
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=codec_type",
             "-of", "json", str(path)], capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise ValueError("ffprobe 读取音视频超过 30 秒，已停止") from None
    if result.returncode:
        raise ValueError(f"ffprobe 无法读取音视频：{path}")
    data = json.loads(result.stdout)
    if not any(s.get("codec_type") == "audio" for s in data.get("streams", [])):
        raise ValueError(f"输入没有音轨：{path}")
    seconds = float(data["format"]["duration"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("音频时长无效")
    return seconds


def window(total: float, start=None, end=None) -> tuple[float, float]:
    start = 0.0 if start is None else float(start)
    end = total if end is None else float(end)
    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= total + 0.05:
        raise ValueError(f"时间窗口无效：须满足 0 <= 起 < 止 <= {total:.3f} 秒")
    return start, min(end, total)


def extract_audio(source: str | Path, target: str | Path, start=None, end=None) -> Path:
    source, target = Path(source), Path(target)
    start, end = window(duration(source), start, end)
    if source.resolve() == target.resolve():
        raise ValueError("音频输出不能覆盖输入")
    if target.exists():
        raise FileExistsError(f"音频输出已存在，不覆盖：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # -n 防止覆盖现有素材；调用方使用独立临时目录或新的输出路径。
    try:
        result = subprocess.run(
            [FFMPEG, "-nostdin", "-v", "error", "-n", "-i", str(source), "-ss", str(start),
             "-t", str(end - start), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
             "-af", "aresample=async=1:first_pts=0,apad", "-c:a", "pcm_s16le", str(target)], capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise ValueError("ffmpeg 抽取音频超过 120 秒，已停止") from None
    if result.returncode:
        raise ValueError(f"ffmpeg 抽取音频失败：{target.name}")
    return target


def chunks(start: float, end: float, max_seconds: float, overlap: float = 1.0):
    if not all(math.isfinite(v) for v in (start, end, max_seconds, overlap)):
        raise ValueError("分段参数必须为有限数")
    if not 0 <= start < end or not 0 <= overlap < max_seconds:
        raise ValueError("分段参数无效")
    while start < end:
        stop = min(end, start + max_seconds)
        yield start, stop
        if stop == end:
            break
        start = stop - overlap
