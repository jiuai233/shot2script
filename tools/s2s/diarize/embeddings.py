"""本地声纹提取与可选 Spleeter 人声分离。"""

from pathlib import Path
import subprocess

import numpy as np

from s2s.audio import FFMPEG

MODEL_FILES = {
    "campp": "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
    "resnet34": "wespeaker_zh_cnceleb_resnet34.onnx",
}
MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "diarize"
SAMPLE_RATE = 16000


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm <= 1e-10:
        return None
    return vector / norm


def decode_audio(source, sample_rate=16000, channels=1):
    result = subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error", "-i", str(source), "-map", "0:a:0",
         "-vn", "-af", "aresample=async=1:first_pts=0", "-ar", str(sample_rate),
         "-ac", str(channels), "-f", "f32le", "-"],
        capture_output=True, timeout=120,
    )
    if result.returncode:
        raise ValueError("ffmpeg 读取音频失败")
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, channels).copy()


class Embedder:
    def __init__(self, model="campp", model_dir=MODEL_DIR, threads=2):
        import sherpa_onnx

        path = Path(model_dir) / MODEL_FILES[model]
        if not path.is_file():
            raise FileNotFoundError(f"缺少声纹模型：{path}")
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(path), num_threads=threads, provider="cpu",
        )
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def __call__(self, samples):
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if len(samples) < 1600 or np.max(np.abs(samples)) < 1e-5:
            return None
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples)
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return None
        return unit(self.extractor.compute(stream))


def window_embedding(samples, embedder, window_seconds=3.0):
    """长句等分成不超过指定长度的窗口，避免末尾留下极短窗口。"""
    count = max(1, int(np.ceil(len(samples) / (SAMPLE_RATE * window_seconds))))
    vectors = [embedder(part) for part in np.array_split(samples, count)]
    vectors = [v for v in vectors if v is not None]
    return unit(np.mean(vectors, axis=0)) if vectors else None


def separate_vocals(source, model_dir=MODEL_DIR, threads=2):
    import sherpa_onnx

    root = Path(model_dir) / "sherpa-onnx-spleeter-2stems-fp16"
    paths = [root / "vocals.fp16.onnx", root / "accompaniment.fp16.onnx"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"缺少去配乐模型：{path}")
    config = sherpa_onnx.OfflineSourceSeparationConfig(
        model=sherpa_onnx.OfflineSourceSeparationModelConfig(
            spleeter=sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(
                vocals=str(paths[0]), accompaniment=str(paths[1]),
            ), num_threads=threads, provider="cpu",
        )
    )
    separator = sherpa_onnx.OfflineSourceSeparation(config)
    stereo = decode_audio(source, 44100, 2)
    # 固定十秒块限制模型临时内存；左右补一秒后裁回，保留原始时间轴。
    blocks = []
    rate, block, context = 44100, 441000, 44100
    for start in range(0, len(stereo), block):
        stop = min(start + block, len(stereo))
        left, right = max(0, start - context), min(len(stereo), stop + context)
        samples = np.ascontiguousarray(stereo[left:right].T)
        output = separator.process(sample_rate=rate, samples=samples)
        vocals = output.stems[0].data.mean(axis=0)
        expected = right - left
        vocals = np.pad(vocals, (0, max(0, expected - len(vocals))))[:expected]
        blocks.append(vocals[start - left:stop - left])
    vocals = np.concatenate(blocks).astype("<f4")
    result = subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error", "-f", "f32le", "-ar", "44100", "-ac", "1",
         "-i", "-", "-ar", "16000", "-f", "f32le", "-"],
        input=vocals.tobytes(), capture_output=True, timeout=120,
    )
    if result.returncode:
        raise ValueError("ffmpeg 重采样人声失败")
    return np.frombuffer(result.stdout, dtype="<f4").copy()
