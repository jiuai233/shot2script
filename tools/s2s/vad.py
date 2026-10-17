"""FSMN-VAD 人声检测；只加载本地模型，不运行语音识别。"""

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np

from .config import PROJECT_ROOT


def waveform(path):
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getframerate(), source.getsampwidth()) != (1, 16000, 2):
            raise ValueError("人声检测输入须为 16 kHz 单声道 PCM16 WAV")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768


def detect(path, settings=None):
    settings = settings or {}
    executable = settings.get("python_executable") or PROJECT_ROOT / "envs/local-asr/bin/python"
    model = Path(settings.get("model_path") or PROJECT_ROOT / "models/asr/fsmn-vad").resolve()
    if not all((model / name).is_file() for name in ("model_quant.onnx", "config.yaml", "am.mvn")):
        raise ValueError(f"缺少本地 FSMN-VAD 文件：{model}；不会自动下载")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = str(PROJECT_ROOT / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    try:
        result = subprocess.run([str(executable), "-m", "s2s.vad", str(path), str(model)],
                                capture_output=True, text=True, env=env,
                                timeout=float(settings.get("timeout_seconds", 60)))
    except subprocess.TimeoutExpired:
        raise ValueError("本地人声检测超时，已停止；未重试") from None
    if result.returncode:
        raise ValueError("本地 FSMN-VAD 运行失败，请核对独立环境和模型；未调用语音识别")
    rows = json.loads(result.stdout)
    total = len(waveform(path)) / 16000
    previous = 0.0
    for row in rows:
        if not 0 <= previous <= row["起"] < row["止"] <= total + .02:
            raise ValueError("FSMN-VAD 返回无效人声边界")
        row["止"] = min(row["止"], total)
        previous = row["止"]
    return rows


def main():
    with redirect_stdout(sys.stderr):
        from funasr_onnx import Fsmn_vad
        model = Fsmn_vad(sys.argv[2], quantize=True, intra_op_num_threads=2, max_end_sil=200)
        # 保留短停顿供统一切句使用，避免默认 800 ms 尾静音吞掉 500 ms 停顿。
        model.vad_scorer_config = {**model.vad_scorer_config, "do_extend": 0}
        result = model(waveform(sys.argv[1]))
        if not isinstance(result, list) or len(result) != 1:
            raise ValueError("FSMN-VAD 没有返回单音轨的人声段")
        rows = [{"起": a / 1000, "止": b / 1000} for a, b in result[0]]
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
