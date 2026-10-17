"""本地 ASR 入口；重依赖可在独立 Python 环境中运行。"""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from uuid import uuid4

from .. import audio
from .base import ASRBackend, Sentence
from .common import ASRError
from .postprocess import words_to_sentences

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "asr"
MODELS = ("paraformer-zh", "paraformer-contextual", "qwen3-asr-0.6b", "whisper-large-v3-turbo")


def local_settings(settings: dict) -> dict:
    result = dict(settings)
    model = result.setdefault("model", "paraformer-zh")
    if model not in MODELS:
        raise ValueError(f"未知本地模型：{model}；可选：{'、'.join(MODELS)}")
    result["model_path"] = str(Path(result.get("model_path") or MODEL_DIR / model).resolve())
    result["aligner_path"] = str(Path(result.get("aligner_path") or MODEL_DIR / "qwen3-aligner-0.6b").resolve())
    result["vad_path"] = str(Path(result.get("vad_path") or MODEL_DIR / "fsmn-vad").resolve())
    hotwords = result.setdefault("hotwords", [])
    if not isinstance(hotwords, list) or any(not isinstance(w, str) or not w.strip() for w in hotwords):
        raise ValueError("hotwords 须为非空字符串列表")
    if hotwords and model == "paraformer-zh":
        raise ValueError("paraformer-zh 不支持热词；请选 paraformer-contextual")
    for key, default in (("pause_seconds", .35), ("max_sentence_seconds", 6), ("timeout_seconds", 300)):
        number = float(result.setdefault(key, default))
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{key} 须为有限正数")
        result[key] = number
    return result


def sentences_from_words(words: list[dict], offset: float, duration: float, settings: dict) -> list[Sentence]:
    sentences = words_to_sentences(words, offset, duration, "local",
                                  pause_seconds=settings["pause_seconds"],
                                  max_sentence_seconds=settings["max_sentence_seconds"])
    return [replace(sentence, speaker="") for sentence in sentences]


class LocalBackend(ASRBackend):
    provider = "local"

    def __init__(self, settings: dict):
        super().__init__(local_settings(settings))
        self._recognizer = None

    def _words(self, path: Path) -> list[dict]:
        executable = self.settings.get("python_executable")
        if not executable:
            from .local_runtime import LocalRecognizer
            if self._recognizer is None:
                self._recognizer = LocalRecognizer(self.settings)
            return self._recognizer.recognize(path, self.settings["hotwords"])["words"]
        env = dict(os.environ)
        tools = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = tools + os.pathsep + env.get("PYTHONPATH", "")
        env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1"
        try:
            completed = subprocess.run(
                [str(executable), "-m", "s2s.asr.local_runtime", str(path)],
                input=json.dumps(self.settings, ensure_ascii=False), capture_output=True,
                text=True, env=env, timeout=self.settings["timeout_seconds"],
            )
        except subprocess.TimeoutExpired:
            raise ASRError(f"本地识别超过 {self.settings['timeout_seconds']:g} 秒，已停止") from None
        if completed.returncode:
            # 子进程的库日志可能包含环境或路径，只返回错误类别。
            raise ASRError("本地识别子进程失败；请检查模型完整性与独立环境依赖")
        try:
            return json.loads(completed.stdout)["words"]
        except (ValueError, KeyError, TypeError):
            raise ASRError("本地识别子进程没有返回词时间戳") from None

    def transcribe(self, audio_path: str | Path, start=None, end=None) -> list[Sentence]:
        begin, stop = audio.window(audio.duration(audio_path), start, end)
        if self.settings["model"] == "qwen3-asr-0.6b" and stop - begin > 300:
            raise ValueError("Qwen 强制对齐单次最多 300 秒，请指定较短窗口")
        with tempfile.TemporaryDirectory(prefix="s2s-local-") as directory:
            wav = audio.extract_audio(audio_path, Path(directory) / "input.wav", begin, stop)
            words = self._words(wav)
        self.last_words = [{**w, "start": w["start"] + begin, "end": w["end"] + begin} for w in words]
        if self.settings.get("raw_dir"):
            raw = Path(self.settings["raw_dir"])
            raw.mkdir(parents=True, exist_ok=True)
            (raw / f"local-{uuid4().hex}.json").write_text(json.dumps(
                {"provider": "local", "model": self.settings["model"], "window": [begin, stop],
                 "response": {"words": words}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return sentences_from_words(words, begin, stop - begin, self.settings)
