"""ONNX / MLX 推理；权重只从给定本地目录读取。"""

from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import sys
import time


def restore_punctuation(text: str, words: list[dict]) -> list[dict]:
    """对齐器不返回标点，把原识别文本的标点附回对应字词。"""
    result, cursor = [], 0
    for word in words:
        token = word["text"]
        index = text.find(token, cursor)
        if index < 0:
            raise ValueError("强制对齐文本与识别文本不一致")
        gap = text[cursor:index]
        if gap.strip():
            if any(c.isalnum() for c in gap):
                raise ValueError("强制对齐遗漏了实字")
            if result:
                result[-1]["text"] += gap.strip()
        result.append(dict(word))
        cursor = index + len(token)
    tail = text[cursor:].strip()
    if any(c.isalnum() for c in tail):
        raise ValueError("强制对齐遗漏了句尾实字")
    if result:
        result[-1]["text"] += tail
    return result


def coalesce_zero_spans(words: list[dict]) -> list[dict]:
    """零长实字并入相邻真实跨度，不补造字时间、不丢字。"""
    result, pending = [], ""
    for item in words:
        word = dict(item)
        if word["end"] < word["start"]:
            raise ValueError("模型词时间逆序")
        if word["end"] == word["start"]:
            if result and not re.search(r"[。！？!?；;][\"'”’）)]*$", result[-1]["text"]):
                result[-1]["text"] += word["text"]
            else:
                pending += word["text"]
            continue
        word["text"] = pending + word["text"]
        pending = ""
        result.append(word)
    if pending:
        if not result:
            raise ValueError("模型只返回零长字时间")
        result[-1]["text"] += pending
    return result


class LocalRecognizer:
    def __init__(self, settings: dict):
        self.settings = settings
        self.name = settings["model"]
        self.path = Path(settings["model_path"])
        if not self.path.is_dir():
            raise FileNotFoundError(f"模型目录不存在：{self.path}")
        self.aligner = None
        self.vad = None
        started = time.perf_counter()
        if self.name in ("paraformer-zh", "paraformer-contextual"):
            from funasr_onnx import ContextualParaformer, Paraformer, Fsmn_vad
            vad_path = Path(settings["vad_path"])
            if not (vad_path / "model_quant.onnx").is_file():
                raise FileNotFoundError(f"缺少 FSMN-VAD 权重：{vad_path}")
            self.vad = Fsmn_vad(str(vad_path), quantize=True,
                               intra_op_num_threads=int(settings.get("threads", 4)))
            if self.name == "paraformer-zh":
                if not (self.path / "model_quant.onnx").is_file():
                    raise FileNotFoundError("缺少 model_quant.onnx")
                self.model = Paraformer(str(self.path), quantize=True,
                                        intra_op_num_threads=int(settings.get("threads", 4)))
            else:
                # 官方热词仓库将主图命名为 model_quant，配套 embedding 图未量化。
                if not all((self.path / n).is_file() for n in ("model.onnx", "model_eb.onnx")):
                    raise FileNotFoundError("热词版需 model.onnx 与 model_eb.onnx；主图可链接到 model_quant.onnx")
                self.model = ContextualParaformer(str(self.path), quantize=False,
                                                  intra_op_num_threads=int(settings.get("threads", 4)))
        elif self.name == "qwen3-asr-0.6b":
            from mlx_audio.stt.utils import load_model
            self.model = load_model(self.path)
        elif self.name == "whisper-large-v3-turbo":
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder
            if not (self.path / "weights.safetensors").is_file():
                raise FileNotFoundError("缺少 weights.safetensors")
            self.model = ModelHolder.get_model(str(self.path), mx.float16)
        else:
            raise ValueError(f"未知本地模型：{self.name}")
        self.asr_load_seconds = time.perf_counter() - started
        if self.name in ("qwen3-asr-0.6b", "paraformer-contextual"):
            from mlx_audio.stt.utils import load_model
            aligner_path = Path(settings["aligner_path"])
            if not aligner_path.is_dir():
                raise FileNotFoundError(f"对齐模型目录不存在：{aligner_path}")
            self.aligner = load_model(aligner_path)
        self.load_seconds = time.perf_counter() - started

    def recognize(self, path: str | Path, hotwords: list[str]) -> dict:
        if self.vad is None:
            return self._recognize_segment(str(path), hotwords)
        import numpy as np
        import soundfile as sf
        started = time.perf_counter()
        waveform, rate = sf.read(str(path), dtype="float32")
        if rate != 16000 or waveform.ndim != 1:
            raise ValueError("本地推理输入须为 16 kHz 单声道 WAV")
        segments = self.vad(waveform)[0]
        vad_seconds = time.perf_counter() - started
        words, texts = [], []
        asr_seconds, alignment_seconds, zero_spans = vad_seconds, 0, 0
        for begin, end in segments:
            piece = np.ascontiguousarray(waveform[int(begin * 16):int(end * 16)])
            result = self._recognize_segment(piece, hotwords)
            texts.append(result["text"])
            words.extend({**w, "start": w["start"] + begin / 1000,
                          "end": w["end"] + begin / 1000} for w in result["words"])
            asr_seconds += result["asr_seconds"]
            alignment_seconds += result["alignment_seconds"]
            zero_spans += result["zero_span_tokens"]
        return {"text": "".join(texts), "words": words, "vad_segments_ms": segments,
                "vad_seconds": vad_seconds, "asr_seconds": asr_seconds,
                "alignment_seconds": alignment_seconds, "zero_span_tokens": zero_spans,
                "recognition_seconds": time.perf_counter() - started}

    def _recognize_segment(self, path, hotwords: list[str]) -> dict:
        started = time.perf_counter()
        if self.name == "whisper-large-v3-turbo":
            import mlx_whisper
            raw = mlx_whisper.transcribe(path, path_or_hf_repo=str(self.path), language="zh",
                                        word_timestamps=True, temperature=0,
                                        initial_prompt="、".join(hotwords) or None,
                                        condition_on_previous_text=False, verbose=None)
            words = [{"text": w["word"], "start": w["start"], "end": w["end"]}
                     for segment in raw["segments"] for w in segment["words"]]
            text = raw["text"]
        elif self.name == "qwen3-asr-0.6b":
            raw = self.model.generate(path, language="Chinese", hotwords=hotwords,
                                      max_tokens=int(self.settings.get("max_tokens", 2048)), verbose=False)
            text = raw.text
            words = None
        else:
            # 不调用 FunASR 的隐式音频转换，输入已由固定流程统一为 WAV。
            if self.name == "paraformer-contextual":
                raw = self.model(path, hotwords=" ".join(hotwords))
            else:
                raw = self.model(path)
            if not raw:
                text, words = "", []
            else:
                prediction = raw[0]
                if self.name == "paraformer-zh":
                    tokens = prediction["preds"].split()
                    timestamps = prediction.get("timestamp")
                    if timestamps is None or len(tokens) != len(timestamps):
                        raise ValueError("Paraformer 未返回与词对齐的真实时间戳")
                    words = [{"text": token, "start": t[0] / 1000, "end": t[1] / 1000}
                             for token, t in zip(tokens, timestamps)]
                    text = "".join(tokens)
                else:
                    text = prediction["preds"][0]
                    words = None
        asr_seconds = time.perf_counter() - started
        align_started = time.perf_counter()
        if words is None:
            if not text.strip():
                words = []
            else:
                aligned = self.aligner.generate(path, text=text, language="Chinese")
                words = restore_punctuation(text, aligned.segments)
        zero_spans = sum(w["end"] == w["start"] for w in words)
        return {"text": text, "words": coalesce_zero_spans(words), "zero_span_tokens": zero_spans,
                "asr_seconds": asr_seconds, "alignment_seconds": time.perf_counter() - align_started,
                "recognition_seconds": time.perf_counter() - started}


def main():
    settings = json.load(sys.stdin)
    # 部分库会向 stdout 打印加载信息，协议输出保持为单个 JSON。
    with redirect_stdout(sys.stderr):
        recognizer = LocalRecognizer(settings)
        result = recognizer.recognize(sys.argv[1], settings.get("hotwords", []))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
