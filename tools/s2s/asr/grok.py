"""xAI /v1/stt：词级时间戳和 diarize 标签交给后处理组成句列表。"""

import json
from pathlib import Path
import subprocess

from .common import ASRError, ChunkedBackend
from .postprocess import words_to_sentences

DEFAULT_MODEL = "grok-voice-transcribe-2.0"


def _pi_bearer_token():
    try:
        result = subprocess.run(
            ["pi", "auth", "print-bearer-token", "--provider", "xai"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ASRError("Pi xAI 凭据获取失败或超过 30 秒；请检查 Pi 登录，或配置 XAI_API_KEY") from None
    # Pi 的标准错误可能包含认证上下文，不转发任何子进程输出。
    token = result.stdout.strip()
    if result.returncode or not token or any(char.isspace() for char in token):
        raise ASRError("Pi 未返回有效的 xAI 凭据；请检查 Pi 登录，或配置 XAI_API_KEY")
    return token


def xai_token(env, key_name="XAI_API_KEY"):
    return env.get(key_name, "").strip() or _pi_bearer_token()


def request_fields(settings: dict) -> dict:
    terms = settings.get("keyterms", [])
    if (not isinstance(terms, list) or len(terms) > 100 or
            any(not isinstance(term, str) or not term.strip() or len(term) > 50 for term in terms)):
        raise ValueError("keyterms 须为最多 100 个非空字符串，每个最多 50 字符")
    formatted = settings.get("format", False)
    if not isinstance(formatted, bool):
        raise ValueError("format 须为布尔值")
    fields = {"model": settings.get("model") or DEFAULT_MODEL, "language": "zh",
              "diarize": "true", "format": str(formatted).lower(), "filler_words": "true"}
    if "vad_threshold" in settings:
        threshold = settings["vad_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("vad_threshold 须在 0 到 1 之间")
        fields["vad_threshold"] = str(threshold)
    if terms:
        # httpx 将列表值编码为重复的同名字段，符合官方 keyterm 格式。
        fields["keyterm"] = list(dict.fromkeys(terms))
    return fields


class GrokBackend(ChunkedBackend):
    provider = "grok"
    # 一小时为本地分段策略；文档仅声明 500 MB 文件上限。
    max_seconds = 3600.0
    max_bytes = 500_000_000

    def transcribe(self, audio_path, start=None, end=None):
        self.last_words = []
        return super().transcribe(audio_path, start, end)

    def credential(self, env):
        return xai_token(env, self.settings["key_env"])

    def send(self, client, path, key):
        fields = request_fields(self.settings)
        with path.open("rb") as file:
            # httpx 先编码 data 字段，再编码 files；官方要求 file 必须为最后一项。
            return client.post(self.settings["api_base"].rstrip("/") + "/stt",
                               headers={"Authorization": f"Bearer {key}"},
                               data=fields, files={"file": (path.name, file, "audio/wav")})

    def parse_response(self, data, offset, duration, namespace):
        if isinstance(data, dict) and data.get("text") == "" and "words" not in data:
            return []
        if not isinstance(data, dict) or not isinstance(data.get("words"), list):
            raise ValueError("响应缺少词级时间戳")
        if not data["words"] and str(data.get("text", "")).strip():
            raise ValueError("有台词但无时间戳")
        if not hasattr(self, "last_words"):
            self.last_words = []
        # 分段重叠区只留本次中点已进入新区的词，不复制造成重复台词。
        covered = getattr(self, "_word_end", offset) if self.last_words else offset
        self.last_words.extend({**w, "start": w["start"] + offset, "end": w["end"] + offset,
                                "speaker": f"{namespace}:{w.get('speaker')}"} for w in data["words"]
                               if offset == 0 or (w["start"] + w["end"]) / 2 + offset >= covered)
        self._word_end = offset + duration
        return words_to_sentences(data["words"], offset, duration, namespace,
                                  pause_seconds=float(self.settings.get("pause_seconds", 0.35)),
                                  max_sentence_seconds=float(self.settings.get("max_sentence_seconds", 6)),
                                  min_speaker_seconds=float(self.settings.get("min_speaker_seconds", 0.3)))


def keyterms_from(path: Path) -> list[str]:
    """只接受角色卡中的姓名及别名称呼；字幕不得进入自动提示。"""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("人名来源须为角色卡数组")
    if any("文字" in row or "类型" in row for row in rows):
        raise ValueError("人名来源不接受字幕记录；--keyterm-from 只接受角色卡")
    if any("名称" not in row or "名称类型" not in row for row in rows):
        raise ValueError("人名来源须为角色卡，包含名称、名称类型")
    terms = []
    for row in rows:
        if row["名称类型"] == "姓名":
            terms.append(row["名称"])
            terms.extend(row.get("别名称呼", []))
    terms = list(dict.fromkeys(term.strip() for term in terms if isinstance(term, str) and term.strip()))
    request_fields({"keyterms": terms})
    return terms
