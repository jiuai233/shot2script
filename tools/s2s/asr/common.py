"""后端共用的分段、限流重试和脱敏原始响应保存。"""

from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import httpx

from .. import audio, config
from .base import ASRBackend
from .postprocess import merge_chunks


class ASRError(RuntimeError):
    """可安全输出的语音识别错误，不包含请求凭据。"""


def redact(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[已脱敏]")
        return value
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if isinstance(value, dict):
        return {redact(k, secrets): redact(v, secrets) for k, v in value.items()}
    return value


def retry_delay(value: str | None, attempt: int) -> float:
    delay = 2 ** attempt
    if value:
        try:
            delay = max(delay, float(value))
        except ValueError:
            try:
                delay = max(delay, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError):
                pass
    if delay > 60:
        raise ASRError("服务要求等待超过 60 秒，停止重试，请稍后重新运行")
    return delay


class ChunkedBackend(ASRBackend):
    provider = ""
    max_seconds = 600.0
    max_bytes = 25_000_000

    def credential(self, env):
        key_name = self.settings["key_env"]
        key = env.get(key_name, "")
        if not key:
            raise ASRError(f"缺少 {key_name}，请在项目 .env 中配置")
        return key

    def transcribe(self, audio_path, start=None, end=None):
        env = config.get_environment()
        key = self.credential(env)
        proxy = env.get(self.settings.get("proxy_env", "")) or None
        self._secrets = (key, proxy)
        source = Path(audio_path)
        start, end = audio.window(audio.duration(source), start, end)
        raw_dir = Path(self.settings.get("raw_dir") or source.parent / "asr_raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        limit = min(float(self.settings.get("chunk_seconds", self.max_seconds)), self.max_seconds,
                    (self.max_bytes - 4096) / 32000)
        if limit <= 1:
            raise ValueError("chunk_seconds 必须大于 1 秒")
        sentences = []
        # 临时素材跟随输入放置；命令行入口会传入本集素材目录。
        try:
            client = httpx.Client(proxy=proxy, trust_env=False, timeout=httpx.Timeout(
                float(self.settings.get("timeout", 180)), connect=30))
        except (ValueError, httpx.InvalidURL, ImportError):
            raise ASRError("HTTP 客户端或代理配置无效，请检查配置及代理依赖") from None
        with client, tempfile.TemporaryDirectory(prefix="asr-", dir=raw_dir.parent) as tmp:
            previous_end = start
            for index, (a, b) in enumerate(audio.chunks(start, end, limit)):
                path = audio.extract_audio(source, Path(tmp) / f"chunk-{index}.wav", a, b)
                if path.stat().st_size > self.max_bytes:
                    raise ASRError("分段音频超过后端文件上限")
                response = self._request(client, path, key, raw_dir, a, b)
                try:
                    current = self.parse_response(response, a, b - a, f"chunk{index}")
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    raise ASRError(f"{self.provider} 返回格式或时间无效（{type(exc).__name__}）；请检查 asr_raw") from None
                sentences = merge_chunks(sentences, current, a, previous_end) if index else current
                previous_end = b
        return sentences

    def _request(self, client, path, key, raw_dir, start, end):
        attempts = int(self.settings.get("max_attempts", 3))
        if not 1 <= attempts <= 3:
            raise ValueError("max_attempts 须为 1 至 3")
        for attempt in range(attempts):
            began = time.monotonic()
            try:
                response = self.send(client, path, key)
            except httpx.TransportError as exc:
                self._save_raw(raw_dir, start, end, began, attempt, 0, {"error": type(exc).__name__})
                if attempt + 1 == attempts:
                    raise ASRError(f"{self.provider} 网络连接或超时失败，已尝试 {attempts} 次") from None
                time.sleep(2 ** attempt)
                continue
            try:
                data = response.json()
            except ValueError:
                data = {"body": response.text}
            self._save_raw(raw_dir, start, end, began, attempt, response.status_code, data)
            if response.is_success:
                return redact(data, self._secrets)
            if response.status_code != 429 and response.status_code < 500:
                raise ASRError(f"{self.provider} HTTP {response.status_code}；请检查 asr_raw")
            if attempt + 1 == attempts:
                raise ASRError(f"{self.provider} HTTP {response.status_code}，已尝试 {attempts} 次；请检查 asr_raw")
            time.sleep(retry_delay(response.headers.get("Retry-After"), attempt))
        raise AssertionError("重试循环未返回")

    def _save_raw(self, directory, start, end, began, attempt, status, data):
        now = datetime.now(timezone(timedelta(hours=8)))
        name = f"{self.provider}-{now.strftime('%Y%m%dT%H%M%S%f%z')}-{uuid4().hex[:8]}.json"
        payload = {"provider": self.provider, "time": now.isoformat(), "window": [start, end],
                   "elapsed_seconds": round(time.monotonic() - began, 3), "attempt": attempt + 1,
                   "status": status, "response": data}
        (directory / name).write_text(json.dumps(redact(payload, self._secrets), ensure_ascii=False, indent=2), encoding="utf-8")
