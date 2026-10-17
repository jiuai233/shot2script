"""ASR 后端注册表。"""

from ..config import get_asr_backend
from .base import ASRBackend, Sentence
from .doubao import DoubaoBackend
from .grok import GrokBackend
from .local import LocalBackend

BACKENDS: dict[str, type[ASRBackend]] = {
    "grok": GrokBackend,
    "doubao": DoubaoBackend,
    "local": LocalBackend,
}


def get_backend(name: str | None = None) -> ASRBackend:
    settings = get_asr_backend(name)
    provider = settings["provider"]
    try:
        backend = BACKENDS[provider]
    except KeyError as exc:
        raise ValueError(f"ASR 后端未注册：{provider}") from exc
    return backend(settings)


__all__ = ["ASRBackend", "Sentence", "BACKENDS", "get_backend"]
