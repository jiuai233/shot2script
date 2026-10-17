"""未编号声音记录；起止时间使用本集内 MM:SS.s。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Sentence:
    start: str
    end: str
    text: str
    speaker: str
    clarity: Literal["清楚", "重叠", "听不清"] = "清楚"


class ASRBackend(ABC):
    def __init__(self, settings: dict):
        self.settings = settings

    @abstractmethod
    def transcribe(
        self, audio_path: str | Path, start: float | None = None, end: float | None = None
    ) -> list[Sentence]:
        """输入窗口以秒计；返回的时间仍相对于整集，不是窗口起点。"""
        raise NotImplementedError
