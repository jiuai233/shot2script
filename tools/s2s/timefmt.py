"""本集时间码与秒数互转。"""

from decimal import Decimal, ROUND_HALF_UP
import math
import re


_TIME = re.compile(r"(\d{2,}):([0-5]\d(?:\.\d+)?)")


def to_sec(value: str) -> float:
    match = _TIME.fullmatch(value)
    if match is None:
        raise ValueError(f"时间须为 MM:SS.s：{value!r}")
    return int(match[1]) * 60 + float(match[2])


def fmt(sec: float) -> str:
    if not math.isfinite(sec) or sec < 0:
        raise ValueError("秒数须为非负有限数")
    ticks = int((Decimal(str(sec)) * 10).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    minutes, remainder = divmod(ticks, 600)
    seconds, tenth = divmod(remainder, 10)
    return f"{minutes:02d}:{seconds:02d}.{tenth}"
