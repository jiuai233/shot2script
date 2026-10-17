"""时间码的精度、进位与非法输入。"""

import math

import pytest

from s2s.timefmt import fmt, to_sec


@pytest.mark.parametrize(("text", "seconds"), [("00:00.0", 0), ("01:23.4", 83.4), ("60:00.0", 3600), ("123:59.9", 7439.9)])
def test_to_sec(text, seconds):
    assert to_sec(text) == pytest.approx(seconds)
    assert fmt(seconds) == text


@pytest.mark.parametrize(("seconds", "text"), [(59.95, "01:00.0"), (0.05, "00:00.1"), (3599.99, "60:00.0"), (1.24, "00:01.2")])
def test_fmt_rounding(seconds, text):
    assert fmt(seconds) == text


@pytest.mark.parametrize("text", ["1:02.3", "00:60.0", "-01:00.0", "00:01x", "", " 00:00.0", "00:00.0\n"])
def test_invalid_text(text):
    with pytest.raises(ValueError):
        to_sec(text)


@pytest.mark.parametrize("seconds", [-0.1, math.inf, -math.inf, math.nan])
def test_invalid_seconds(seconds):
    with pytest.raises(ValueError):
        fmt(seconds)
