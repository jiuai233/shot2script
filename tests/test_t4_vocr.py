import json
import subprocess
import sys

import pytest
from PIL import Image, ImageDraw, ImageFont

from s2s.subtitles import ensure_vocr


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision 仅在 macOS 可用")
def test_swift_batch_json_and_failure(tmp_path):
    image = Image.new("RGB", (640, 240), "white")
    ImageDraw.Draw(image).text((80, 80), "Hello 123", fill="black",
                              font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48))
    first, second = tmp_path / "one.jpg", tmp_path / "two.jpg"
    image.save(first)
    image.save(second)
    absent = tmp_path / "missing.jpg"
    result = subprocess.run([str(ensure_vocr()), str(first), str(second), str(absent)],
                            capture_output=True, text=True, check=True, timeout=30)
    rows = json.loads(result.stdout)
    assert [r["path"] for r in rows] == [str(first), str(second), str(absent)]
    assert rows[-1]["error"] == "无法读取图片" and not rows[-1]["results"]
    for row in rows[:2]:
        assert any("Hello" in r["text"] for r in row["results"])
        for text in row["results"]:
            assert 0 <= text["confidence"] <= 1
            b = text["box"]
            assert 0 <= b["x"] <= b["x"] + b["width"] <= 1
            assert 0 <= b["y"] <= b["y"] + b["height"] <= 1
            assert b["y"] < .5
