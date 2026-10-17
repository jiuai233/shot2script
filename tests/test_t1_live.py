"""执行 uv run pytest -m live -v -s；真实调用不自动重试。"""

import json
from pathlib import Path
import subprocess
import time

import pytest

from s2s import agent, config, vision

pytestmark = pytest.mark.live
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp" / "T1"


@pytest.fixture(scope="module")
def frame20():
    video = ROOT / "videos" / "sample" / "E01.mp4"
    if not video.is_file():
        pytest.fail(f"样本不存在：{video}")
    WORK.mkdir(parents=True, exist_ok=True)
    frame = WORK / "frame20.jpg"
    if not frame.exists():
        subprocess.run(["/opt/homebrew/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-ss", "20", "-i", str(video), "-frames:v", "1", str(frame)], check=True, timeout=30, capture_output=True, text=True)
    return frame


def record(name, result, elapsed):
    payload = {"输出": result, "耗时秒": round(elapsed, 2)}
    (WORK / f"live_{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{name}（{elapsed:.2f} 秒）：{result}")


@pytest.mark.parametrize("model", ["opencode-go/deepseek-v4.1-flash", "opencode-go/glm-5.3-flash"])
def test_vision_switch_model(monkeypatch, frame20, model):
    monkeypatch.setenv("S2S_VISION_MODEL", model)
    assert config.get_vision()["model"] == model
    started = time.monotonic()
    result = vision.ask([frame20], "用中文简短描述画面中可见的人物、服装和位置。仅描述实际画面，不推断动作。", timeout=90)
    record(model.split("/")[-1], result, time.monotonic() - started)
    assert len(result.strip()) >= 5


def test_default_agent(monkeypatch):
    for name in ("S2S_CONFIG", "S2S_AGENT_MODEL", "S2S_AGENT_RUNNER"):
        monkeypatch.delenv(name, raising=False)
    assert config.get_agent("角色卡")["model"] == "openai-codex/gpt-6.1-sol"
    started = time.monotonic()
    result = agent.run("角色卡", "只回复 OK", timeout=90)
    WORK.mkdir(parents=True, exist_ok=True)
    record("agent", result, time.monotonic() - started)
    assert result.strip() == "OK"
