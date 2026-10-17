from dataclasses import asdict

import pytest

from s2s import agent, vision
from s2s.asr import BACKENDS, Sentence, get_backend
from s2s import config
from s2s.asr import grok
from s2s.asr.common import ASRError


@pytest.fixture
def mocked_runner(monkeypatch):
    calls = []
    def fake(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return "结果"
    monkeypatch.setattr(vision.runners, "run", fake)
    monkeypatch.setattr(config, "get_environment", lambda: {})
    def missing_pi():
        raise ASRError("缺少 XAI_API_KEY 或 Pi xAI 登录")
    monkeypatch.setattr(grok, "_pi_bearer_token", missing_pi)
    return calls


def test_vision_overrides(mocked_runner):
    assert vision.ask(["frame.jpg"], "描述", model="opencode-go/deepseek-v4.1-flash", runner="opencode", timeout=15) == "结果"
    prompt, settings = mocked_runner[0]
    assert prompt == "描述"
    assert settings == {"images": ["frame.jpg"], "runner": "opencode", "model": "opencode-go/deepseek-v4.1-flash", "thinking": "high", "timeout": 15}


def test_vision_image_limit(mocked_runner):
    vision.ask(["f"] * 12, "描述")
    assert len(mocked_runner) == 1
    with pytest.raises(ValueError, match="超过"):
        vision.ask(["f"] * 13, "描述")
    assert len(mocked_runner) == 1
    vision.ask(["f"] * 13, "描述", max_images_per_call=13)


def test_agent_attachments_and_inheritance(mocked_runner):
    assert agent.run("角色卡", "只回复 OK", ["record.json"], ["frame.jpg"], model="opencode-go/deepseek-v4.1-flash", tools=True) == "结果"
    _, settings = mocked_runner[0]
    assert settings == {"images": ["frame.jpg"], "files": ["record.json"], "model": "opencode-go/deepseek-v4.1-flash", "runner": "pi", "thinking": "high", "tools": True}


def test_agent_vision_role_limit(mocked_runner):
    with pytest.raises(ValueError, match="超过"):
        agent.run("动作识别", "描述", images=["f"] * 13)
    assert not mocked_runner


@pytest.mark.parametrize("name", ["grok", "doubao"])
def test_asr_missing_key(name, mocked_runner):
    backend = get_backend(name)
    assert isinstance(backend, BACKENDS[name])
    assert backend.settings["provider"] == name
    with pytest.raises(ASRError, match="缺少"):
        backend.transcribe("audio.wav", start=1, end=2)


def test_asr_default_and_sentence(mocked_runner):
    assert isinstance(get_backend(), BACKENDS["local"])
    assert asdict(Sentence("00:01.0", "00:02.0", "你好", "V1")) == {"start": "00:01.0", "end": "00:02.0", "text": "你好", "speaker": "V1", "clarity": "清楚"}
