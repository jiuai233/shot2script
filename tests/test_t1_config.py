import json
import os

import pytest

from s2s import config


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("S2S_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(config, "get_environment", lambda: dict(os.environ))


@pytest.fixture
def custom_config(tmp_path, monkeypatch):
    data = json.loads(config.DEFAULT_CONFIG.read_text())
    path = tmp_path / "models.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("S2S_CONFIG", str(path))
    return data, path


def test_defaults():
    assert config.get_asr_backend()["provider"] == "local"
    assert config.get_asr_backend("doubao")["proxy_env"] == "ARK_DOUBAO_PROXY"
    assert config.get_vision()["model"] == "opencode-go/deepseek-v4.1-flash"
    assert config.get_agent("角色卡")["thinking"] == "high"
    assert config.get_agent("动作识别") == config.get_vision()
    assert config.get_agent("验收") == config.get_vision()
    assert config.get_runner_config("pi") == {"bin": "pi"}


@pytest.mark.parametrize("variable,value,reader,field", [
    ("S2S_ASR_PROVIDER", "doubao", config.get_asr_backend, "provider"),
    ("S2S_VISION_MODEL", "opencode-go/glm-5.3-flash", config.get_vision, "model"),
    ("S2S_VISION_RUNNER", "opencode", config.get_vision, "runner"),
    ("S2S_AGENT_MODEL", "provider/other", lambda: config.get_agent("角色卡"), "model"),
    ("S2S_AGENT_RUNNER", "opencode", lambda: config.get_agent("角色卡"), "runner"),
])
def test_environment_override(monkeypatch, variable, value, reader, field):
    monkeypatch.setenv(variable, value)
    assert reader()[field] == value


def test_vision_inheritance_tracks_environment(monkeypatch):
    monkeypatch.setenv("S2S_VISION_MODEL", "provider/new")
    assert config.get_agent("动作识别")["model"] == "provider/new"
    monkeypatch.setenv("S2S_AGENT_MODEL", "provider/text")
    assert config.get_agent("验收")["model"] == "provider/new"
    assert config.get_agent("角色卡")["model"] == "provider/text"


def test_role_override_wins_over_inherited_fields(custom_config, monkeypatch):
    data, path = custom_config
    data["agents"]["动作识别"].update(model="provider/specific", thinking="low")
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("S2S_VISION_MODEL", "provider/shared")
    result = config.get_agent("动作识别")
    assert result["model"] == "provider/specific"
    assert result["thinking"] == "low"
    assert result["max_images_per_call"] == 12
    assert "use" not in result


def test_whole_config_switch(custom_config):
    data, path = custom_config
    data["vision"]["model"] = "other/model"
    data["runners"]["pi"]["bin"] = "/custom/pi"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert config.get_vision()["model"] == "other/model"
    assert config.get_runner_config("pi")["bin"] == "/custom/pi"


def test_config_reads_fresh_objects():
    result = config.get_vision()
    result["model"] = "modified"
    assert config.get_vision()["model"] == "opencode-go/deepseek-v4.1-flash"


def test_empty_overrides_use_defaults(monkeypatch):
    monkeypatch.setenv("S2S_CONFIG", "")
    monkeypatch.setenv("S2S_VISION_MODEL", "")
    assert config.get_vision()["model"] == "opencode-go/deepseek-v4.1-flash"


@pytest.mark.parametrize("reader", [lambda: config.get_agent("不存在"), lambda: config.get_asr_backend("不存在"), lambda: config.get_runner_config("不存在")])
def test_unknown_reference(reader):
    with pytest.raises(config.ConfigError, match="未配置"):
        reader()


def test_invalid_parent(custom_config):
    data, path = custom_config
    data["agents"]["角色卡"]["use"] = "unknown"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(config.ConfigError, match="use 无效"):
        config.get_agent("角色卡")


@pytest.mark.parametrize("content", ["{bad", "[]", "{}"])
def test_invalid_config(custom_config, content):
    _, path = custom_config
    path.write_text(content)
    with pytest.raises(config.ConfigError, match="无法读取模型配置"):
        config.get_vision()


def test_missing_config(monkeypatch, tmp_path):
    monkeypatch.setenv("S2S_CONFIG", str(tmp_path / "absent.json"))
    with pytest.raises(config.ConfigError, match="无法读取模型配置"):
        config.get_vision()


def test_disallowed_model_rejected():
    with pytest.raises(config.ConfigError):
        config.check_model("opencode-go/kimi-k3")
    with pytest.raises(config.ConfigError):
        config.check_model("opencode-go/kimi-k3:high")
    with pytest.raises(config.ConfigError):
        config.check_model("opencode-go/glm-5.3")
    config.check_model("opencode-go/deepseek-v4.1-flash")
    config.check_model("opencode-go/glm-5.3-flash:high")
    config.check_model("openai-codex/gpt-6.1-sol")


def test_runner_refuses_disallowed_model():
    from s2s import runners
    with pytest.raises(config.ConfigError):
        runners.run("x", model="opencode-go/kimi-k3", thinking="off")
