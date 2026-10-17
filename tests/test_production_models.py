"""生产白名单覆盖配置、直接调用与重答；不启动真实模型。"""

import json
import subprocess

import pytest

from s2s import agent, config, runners, vision

MODEL = config.PRODUCTION_MODEL
FORBIDDEN = ("openai-codex/gpt-6.1-sol", "opencode-go/glm-5.3-flash",
             "xai/grok", "unknown/deepseek-v4.1-flash", MODEL + ":high")


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setattr(config, "get_environment", lambda: {})


@pytest.mark.parametrize("model", FORBIDDEN)
@pytest.mark.parametrize("entry", [
    lambda m: agent.run("角色卡", "核对", model=m),
    lambda m: vision.ask([], "核对", model=m),
    lambda m: runners.run("核对", model=m, thinking="high"),
])
def test_direct_override_refused_before_launch(monkeypatch, entry, model):
    def forbidden(*args, **kwargs):
        pytest.fail("禁止模型不得启动子进程")
    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(config.ConfigError, match="生产模型只允许"):
        entry(model)


@pytest.mark.parametrize("section", ["vision", "default", "全剧主控", "单集主控", "角色卡", "动作识别", "成稿", "验收"])
def test_custom_config_cannot_widen_allowlist(tmp_path, monkeypatch, section):
    data = json.loads(config.DEFAULT_CONFIG.read_text())
    target = data["vision"] if section == "vision" else data["agents"][section]
    target["model"] = "deepseek/deepseek-v4-pro"
    data["allowed_models"]["deepseek"].append("deepseek-v4-pro")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(config, "get_environment", lambda: {"S2S_CONFIG": str(path)})
    with pytest.raises(config.ConfigError, match="生产模型只允许"):
        config.get_agent("角色卡")


@pytest.mark.parametrize("variable", ["S2S_VISION_MODEL", "S2S_AGENT_MODEL"])
def test_environment_cannot_switch_models(monkeypatch, variable):
    monkeypatch.setattr(config, "get_environment", lambda: {variable: FORBIDDEN[0]})
    with pytest.raises(config.ConfigError):
        config.load_config()


def test_all_roles_resolve_deepseek():
    for role in ("全剧主控", "单集主控", "角色卡", "动作识别", "成稿", "验收"):
        assert config.get_agent(role)["model"] == MODEL


def test_runner_failure_does_not_fallback(monkeypatch):
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "不可用")
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(runners.RunnerError, match="调用失败"):
        agent.run("成稿", "成稿")
    assert len(calls) == 1
    assert calls[0][calls[0].index("--model") + 1] == MODEL
