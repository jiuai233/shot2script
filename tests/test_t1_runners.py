import json
import subprocess
from types import SimpleNamespace

import pytest

from s2s import runners


@pytest.fixture
def captured(monkeypatch):
    calls = []
    monkeypatch.setattr(runners, "get_environment", lambda: {"TEST_API_KEY": "unit-test-secret"})
    monkeypatch.setattr(runners, "get_runner_config", lambda name: {"bin": f"/custom/{name}"})

    def fake(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="  OK\n", stderr="", returncode=0)

    monkeypatch.setattr(runners.subprocess, "run", fake)
    return calls


def test_pi_command(captured, tmp_path):
    image = tmp_path / "画面 空格.jpg"
    image.touch()
    file = tmp_path / "资料.md"
    file.touch()
    assert runners.run("--只回复 OK", [image], [file], model="provider/model", thinking="off") == "OK"
    command, kwargs = captured[0]
    assert command == ["/custom/pi", "-p", "--no-session", "--no-context-files", "--model", "provider/model", "--thinking", "off", "--no-tools", "--", f"@{image}", f"@{file}", "--只回复 OK"]
    assert kwargs["timeout"] == 120
    assert kwargs["capture_output"] is True
    assert "unit-test-secret" not in command


def test_tools_enabled_and_bin_override(captured):
    runners.run("OK", model="p/m", thinking="high", tools=True, bin="/alternate/pi", timeout=3)
    command, kwargs = captured[0]
    assert "--no-tools" not in command
    assert command[0] == "/alternate/pi"
    assert kwargs["timeout"] == 3


def test_opencode_command_and_output(captured, monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    image.touch()

    def fake(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr="", stdout='{"type":"step_start"}\n{"type":"text","part":{"text":"hello "}}\n{"type":"text","part":{"text":"world"}}\n')

    monkeypatch.setattr(runners.subprocess, "run", fake)
    assert runners.run("describe", [image], model="p/m", thinking="high", runner="opencode") == "hello world"
    command, kwargs = captured[0]
    assert command == ["/custom/opencode", "run", "--format", "json", "--model", "p/m", "--variant", "high", "--file", str(image), "--", "describe"]
    assert json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])["permission"] == "deny"


def test_opencode_thinking_off_and_tools_enabled(captured, monkeypatch):
    monkeypatch.setattr(runners.subprocess, "run", lambda cmd, **kw: (captured.append((cmd, kw)) or SimpleNamespace(returncode=0, stderr="", stdout='{"type":"text","part":{"text":"OK"}}')))
    assert runners.run("OK", model="p/m", thinking="off", runner="opencode", tools=True) == "OK"
    command, kwargs = captured[0]
    assert "--variant" not in command
    assert "OPENCODE_CONFIG_CONTENT" not in kwargs["env"]


@pytest.mark.parametrize("stdout,code,reason", [("", 0, "空输出"), ("irrelevant", 2, "调用失败")])
def test_failures_have_context(captured, monkeypatch, stdout, code, reason):
    stderr = "\n".join(f"line-{i}" for i in range(25)) + "\nunit-test-secret"
    monkeypatch.setattr(runners.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr))
    with pytest.raises(runners.RunnerError) as info:
        runners.run("OK", model="p/m", thinking="off")
    message = str(info.value)
    assert all(part in message for part in ["运行器=pi", "模型=p/m", f"退出码={code}", reason, "line-24", "[已隐藏]"])
    assert "unit-test-secret" not in message
    assert "line-5\n" not in message
    assert "line-6\n" in message


@pytest.mark.parametrize("error,reason", [(subprocess.TimeoutExpired("pi", 1, stderr=b"timed out"), "超时"), (FileNotFoundError("pi missing"), "无法启动")])
def test_launch_failures(captured, monkeypatch, error, reason):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(runners.subprocess, "run", fail)
    with pytest.raises(runners.RunnerError, match=reason):
        runners.run("OK", model="p/m", thinking="off", timeout=1)


@pytest.mark.parametrize("stdout,reason", [('{"type":"error","error":{"message":"unit-test-secret"}}', "接口错误"), ("not json", "无法解析"), ('{"type":"step_finish"}', "空输出")])
def test_opencode_failures(captured, monkeypatch, stdout, reason):
    monkeypatch.setattr(runners.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=stdout, stderr=""))
    with pytest.raises(runners.RunnerError, match=reason) as info:
        runners.run("OK", model="p/m", thinking="off", runner="opencode")
    assert "unit-test-secret" not in str(info.value)


def test_missing_attachment_does_not_launch(captured, tmp_path):
    with pytest.raises(runners.RunnerError, match="附件不存在"):
        runners.run("OK", [tmp_path / "missing.jpg"], model="p/m", thinking="off")
    assert captured == []


def test_unknown_runner():
    with pytest.raises(runners.RunnerError, match="未知运行器"):
        runners.run("OK", model="p/m", thinking="off", runner="missing")
