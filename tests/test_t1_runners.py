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
    assert runners.run("--只回复 OK", [image], [file], model="deepseek/deepseek-flash", thinking="off") == "OK"
    command, kwargs = captured[0]
    assert command == ["/custom/pi", "-p", "--no-session", "--no-context-files", "--model", "deepseek/deepseek-flash", "--thinking", "off", "--system-prompt", runners.DEFAULT_SYSTEM, "--no-tools", "--", f"@{image}", f"@{file}", "--只回复 OK"]
    assert kwargs["timeout"] == 120
    assert kwargs["capture_output"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert "unit-test-secret" not in command


def test_tools_enabled_and_bin_override(captured):
    runners.run("OK", model="deepseek/deepseek-flash", thinking="high", tools=True, bin="/alternate/pi", timeout=3)
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
    assert runners.run("describe", [image], model="deepseek/deepseek-flash", thinking="high", runner="opencode") == "hello world"
    command, kwargs = captured[0]
    assert command == ["/custom/opencode", "run", "--format", "json", "--model", "deepseek/deepseek-flash", "--variant", "high", "--file", str(image), "--", "describe"]
    assert json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])["permission"] == "deny"


def test_opencode_thinking_off_and_tools_enabled(captured, monkeypatch):
    monkeypatch.setattr(runners.subprocess, "run", lambda cmd, **kw: (captured.append((cmd, kw)) or SimpleNamespace(returncode=0, stderr="", stdout='{"type":"text","part":{"text":"OK"}}')))
    assert runners.run("OK", model="deepseek/deepseek-flash", thinking="off", runner="opencode", tools=True) == "OK"
    command, kwargs = captured[0]
    assert "--variant" not in command
    assert "OPENCODE_CONFIG_CONTENT" not in kwargs["env"]


@pytest.mark.parametrize("stdout,code,reason", [("", 0, "空输出"), ("irrelevant", 2, "调用失败")])
def test_failures_have_context(captured, monkeypatch, stdout, code, reason):
    stderr = "\n".join(f"line-{i}" for i in range(25)) + "\nunit-test-secret"
    monkeypatch.setattr(runners.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr))
    with pytest.raises(runners.RunnerError) as info:
        runners.run("OK", model="deepseek/deepseek-flash", thinking="off")
    message = str(info.value)
    assert all(part in message for part in ["运行器=pi", "模型=deepseek/deepseek-flash", f"退出码={code}", reason, "line-24", "[已隐藏]"])
    assert "unit-test-secret" not in message
    assert "line-5\n" not in message
    assert "line-6\n" in message


@pytest.mark.parametrize("error,reason", [(subprocess.TimeoutExpired("pi", 1, stderr=b"timed out"), "超时"), (FileNotFoundError("pi missing"), "无法启动")])
def test_launch_failures(captured, monkeypatch, error, reason):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(runners.subprocess, "run", fail)
    with pytest.raises(runners.RunnerError, match=reason):
        runners.run("OK", model="deepseek/deepseek-flash", thinking="off", timeout=1)


@pytest.mark.parametrize("stdout,reason", [('{"type":"error","error":{"message":"unit-test-secret"}}', "接口错误"), ("not json", "无法解析"), ('{"type":"step_finish"}', "空输出")])
def test_opencode_failures(captured, monkeypatch, stdout, reason):
    monkeypatch.setattr(runners.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=stdout, stderr=""))
    with pytest.raises(runners.RunnerError, match=reason) as info:
        runners.run("OK", model="deepseek/deepseek-flash", thinking="off", runner="opencode")
    assert "unit-test-secret" not in str(info.value)


def test_missing_attachment_does_not_launch(captured, tmp_path):
    with pytest.raises(runners.RunnerError, match="附件不存在"):
        runners.run("OK", [tmp_path / "missing.jpg"], model="deepseek/deepseek-flash", thinking="off")
    assert captured == []


def test_unknown_runner():
    with pytest.raises(runners.RunnerError, match="未知运行器"):
        runners.run("OK", model="deepseek/deepseek-flash", thinking="off", runner="missing")


@pytest.fixture
def deepseek(monkeypatch):
    requests, replies = [], []
    monkeypatch.setattr(runners, "get_environment", lambda: {"DEEPSEEK_API_KEY": "unit-test-secret"})
    monkeypatch.setattr(runners, "get_runner_config", lambda name: {"base_url": "https://api.example", "api_key_env": "DEEPSEEK_API_KEY"})
    monkeypatch.setattr(runners.time, "sleep", lambda seconds: None)

    class Reply:
        def __init__(self, body):
            self.body = body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps(self.body).encode()

    def fake(request, timeout):
        requests.append((request, timeout))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Reply(reply)

    monkeypatch.setattr(runners.urllib.request, "urlopen", fake)
    return requests, replies


def answer(text="OK", **usage):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "prompt_cache_hit_tokens": 60, "completion_tokens": 20,
                      "completion_tokens_details": {"reasoning_tokens": 15}, "total_tokens": 120, **usage}}


def test_deepseek_labels_each_image_and_returns_events(deepseek, tmp_path):
    requests, replies = deepseek
    replies.append(answer("{}"))
    images = [tmp_path / "K001_01.jpg", tmp_path / "K002_01.jpg"]
    for path in images:
        path.write_bytes(b"jpg")
    raw = runners.run("识别", images, model="deepseek/deepseek-flash", thinking="high", system="系统",
                      labels=["甲", "乙"], events=True, runner="deepseek", timeout=30)
    request, timeout = requests[0]
    body = json.loads(request.data)
    assert request.full_url == "https://api.example/chat/completions" and timeout == 30
    assert request.get_header("Authorization") == "Bearer unit-test-secret"
    assert body["model"] == "deepseek-flash" and body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "high"
    assert body["messages"][0] == {"role": "system", "content": "系统"}
    content = body["messages"][1]["content"]
    assert [part.get("text") or part["image_url"]["url"][:23] for part in content] == \
        ["识别", "甲", "data:image/jpeg;base64,", "乙", "data:image/jpeg;base64,"]
    event = json.loads(raw)
    assert event["type"] == "message_end" and event["message"]["content"][0]["text"] == "{}"
    assert event["message"]["usage"] == {"input": 40, "cacheRead": 60, "output": 20, "reasoning": 15, "totalTokens": 120}


def test_deepseek_plain_text_and_thinking_off(deepseek):
    requests, replies = deepseek
    replies.append(answer("  结果 "))
    assert runners.run("问", model="deepseek/deepseek-flash", thinking="off", runner="deepseek") == "结果"
    body = json.loads(requests[0][0].data)
    assert body["thinking"] == {"type": "disabled"} and "reasoning_effort" not in body


def test_deepseek_retries_transient_errors_only(deepseek):
    requests, replies = deepseek
    replies.extend([runners.urllib.error.HTTPError("u", 429, "busy", {}, None), answer()])
    assert runners.run("问", model="deepseek/deepseek-flash", thinking="high", runner="deepseek") == "OK"
    assert len(requests) == 2
    replies.append(runners.urllib.error.HTTPError("u", 400, "bad", {}, None))
    with pytest.raises(runners.RunnerError, match="退出码=400"):
        runners.run("问", model="deepseek/deepseek-flash", thinking="high", runner="deepseek")
    replies.append(TimeoutError())
    with pytest.raises(runners.RunnerError, match="调用超时"):
        runners.run("问", model="deepseek/deepseek-flash", thinking="high", runner="deepseek")
    assert len(requests) == 4


@pytest.mark.parametrize("kwargs,reason", [
    ({"files": ["资料.md"]}, "不开放工具和文件附件"),
    ({"tools": True}, "不开放工具和文件附件"),
    ({"labels": ["多余"]}, "标注数量"),
])
def test_deepseek_rejects_unsupported_input(deepseek, tmp_path, kwargs, reason):
    if "files" in kwargs:
        (tmp_path / "资料.md").write_text("x")
        kwargs = {"files": [tmp_path / "资料.md"]}
    with pytest.raises(runners.RunnerError, match=reason):
        runners.run("问", model="deepseek/deepseek-flash", thinking="high", runner="deepseek", **kwargs)
    assert not deepseek[0]


def test_deepseek_requires_key(deepseek, monkeypatch):
    monkeypatch.setattr(runners, "get_environment", lambda: {})
    with pytest.raises(runners.RunnerError, match="缺少 DEEPSEEK_API_KEY"):
        runners.run("问", model="deepseek/deepseek-flash", thinking="high", runner="deepseek")


def test_labels_rejected_for_pi(captured):
    with pytest.raises(runners.RunnerError, match="只用于 deepseek"):
        runners.run("问", model="deepseek/deepseek-flash", thinking="high", labels=["甲"])
