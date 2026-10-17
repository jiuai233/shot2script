"""仅使用假凭据；不调用真实 Pi，也不读取本机登录文件。"""

import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from s2s import audio, config
from s2s.asr import common, grok
from s2s.asr.common import ASRError
from s2s.asr.doubao import DoubaoBackend
from s2s.asr.grok import GrokBackend, xai_token


@pytest.fixture
def pi_run(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="fake-oauth-bearer\n", stderr="")
    monkeypatch.setattr(grok.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("env", [{"XAI_API_KEY": "fake-api-key"}, {"XAI_API_KEY": "  fake-api-key  "}])
def test_api_key_precedes_pi(env, pi_run):
    assert xai_token(env) == "fake-api-key"
    assert pi_run == []


@pytest.mark.parametrize("env", [{}, {"XAI_API_KEY": ""}, {"XAI_API_KEY": "  "}])
def test_pi_fallback_capture_and_timeout(env, pi_run):
    assert xai_token(env) == "fake-oauth-bearer"
    assert pi_run == [(["pi", "auth", "print-bearer-token", "--provider", "xai"],
                       {"capture_output": True, "text": True, "timeout": 30, "check": False})]


def test_configured_key_name(pi_run):
    assert xai_token({"CUSTOM_KEY": "fake-custom-key"}, "CUSTOM_KEY") == "fake-custom-key"
    assert not pi_run


@pytest.mark.parametrize("status,out", [(1, "fake-secret"), (0, ""), (0, " \n"),
                                         (0, "notice\nfake-secret"), (0, "bad token")])
def test_invalid_pi_output_is_not_exposed(monkeypatch, status, out, capsys):
    monkeypatch.setattr(grok.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=status, stdout=out, stderr="fake-secret"))
    with pytest.raises(ASRError, match="Pi 未返回有效") as exc:
        xai_token({})
    assert "fake-secret" not in str(exc.value)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("error", [FileNotFoundError("fake-secret"), PermissionError("fake-secret"),
    subprocess.TimeoutExpired("pi", 30, output="fake-secret", stderr="fake-secret")])
def test_pi_failure_is_safe_and_not_retried(monkeypatch, error, capsys):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr(grok.subprocess, "run", fail)
    with pytest.raises(ASRError, match="30 秒") as exc:
        xai_token({})
    assert "fake-secret" not in str(exc.value)
    assert calls == [1]
    assert capsys.readouterr() == ("", "")


def test_oauth_used_in_request_and_redacted_in_raw(monkeypatch, tmp_path, pi_run):
    monkeypatch.setattr(config, "get_environment", lambda: {})
    monkeypatch.setattr(audio, "duration", lambda _: 2)
    def extract(source, target, *args):
        target.write_bytes(b"fake-wav")
        return target
    monkeypatch.setattr(audio, "extract_audio", extract)
    def handler(request):
        assert request.headers["Authorization"] == "Bearer fake-oauth-bearer"
        return httpx.Response(200, json={"words": [], "debug": "fake-oauth-bearer"})
    client = httpx.Client
    monkeypatch.setattr(common.httpx, "Client", lambda **kw:
                        client(transport=httpx.MockTransport(handler), **kw))
    backend = GrokBackend({"key_env": "XAI_API_KEY", "api_base": "https://example.test/v1",
                           "raw_dir": str(tmp_path / "asr_raw")})
    assert backend.transcribe("fake.wav") == []
    assert len(pi_run) == 1
    raw = next(tmp_path.rglob("*.json")).read_text()
    assert "fake-oauth-bearer" not in raw
    assert json.loads(raw)["response"]["debug"] == "[已脱敏]"
    assert "fake-oauth-bearer" not in repr(backend.settings)


def test_doubao_never_uses_pi(pi_run):
    with pytest.raises(ASRError, match="缺少 ARK_DOUBAO_API_KEY"):
        DoubaoBackend({"key_env": "ARK_DOUBAO_API_KEY"}).credential({})
    assert not pi_run


def test_synth_uses_same_oauth_resolver(monkeypatch, tmp_path, pi_run):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / "tmp/T3/synth_test.py"
    spec = importlib.util.spec_from_file_location("t3_synth_oauth", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_environment", lambda: {})
    module.__file__ = str(tmp_path / "synth_test.py")
    (tmp_path / "synthetic").mkdir()
    (tmp_path / "synthetic/existing").touch()
    with pytest.raises(ASRError, match="已有产物"):
        module.main()
    assert len(pi_run) == 1
