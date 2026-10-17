"""Offline-only Pi / direct API benchmark checks. No model is contacted."""
import json
from pathlib import Path
import sys

import httpx
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark_dp as b


def test_payload_same_model_and_images(tmp_path):
    file = tmp_path / "frame.jpg"
    Image.new("RGB", (4, 4)).save(file)
    obj = b.make_payload("只记录动作", [file], reasoning="high")
    assert obj["model"] == "deepseek-v4.1-flash"
    assert obj["thinking"] == {"type": "enabled"}
    assert obj["reasoning_effort"] == "high"
    parts = obj["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "只记录动作"}
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert b.fingerprint("ok", [file]) == b.fingerprint("ok", [file])


def test_reject_mixed_model_and_fake_image(tmp_path):
    with pytest.raises(ValueError):
        b.make_payload("hi", [], model="deepseek-chat")
    with pytest.raises(ValueError):
        b.make_payload("hi", [], reasoning="medium")
    fake = tmp_path / "photo.exe"
    fake.write_bytes(b"hi")
    with pytest.raises(ValueError):
        b.make_payload("hi", [fake])


def test_direct_call_mock_no_secret_output():
    seen = {}
    def handler(request):
        seen["req"] = request
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{\"人物\":\"沈宁\"}"}}],
            "usage": {"prompt_tokens": 28, "completion_tokens": 10},
        })
    result = b.direct_call(
        "人物是谁", [], url="http://127.0.0.1:1234/v1/chat/completions", api_key="secret",
        reasoning="high", max_tokens=100, timeout=3, transport=httpx.MockTransport(handler)
    )
    assert result.text == "{\"人物\":\"沈宁\"}"
    assert seen["req"].headers["Authorization"] == "Bearer secret"
    assert seen["payload"]["model"] == b.MODEL
    assert result.usage["prompt_tokens"] == 28


def test_direct_rejects_plain_http_and_missing_key():
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        b.direct_call("x", [], url=b.DEFAULT_URL, api_key="", reasoning="high", max_tokens=100, timeout=1)
    with pytest.raises(ValueError, match="HTTPS"):
        b.direct_call("x", [], url="http://example.com", api_key="secret", reasoning="high", max_tokens=100, timeout=1)


def test_json_schema_is_not_video_truth():
    sch = {"type": "object", "required": ["动作"]}
    assert b.result_checks("{\"动作\":\"举手\"}", expected_schema=sch) == {
        "json_valid": True, "schema_valid": True
    }
    assert b.result_checks("不是 JSON", expected_schema=sch) == {
        "json_valid": False, "schema_valid": None
    }


def test_offline_preflight_never_calls_api(tmp_path, monkeypatch, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("按原素材写动作")
    def forbidden(*a, **kw):
        raise AssertionError("should not call")
    monkeypatch.setattr(b, "direct_call", forbidden)
    monkeypatch.setattr(b, "pi_call", forbidden)
    assert b.main(["--prompt", str(prompt), "--variants", "direct,pi-attach"]) == 0
    assert "offline-preflight" in capsys.readouterr().out


def test_tuned_requires_separate_prompt(tmp_path):
    p = tmp_path / "p"
    p.write_text("x")
    with pytest.raises(SystemExit):
        b.options(["--prompt", str(p), "--variants", "direct-tuned"])
