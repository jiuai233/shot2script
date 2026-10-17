import importlib.util
import json
from pathlib import Path
import wave

import httpx
import pytest

from s2s import audio, config
from s2s.asr.base import Sentence
from s2s.asr.common import ASRError, retry_delay
from s2s.asr import common, grok
from s2s.asr.doubao import DoubaoBackend, build_payload
from s2s.asr.grok import GrokBackend, request_fields
from s2s.asr.postprocess import merge_chunks, records, seconds, timestamp


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.setattr(config, "get_environment", lambda: {})
    def no_pi():
        raise ASRError("缺少 TEST_KEY，测试禁用真实 Pi 凭据")
    monkeypatch.setattr(grok, "_pi_bearer_token", no_pi)


def settings(provider, tmp_path):
    return {"provider": provider, "key_env": "TEST_KEY", "proxy_env": "TEST_PROXY",
            "api_base": "https://example.test/v1", "model": "" if provider == "grok" else "doubao-seed-2-0-lite-260428",
            "raw_dir": str(tmp_path / "素材/asr_raw")}


def backend(provider, tmp_path):
    return (GrokBackend if provider == "grok" else DoubaoBackend)(settings(provider, tmp_path))


def install_transport(monkeypatch, handler):
    client = httpx.Client
    def factory(**kw):
        kw.pop("proxy", None)
        return client(transport=httpx.MockTransport(handler), **kw)
    monkeypatch.setattr(common.httpx, "Client", factory)


def fake_audio(monkeypatch, tmp_path, length=15):
    monkeypatch.setattr(audio, "duration", lambda _: length)
    calls = []
    def extract(source, target, start=None, end=None):
        calls.append((start, end))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"RIFF-test-wav")
        return target
    monkeypatch.setattr(audio, "extract_audio", extract)
    return calls


def sentence(a, b, text, speaker="x"):
    return Sentence(timestamp(a), timestamp(b), text, speaker)


@pytest.mark.parametrize("provider", ["grok", "doubao"])
def test_missing_key_fails_before_file_or_network(provider, tmp_path):
    with pytest.raises(ASRError, match="缺少 TEST_KEY"):
        backend(provider, tmp_path).transcribe("nonexistent.wav")
    assert not list(tmp_path.rglob("*.json"))


def test_grok_multipart_request(monkeypatch, tmp_path):
    seen = []
    def handler(request):
        body = request.read().decode()
        assert request.url.path == "/v1/stt"
        assert request.headers["Authorization"] == "Bearer secret"
        for field, value in request_fields({}).items():
            assert f'name="{field}"\r\n\r\n{value}' in body
            assert body.index(f'name="{field}"') < body.index('name="file"')
        assert 'filename="chunk-0.wav"' in body
        assert "audio/wav" in body
        seen.append(body)
        return httpx.Response(200, json={"text": "你好。", "words": [{"text": "你好。", "start": .2, "end": 1.2, "speaker": 0}]})
    install_transport(monkeypatch, handler)
    monkeypatch.setattr(config, "get_environment", lambda: {"TEST_KEY": "secret"})
    fake_audio(monkeypatch, tmp_path)
    result = backend("grok", tmp_path).transcribe("clip.wav", 10, 13)
    assert [(s.start, s.end, s.text) for s in result] == [("00:10.2", "00:11.2", "你好。")]
    assert len(seen) == 1
    raw = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert raw["window"] == [10, 13]
    assert raw["status"] == 200
    assert "secret" not in json.dumps(raw)


def test_doubao_payload_and_window(monkeypatch, tmp_path):
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.read())
        assert payload["model"] == "doubao-seed-2-0-lite-260428"
        content = payload["messages"][0]["content"]
        assert content[0]["input_audio"] == {"format": "wav", "data": "UklGRi10ZXN0LXdhdg=="}
        assert "start" in content[1]["text"]
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '[{"start":0.3,"end":1.5,"text":"嗯。","speaker":"spk0","clarity":"清楚"}]'}}]})
    install_transport(monkeypatch, handler)
    monkeypatch.setattr(config, "get_environment", lambda: {"TEST_KEY": "secret"})
    fake_audio(monkeypatch, tmp_path)
    result = backend("doubao", tmp_path).transcribe("clip.wav", 10, 13)
    assert result[0].start == "00:10.3"
    assert result[0].end == "00:11.5"


@pytest.mark.parametrize("status,tries", [(429, 3), (500, 3), (503, 3), (401, 1), (403, 1), (400, 1)])
def test_retry_and_redact(monkeypatch, tmp_path, status, tries):
    seen, sleeps = [], []
    def handler(request):
        seen.append(request)
        return httpx.Response(status, json={"error": "secret https://user:pass@proxy.invalid"}, headers={"Retry-After": "2"})
    install_transport(monkeypatch, handler)
    monkeypatch.setattr(common.time, "sleep", sleeps.append)
    monkeypatch.setattr(config, "get_environment", lambda: {"TEST_KEY": "secret", "TEST_PROXY": "https://user:pass@proxy.invalid"})
    fake_audio(monkeypatch, tmp_path)
    with pytest.raises(ASRError, match=f"HTTP {status}"):
        backend("grok", tmp_path).transcribe("clip.wav")
    assert len(seen) == tries
    assert sleeps == ([2, 2] if tries == 3 else [])
    raws = list(tmp_path.rglob("*.json"))
    assert len(raws) == tries
    for raw in raws:
        text = raw.read_text()
        assert "secret" not in text and "user:pass" not in text


def test_network_timeout_retries(monkeypatch, tmp_path):
    sleeps = []
    def handler(request):
        raise httpx.ReadTimeout("secret must never leave error")
    install_transport(monkeypatch, handler)
    monkeypatch.setattr(common.time, "sleep", sleeps.append)
    monkeypatch.setattr(config, "get_environment", lambda: {"TEST_KEY": "secret"})
    fake_audio(monkeypatch, tmp_path)
    with pytest.raises(ASRError) as error:
        backend("grok", tmp_path).transcribe("clip.wav")
    assert "secret" not in str(error.value)
    assert sleeps == [1, 2]
    assert len(list(tmp_path.rglob("*.json"))) == 3


def test_retry_then_success(monkeypatch, tmp_path):
    statuses = iter([429, 503, 200])
    def handler(request):
        return httpx.Response(next(statuses), json={"text": "", "words": []})
    install_transport(monkeypatch, handler)
    monkeypatch.setattr(common.time, "sleep", lambda _: None)
    monkeypatch.setattr(config, "get_environment", lambda: {"TEST_KEY": "secret"})
    fake_audio(monkeypatch, tmp_path)
    assert backend("grok", tmp_path).transcribe("clip.wav") == []


def test_long_rate_limit_stops():
    with pytest.raises(ASRError, match="60 秒"):
        retry_delay("300", 0)
    assert retry_delay("invalid", 1) == 2
    assert retry_delay("Wed, 21 Oct 2015 07:28:00 GMT", 0) == 1


def test_grok_word_to_sentence(tmp_path):
    data = {"words": [
        {"text": "今天", "start": .1, "end": .5, "speaker": 0},
        {"text": "必须说清楚。", "start": .5, "end": 1.2, "speaker": 0},
        {"text": "嗯", "start": 1.3, "end": 1.5, "speaker": 1},
        {"text": "好", "start": 2.5, "end": 3, "speaker": 1},
    ]}
    result = backend("grok", tmp_path).parse_response(data, 60, 10, "chunk0")
    assert [s.text for s in result] == ["今天必须说清楚。", "嗯", "好"]
    assert result[0].start == "01:00.1"
    assert result[1].speaker == result[2].speaker != result[0].speaker


@pytest.mark.parametrize("data", [{"text":"有台词","words":[]}, {"text":"没有 words"},
    {"words":[{"text":"越界", "start":1, "end":90}]}, {"words":[{"text":"倒置", "start":2,"end":1}]}])
def test_grok_rejects_unusable_timestamps(tmp_path, data):
    with pytest.raises(ValueError):
        backend("grok", tmp_path).parse_response(data, 0, 10, "c")


def response(rows, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": "```json\n" + json.dumps(rows, ensure_ascii=False) + "\n```"}}]}


def test_doubao_parse_unknown_and_clarity(tmp_path):
    rows = [{"start":2,"end":3,"text":"听不清","speaker":None,"clarity":"清楚"},
            {"start":.1,"end":.5,"text":"嗯","speaker":"spk1","clarity":"重叠"},
            {"start":4,"end":5,"text":" ","speaker":None,"clarity":"清楚"},
            {"start":6,"end":7,"text":"好","speaker":None,"clarity":"清楚"}]
    result = backend("doubao", tmp_path).parse_response(response(rows), 60, 10, "c")
    assert [s.clarity for s in result] == ["重叠", "听不清", "清楚"]
    assert result[1].speaker != result[2].speaker
    assert result[0].start == "01:00.1"


@pytest.mark.parametrize("rows,finish", [({},"stop"), ([],"length"),
    ([{"start":0,"end":15,"text":"错","speaker":"s","clarity":"清楚"}],"stop"),
    ([{"start":0,"end":1,"text":"错","speaker":"s","clarity":"不确定"}],"stop")])
def test_doubao_rejects_invalid_output(tmp_path, rows, finish):
    with pytest.raises(ValueError):
        backend("doubao", tmp_path).parse_response(response(rows, finish), 0, 10, "c")


def test_merge_duplicate_partial_and_renumber():
    old = [sentence(8, 9.7, "今天必须说清楚", "c0:0")]
    new = [sentence(9.1, 11, "说清楚这件事。", "c1:3"), sentence(12, 13, "嗯。", "c1:3")]
    merged = merge_chunks(old, new, 9, 10)
    assert [s.text for s in merged] == ["今天必须说清楚这件事。", "嗯。"]
    assert merged[0].start == "00:08.0" and merged[0].end == "00:11.0"
    assert merged[0].speaker == merged[1].speaker
    assert new[0].speaker == "c1:3"  # 不修改调用方输入。
    assert records(merged, 1)[1]["声线"] == "E01-V1"


def test_merge_full_repetition_and_no_false_cross_speaker_identity():
    old = [sentence(8.7, 9.7, "好。", "c0:0")]
    new = [sentence(9, 9.7, "好。", "c1:2"), sentence(11, 12, "你好", "c1:1")]
    result = merge_chunks(old, new, 9, 10)
    assert len(result) == 2
    assert result[0].speaker != result[1].speaker
    assert result[0].start == "00:08.7"


def test_no_text_match_does_not_fabricate_speaker_merge():
    result = merge_chunks([sentence(9,10,"甲","c0:0")], [sentence(9,10,"乙","c1:0")],9,10)
    assert len(result) == 2
    assert result[0].speaker != result[1].speaker


def test_same_spelling_outside_overlap_not_deduplicated():
    result = merge_chunks([sentence(7,8,"好。","c0:0")], [sentence(10,11,"好。","c1:0")],9,10)
    assert len(result) == 2


def test_records_sort_ids_labels_overlap_and_empty():
    values = [sentence(4,5," "), sentence(2,4,"乙","b"), sentence(1,3,"甲","a"), sentence(6,7,"甲又说","a")]
    result = records(values,"E01")
    assert [r["id"] for r in result] == ["E01-S001","E01-S002","E01-S003"]
    assert [r["声线"] for r in result] == ["E01-V1","E01-V2","E01-V1"]
    assert [r["听辨"] for r in result] == ["重叠","重叠","清楚"]
    assert set(result[0]) == {"id","起","止","文本","声线","听辨"}


@pytest.mark.parametrize("value", ["nan",float("inf"),-1,"00:60.0", True])
def test_invalid_times(value):
    with pytest.raises(ValueError):
        seconds(value)


def test_time_rollover():
    assert timestamp(59.96) == "01:00.0"
    assert seconds("01:23.4") == 83.4
    assert seconds("01:01:23.4") == 3683.4


def test_chunk_plan():
    assert list(audio.chunks(20,43,10)) == [(20,30),(29,39),(38,43)]
    with pytest.raises(ValueError):
        list(audio.chunks(0,20,1))
    with pytest.raises(ValueError):
        audio.window(10,0,11)


def test_transcribe_multi_chunk_real_offset(monkeypatch,tmp_path):
    calls = fake_audio(monkeypatch,tmp_path,23)
    monkeypatch.setattr(config,"get_environment",lambda:{"TEST_KEY":"secret"})
    responses = iter([
        {"words":[{"text":"今天必须说清楚", "start":8,"end":9.7,"speaker":0}]},
        {"words":[{"text":"说清楚这件事。", "start":.1,"end":2,"speaker":3}]},
        {"words":[{"text":"后来。", "start":.1,"end":2,"speaker":0}]},
    ])
    install_transport(monkeypatch,lambda r:httpx.Response(200,json=next(responses)))
    b=backend("grok",tmp_path); b.settings["chunk_seconds"]=10
    result=b.transcribe("source.wav")
    assert calls == [(0,10),(9,19),(18,23)]
    assert result[0].text == "今天必须说清楚这件事。"
    assert result[0].end == "00:11.0"
    assert result[1].start == "00:18.1"
    assert result[1].speaker != result[0].speaker


def load_cli(name):
    path=Path(__file__).parents[1]/"tools"/f"{name}.py"
    spec=importlib.util.spec_from_file_location("t3_"+name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_forwards_single_route_arguments(monkeypatch, tmp_path, capsys):
    module = load_cli("asr")
    calls = []
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or {"句数": 1})
    assert module.main(["source.mp4", str(tmp_path / "E01"), "--provider", "local", "--words", "raw.json"]) == 0
    assert calls[0][1]["word_paths"] == [Path("raw.json")]
    assert json.loads(capsys.readouterr().out)["句数"] == 1


def test_cli_failed_episode_does_not_write_empty_record(monkeypatch, tmp_path):
    module = load_cli("asr")
    def fail(*args, **kwargs):
        raise ValueError("真实词时间缺失")
    monkeypatch.setattr(module, "run", fail)
    assert module.main(["source.mp4", str(tmp_path / "E01")]) == 1
    assert not (tmp_path / "E01/声音记录.json").exists()


def test_window_cli_json_no_record_write(monkeypatch,tmp_path,capsys):
    module=load_cli("asr_window")
    b=backend("grok",tmp_path)
    calls=[]
    def transcribe(path,start,end):
        calls.append((start,end));return [sentence(10.1,11,"你好。")]
    monkeypatch.setattr(b,"transcribe",transcribe)
    monkeypatch.setattr(module,"get_backend",lambda _:b)
    assert module.main([str(tmp_path/"source.mp4"),"00:10.0","12"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["起"] == "00:10.1"
    assert calls == [(10,12)]
    assert not list(tmp_path.rglob("声音记录.json"))


def test_real_ffmpeg_extract(tmp_path):
    source=tmp_path/"source.wav"
    with wave.open(str(source),"wb") as writer:
        writer.setparams((2,2,44100,0,"NONE","not compressed"))
        writer.writeframes(b"\x00\x00" * 2 * 44100 * 3)
    out=audio.extract_audio(source,tmp_path/"window.wav",.5,1.7)
    assert audio.duration(out) == pytest.approx(1.2,abs=.01)
    with wave.open(str(out),"rb") as reader:
        assert (reader.getframerate(),reader.getnchannels(),reader.getsampwidth()) == (16000,1,2)
    with pytest.raises(FileExistsError,match="不覆盖"):
        audio.extract_audio(source,out,.5,1.7)


def test_synth_request_body():
    path=Path(__file__).parents[1]/"tmp/T3/synth_test.py"
    spec=importlib.util.spec_from_file_location("t3_synth",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert len(module.LINES) == 10
    assert len({v for v,_ in module.LINES}) == 3
    assert module.request_body("eve","你好") == {"text":"你好","voice_id":"eve","language":"zh","with_timestamps":True,"output_format":{"codec":"wav","sample_rate":16000}}


def test_grok_repeated_keyterms_before_file(monkeypatch, tmp_path):
    def handler(request):
        body = request.read().decode()
        assert body.count('name="keyterm"') == 2
        assert 'name="keyterm"\r\n\r\n林舟' in body
        assert 'name="keyterm"\r\n\r\n沈宁' in body
        assert body.rindex('name="keyterm"') < body.index('name="file"')
        return httpx.Response(200, json={"words": []})
    install_transport(monkeypatch, handler)
    monkeypatch.setattr(config, "get_environment", lambda: {"TEST_KEY": "secret"})
    fake_audio(monkeypatch, tmp_path)
    b = backend("grok", tmp_path)
    b.settings["keyterms"] = ["林舟", "沈宁", "林舟"]
    assert b.transcribe("source.wav") == []


@pytest.mark.parametrize("terms", ["林舟", [""], ["名" * 51], ["人"] * 101, [None]])
def test_keyterm_limits(terms):
    with pytest.raises(ValueError, match="keyterms"):
        request_fields({"keyterms": terms})


def test_grok_null_speakers_are_not_one_person(tmp_path):
    values = {"words": [{"text": "你好。", "start": .1, "end": .4, "speaker": None},
                        {"text": "再见。", "start": .5, "end": 1, "speaker": None}]}
    rows = records(backend("grok", tmp_path).parse_response(values, 0, 2, "c"), 1)
    assert rows[0]["声线"] != rows[1]["声线"]


@pytest.mark.parametrize("provider", ["grok", "doubao"])
def test_sub_tenth_second_acknowledgement_kept(tmp_path, provider):
    row = {"text": "嗯。", "start": .01, "end": .04, "speaker": 0, "clarity": "清楚"}
    data = {"words": [row]} if provider == "grok" else response([row])
    rows = records(backend(provider, tmp_path).parse_response(data, 0, 2, "c"), 1)
    assert rows[0]["起"] == "00:00.0" and rows[0]["止"] == "00:00.1"


@pytest.mark.parametrize("provider", ["grok", "doubao"])
@pytest.mark.parametrize("bad", [[], {"choices": [None]}, {"words": [{"text": None, "start": 0, "end": 1}]}])
def test_malformed_response_has_safe_error(monkeypatch, tmp_path, provider, bad):
    install_transport(monkeypatch, lambda r: httpx.Response(200, json=bad))
    monkeypatch.setattr(config, "get_environment", lambda: {"TEST_KEY": "secret"})
    fake_audio(monkeypatch, tmp_path)
    with pytest.raises(ASRError, match="返回格式或时间无效"):
        backend(provider, tmp_path).transcribe("source.wav")


def test_boundary_speaker_mapping_must_be_one_to_one():
    old = [sentence(8, 9.8, "说清楚", "c0:0")]
    new = [sentence(9, 9.5, "说清楚", "c1:0"), sentence(9.4, 9.8, "说清楚", "c1:1")]
    result = merge_chunks(old, new, 9, 10)
    assert len({s.speaker for s in result}) == 3


def test_cross_chunk_residual_sentence_joined():
    old = [sentence(8, 9.5, "今天必须说清楚", "c0:0")]
    new = [sentence(9, 9.5, "说清楚", "c1:0"), sentence(9.6, 11, "这件事。", "c1:0")]
    result = merge_chunks(old, new, 9, 10)
    assert len(result) == 1
    assert result[0].text == "今天必须说清楚这件事。"
    assert result[0].end == "00:11.0"


@pytest.mark.parametrize("name", ["asr", "asr_window"])
def test_cli_refuses_removed_remote_keyterms(tmp_path, name):
    module = load_cli(name)
    args = ["source.mp4", str(tmp_path / "E01")] if name == "asr" else ["source.mp4", "0", "1"]
    with pytest.raises(SystemExit) as exc:
        module.main(args + ["--keyterm", "林舟"])
    assert exc.value.code == 2


def test_audio_subprocess_timeout_is_bounded(monkeypatch, tmp_path):
    import subprocess
    source = tmp_path / "source.wav"
    source.touch()
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 30)
    monkeypatch.setattr(audio.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="30 秒"):
        audio.duration(source)
    monkeypatch.setattr(audio, "duration", lambda _: 10)
    with pytest.raises(ValueError, match="120 秒"):
        audio.extract_audio(source, tmp_path / "window.wav", 0, 1)


def test_synth_all_ten_lines_mixed_and_timestamped(monkeypatch, tmp_path, capsys):
    import base64
    import io
    path = Path(__file__).parents[1] / "tmp/T3/synth_test.py"
    spec = importlib.util.spec_from_file_location("t3_synth_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(tmp_path / "synth_test.py")
    monkeypatch.setattr(module, "get_environment", lambda: {"XAI_API_KEY": "secret"})
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x10\x00" * 16000)
    def handler(request):
        assert request.url.path == "/v1/tts"
        return httpx.Response(200, json={"audio": base64.b64encode(stream.getvalue()).decode(),
            "audio_timestamps": {"graph_chars": ["好"], "graph_times": [[.1, .8]]}})
    install_transport(monkeypatch, handler)
    assert module.main() == 0
    marks = json.loads((tmp_path / "synthetic/ground_truth.json").read_text())
    assert len(marks) == 10 and len({m["speaker"] for m in marks}) == 3
    assert marks[7]["start"] == marks[8]["start"]
    assert [m["overlap"] for m in marks] == [False] * 7 + [True, True, False]
    with wave.open(str(tmp_path / "synthetic/benchmark.wav"), "rb") as reader:
        assert reader.getnframes() / reader.getframerate() == pytest.approx(marks[-1]["clip_end"])
    with pytest.raises(ASRError, match="已有产物"):
        module.main()


def test_real_delayed_audio_keeps_video_time_origin(tmp_path):
    from array import array
    import subprocess
    source = tmp_path / "delayed.mkv"
    subprocess.run([audio.FFMPEG, "-nostdin", "-v", "error", "-n",
        "-f", "lavfi", "-i", "color=c=black:s=16x16:r=10:d=2",
        "-itsoffset", "0.35", "-f", "lavfi", "-i", "sine=f=440:r=16000:d=1",
        "-c:v", "ffv1", "-c:a", "pcm_s16le", str(source)], check=True, capture_output=True, timeout=20)
    full = audio.extract_audio(source, tmp_path / "full.wav")
    window = audio.extract_audio(source, tmp_path / "window.wav", .5, .7)
    with wave.open(str(full), "rb") as reader:
        samples = array("h", reader.readframes(reader.getnframes()))
    assert max(abs(v) for v in samples[:3200]) == 0
    assert max(abs(v) for v in samples[6400:9600]) > 1000
    assert audio.duration(full) == pytest.approx(2, abs=.001)
    with wave.open(str(window), "rb") as reader:
        cut = array("h", reader.readframes(reader.getnframes()))
    assert list(cut) == list(samples[8000:11200])


def test_grok_zero_duration_punctuation_from_real_response(tmp_path):
    data = {"words": [
        {"text": "奶", "start": 1.02, "end": 1.18, "speaker": 0},
        {"text": "奶", "start": 1.26, "end": 1.34, "speaker": 0},
        {"text": "！", "start": 1.34, "end": 1.34, "speaker": 0},
        {"text": "你好", "start": 2, "end": 3, "speaker": 1},
        {"text": "，", "start": 3, "end": 3, "speaker": 1},
        {"text": "再见", "start": 3.1, "end": 4, "speaker": 1},
        {"text": "。", "start": 4, "end": 4, "speaker": 1},
    ]}
    rows = records(backend("grok", tmp_path).parse_response(data, 10, 5, "c"), 1)
    assert [r["文本"] for r in rows] == ["奶奶！", "你好，再见。"]
    assert rows[0]["起"] == "00:11.0" and rows[0]["止"] == "00:11.3"
    assert rows[1]["起"] == "00:12.0" and rows[1]["止"] == "00:14.0"


def test_grok_zero_duration_spoken_word_still_rejected(tmp_path):
    with pytest.raises(ValueError, match="词时间戳越界"):
        backend("grok", tmp_path).parse_response(
            {"words": [{"text": "好", "start": 1, "end": 1, "speaker": 0}]}, 0, 2, "c")


def test_grok_zero_duration_punctuation_does_not_create_speaker(tmp_path):
    data = {"words": [
        {"text": "我就是怕孩子等", "start": 34, "end": 37.448, "speaker": 2},
        {"text": "。", "start": 37.448, "end": 37.448, "speaker": 3},
        {"text": "今天来得够早啊", "start": 37.468, "end": 38.5, "speaker": 3},
        {"text": "。", "start": 38.5, "end": 38.5, "speaker": 3},
    ]}
    rows = records(backend("grok", tmp_path).parse_response(data, 0, 40, "c"), 1)
    assert [r["文本"] for r in rows] == ["我就是怕孩子等。", "今天来得够早啊。"]
    assert rows[0]["声线"] != rows[1]["声线"]
