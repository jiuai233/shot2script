import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from s2s.asr import local
from s2s.asr.common import ASRError
from s2s.asr.local import LocalBackend, local_settings, sentences_from_words
from s2s.asr.local_runtime import coalesce_zero_spans, restore_punctuation


def word(text, start, end):
    return {"text": text, "start": start, "end": end}


def test_defaults_and_local_paths():
    assert LocalBackend.provider == "local"
    settings = local_settings({})
    assert settings["model"] == "paraformer-zh"
    assert Path(settings["model_path"]).is_absolute()
    assert Path(settings["model_path"]).parent.name == "asr"
    assert settings["pause_seconds"] == .35
    assert settings["max_sentence_seconds"] == 6


@pytest.mark.parametrize("settings", [
    {"model": "unknown"}, {"hotwords": "桂芝"}, {"hotwords": [""]},
    {"hotwords": ["桂芝"]}, {"pause_seconds": 0}, {"max_sentence_seconds": float('nan')},
    {"timeout_seconds": -1},
])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        local_settings(settings)


@pytest.mark.parametrize("model", ["paraformer-contextual", "qwen3-asr-0.6b", "whisper-large-v3-turbo"])
def test_hotword_models(model):
    assert local_settings({"model": model, "hotwords": ["桂芝", "朵朵"]})["hotwords"] == ["桂芝", "朵朵"]


def test_punctuation_and_pause_splitting_empty_speaker_absolute_offset():
    words = [word("桂", .1, .3), word("芝。", .3, .6), word("你好", .8, 1.1), word("朵朵", 1.6, 2)]
    rows = sentences_from_words(words, 10, 3, local_settings({}))
    assert [(r.start, r.end, r.text, r.speaker) for r in rows] == [
        ("00:10.1", "00:10.6", "桂芝。", ""), ("00:10.8", "00:11.1", "你好", ""),
        ("00:11.6", "00:12.0", "朵朵", "")]


def test_six_second_limit():
    words = [word("字", i * .5, (i + 1) * .5) for i in range(30)]
    rows = sentences_from_words(words, 0, 15, local_settings({}))
    assert len(rows) == 3
    assert ''.join(r.text for r in rows) == '字' * 30


def test_timestamp_not_invented():
    with pytest.raises(ValueError, match="越界"):
        sentences_from_words([word("字", 0, 12)], 0, 3, local_settings({}))


def test_silence():
    assert sentences_from_words([], 0, 2, local_settings({})) == []


def test_restore_punctuation_without_rewriting_words():
    words = [word("桂", 0, .2), word("芝", .2, .4), word("朵", 1, 1.2), word("朵", 1.2, 1.4)]
    result = restore_punctuation("桂芝，朵朵！", words)
    assert ''.join(w['text'] for w in result) == '桂芝，朵朵！'
    assert words[1]['text'] == '芝'


@pytest.mark.parametrize('text,words', [('你好', [word('你', 0, 1)]), ('甲乙丙', [word('甲', 0, 1), word('丙', 1, 2)]), ('你好', [word('他', 0, 1)])])
def test_aligner_missing_real_char_rejected(text, words):
    with pytest.raises(ValueError):
        restore_punctuation(text, words)


def test_zero_spans_preserve_text_and_measured_time():
    result = coalesce_zero_spans([word('甲', 0, 0), word('乙', .1, .4), word('丙', .4, .4)])
    assert result == [word('甲乙丙', .1, .4)]


@pytest.mark.parametrize('words', [[word('字', .1, .1)], [word('字', .2, .1)]])
def test_unusable_timestamps_rejected(words):
    with pytest.raises(ValueError):
        coalesce_zero_spans(words)


def fake_audio(monkeypatch):
    monkeypatch.setattr(local.audio, 'duration', lambda _: 20)
    def extract(source, target, start, end):
        assert (start, end) == (10, 13)
        target.write_bytes(b'fake wav')
        return target
    monkeypatch.setattr(local.audio, 'extract_audio', extract)


def test_transcribe_window_and_temp_cleanup(monkeypatch):
    fake_audio(monkeypatch)
    backend = LocalBackend({})
    paths = []
    def words(path):
        assert path.exists()
        paths.append(path)
        return [word('你好。', .2, 1)]
    monkeypatch.setattr(backend, '_words', words)
    rows = backend.transcribe('input.mp4', 10, 13)
    assert rows[0].start == '00:10.2'
    assert rows[0].speaker == ''
    assert not paths[0].exists()


def test_subprocess_has_isolated_paths_and_no_credentials(monkeypatch):
    backend = LocalBackend({'python_executable': '/fake/python'})
    def run(args, **kwargs):
        assert args == ['/fake/python', '-m', 's2s.asr.local_runtime', 'input.wav']
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        assert Path(kwargs['env']['PYTHONPATH'].split(':')[0]).name == 'tools'
        assert json.loads(kwargs['input'])['model'] == 'paraformer-zh'
        return SimpleNamespace(returncode=0, stdout=json.dumps({'words': [word('好', 0, 1)]}))
    monkeypatch.setattr(local.subprocess, 'run', run)
    assert backend._words(Path('input.wav')) == [word('好', 0, 1)]


@pytest.mark.parametrize('out', [SimpleNamespace(returncode=1, stdout='secret'), SimpleNamespace(returncode=0, stdout='not JSON')])
def test_worker_failure_does_not_forward_logs(monkeypatch, out):
    monkeypatch.setattr(local.subprocess, 'run', lambda *a, **k: out)
    with pytest.raises(ASRError) as error:
        LocalBackend({'python_executable': 'python'})._words(Path('input.wav'))
    assert 'secret' not in str(error.value)


def test_worker_timeout(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired('python', 1)
    monkeypatch.setattr(local.subprocess, 'run', run)
    with pytest.raises(ASRError, match='已停止'):
        LocalBackend({'python_executable': 'python', 'timeout_seconds': 1})._words(Path('input.wav'))


@pytest.mark.parametrize('segments,expected', [([[1000, 2000]], [word('字', 1.1, 1.5)]), ([], [])])
def test_vad_uses_absolute_word_offsets_without_inventing_speakers(monkeypatch, segments, expected):
    import sys
    from s2s.asr.local_runtime import LocalRecognizer
    class Waveform(list):
        ndim = 1
    waveform = Waveform([0.] * 32000)
    monkeypatch.setitem(sys.modules, 'soundfile', SimpleNamespace(read=lambda *a, **k: (waveform, 16000)))
    monkeypatch.setitem(sys.modules, 'numpy', SimpleNamespace(ascontiguousarray=lambda a: a))
    runtime = object.__new__(LocalRecognizer)
    runtime.vad = lambda wave: [segments]
    runtime._recognize_segment = lambda path, hotwords: {
        'text': '字', 'words': [word('字', .1, .5)], 'asr_seconds': .01,
        'alignment_seconds': 0, 'zero_span_tokens': 0,
    }
    assert runtime.recognize('input.wav', [])['words'] == expected


def test_zero_length_sentence_tail_keeps_punctuation_at_boundary():
    words = [word('你', 0, .2), word('好。', .2, .2), word('下一句', 1, 2)]
    merged = coalesce_zero_spans(words)
    assert merged == [word('你好。', 0, .2), word('下一句', 1, 2)]
    assert [s.text for s in sentences_from_words(merged, 0, 3, local_settings({}))] == ['你好。', '下一句']


def test_qwen_aligner_duration_limit_is_checked_before_extraction(monkeypatch):
    monkeypatch.setattr(local.audio, 'duration', lambda _: 400)
    with pytest.raises(ValueError, match='300 秒'):
        LocalBackend({'model': 'qwen3-asr-0.6b'}).transcribe('long.wav')


def test_missing_model_directory_does_not_download(tmp_path):
    from s2s.asr.local_runtime import LocalRecognizer
    with pytest.raises(FileNotFoundError, match='模型目录不存在'):
        LocalRecognizer(local_settings({'model_path': str(tmp_path / 'missing')}))
