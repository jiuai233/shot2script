import importlib.util
import json
from pathlib import Path

import pytest

from s2s.asr.base import Sentence
from s2s.asr.grok import GrokBackend, keyterms_from, request_fields
from s2s.asr.postprocess import records, seconds, words_to_sentences


def cli(name='asr'):
    path = Path(__file__).parents[1] / 'tools' / f'{name}.py'
    spec = importlib.util.spec_from_file_location('t3c_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def word(text, start, end, speaker=0):
    return dict(text=text, start=start, end=end, speaker=speaker)


def test_format_switch_and_language():
    assert request_fields({})['format'] == 'false'
    fields = request_fields({'format': True})
    assert fields['format'] == 'true' and fields['language'] == 'zh'
    with pytest.raises(ValueError, match='format'):
        request_fields({'format': 'false'})


@pytest.mark.parametrize('value', [-1, 1.1, True, '0', float('nan')])
def test_invalid_vad(value):
    with pytest.raises(ValueError, match='vad_threshold'):
        request_fields({'vad_threshold': value})


def test_pause_threshold_configurable_without_input_mutation():
    data = [word('先说，', 0, 1), word('再说。', 1.36, 2)]
    original = json.dumps(data)
    assert len(words_to_sentences(data, 0, 3, 'c')) == 2
    assert len(words_to_sentences(data, 0, 3, 'c', pause_seconds=.5)) == 1
    assert json.dumps(data) == original


def test_long_sentence_clauses_detected_before_pause_split():
    data = [word('第一句，', 0, 2), word('第二句，', 2.1, 4),
            word('第三句，', 4.5, 6), word('第四句。', 6.1, 8)]
    rows = words_to_sentences(data, 0, 9, 'c')
    assert [s.text for s in rows] == ['第一句，', '第二句，', '第三句，', '第四句。']
    assert all(seconds(s.end) - seconds(s.start) <= 6 for s in rows)


def test_duration_limit_at_whole_word_boundary():
    data = [word('短剧', 0, 3), word('转写', 3, 6), word('完成。', 6, 9)]
    rows = words_to_sentences(data, 0, 10, 'c')
    assert [s.text for s in rows] == ['短剧转写', '完成。']
    # 服务给的单个超长词不能伪造内部时间。
    assert words_to_sentences([word('不可分割', 0, 7)], 0, 8, 'c')[0].end == '00:07.0'


def test_speaker_jitter_er_suffix_and_suspect_mark():
    data = [word('小', 3.261, 3.521), word('孩', 3.801, 4.061),
            word('儿', 5.521, 5.541, 1), word('，', 5.541, 5.541, 1),
            word('让', 5.561, 5.661, 1), word('开', 5.661, 5.721, 1),
            word('！', 5.721, 5.721, 1)]
    rows = words_to_sentences(data, 0, 6, 'c')
    assert [s.text for s in rows] == ['小孩儿，让开！']
    assert rows[0].clarity == '听不清'
    assert rows[0].start == '00:03.3' and rows[0].end == '00:05.7'
    assert data[2]['speaker'] == 1


def test_short_standalone_reply_keeps_own_speaker():
    rows = words_to_sentences([word('你好。', 0, 1), word('嗯。', 1.1, 1.3, 1)], 0, 2, 'c')
    assert len(rows) == 2 and rows[0].speaker != rows[1].speaker
    assert rows[1].clarity == '清楚'


def test_short_middle_run_joins_continuous_sentence():
    rows = words_to_sentences([word('这', 0, .5), word('是', .5, .6, 1), word('测试。', .6, 1.4)], 0, 2, 'c')
    assert [s.text for s in rows] == ['这是测试。']
    assert rows[0].clarity == '听不清'


@pytest.mark.parametrize('kwargs', [{'pause_seconds': 0}, {'max_sentence_seconds': float('inf')},
                                    {'min_speaker_seconds': -.1}])
def test_split_parameter_validation(kwargs):
    with pytest.raises(ValueError, match='切句参数'):
        words_to_sentences([], 0, 2, 'c', **kwargs)


def test_empty_service_response_without_words_is_silence():
    backend = GrokBackend({})
    assert backend.parse_response({'text': '', 'language': '', 'duration': 2}, 0, 2, 'c') == []
    with pytest.raises(ValueError, match='时间戳'):
        backend.parse_response({'text': '有台词'}, 0, 2, 'c')


def write_json(tmp_path, value):
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(value, ensure_ascii=False))
    return path


def test_role_keyterms_names_and_aliases_only(tmp_path):
    rows = [dict(名称='桂芝', 名称类型='姓名', 别名称呼=['桂阿姨', '桂芝']),
            dict(名称='穿红衣的女孩', 名称类型='描述名', 别名称呼=['红衣女孩'])]
    assert keyterms_from(write_json(tmp_path, rows)) == ['桂芝', '桂阿姨']


def test_subtitle_keyterms_are_rejected(tmp_path):
    rows = [dict(文字='桂芝人呢', 类型='台词字幕')]
    with pytest.raises(ValueError, match='只接受角色卡'):
        keyterms_from(write_json(tmp_path, rows))


@pytest.mark.parametrize('value', [{}, [None], [{'unknown': 'name'}]])
def test_invalid_keyterm_source(tmp_path, value):
    with pytest.raises(ValueError, match='人名来源'):
        keyterms_from(write_json(tmp_path, value))


@pytest.mark.parametrize('name', ['asr', 'asr_window'])
def test_cli_keyterm_from_merge_and_deduplicate(tmp_path, monkeypatch, name):
    module = cli(name)
    source = write_json(tmp_path, [dict(名称='桂芝', 名称类型='姓名', 别名称呼=['桂阿姨'])])
    class Backend:
        provider = 'grok'
        settings = {'keyterms': ['已有姓名']}
        def transcribe(self, *args):
            return []
    backend = Backend()
    monkeypatch.setattr(module, 'get_backend', lambda _: backend)
    if name == 'asr':
        import numpy as np
        monkeypatch.setattr(module.audio, 'extract_audio', lambda *args: Path('audio.wav'))
        monkeypatch.setattr(module, 'detect', lambda *args: [])
        monkeypatch.setattr(module, 'waveform', lambda *args: np.zeros(32000))
        backend.last_words=[]
        args = ['video.mp4', str(tmp_path / 'E01')]
    else:
        args = ['video.mp4', '0', '1']
    assert module.main(args + ['--keyterm', '桂芝', '--keyterm-from', str(source)]) == 0
    assert backend.settings['keyterms'] == ['已有姓名', '桂芝', '桂阿姨']


def subtitle(text='桂芝人呢', start='00:05.0', end='00:05.8', **extra):
    return dict(文字=text, 起=start, 止=end, 类型='台词字幕', **extra)


class RecoveryBackend:
    provider = 'grok'
    def __init__(self, response):
        self.settings = {}
        self.calls = []
        self.response = response
    def transcribe(self, path, start, end):
        self.calls.append((start, end, self.settings.get('vad_threshold')))
        self.last_words = self.response
        return []


def test_recovery_only_audio_text_and_speaker_isolation(monkeypatch):
    module = cli()
    monkeypatch.setattr(module.audio, 'duration', lambda _: 10)
    backend = RecoveryBackend([word('慧芝，人呢？', 5, 5.8)])
    previous = [word('你好。', 1, 2)]
    rows, checks = module.recover_missing(backend, 'audio.wav', previous,
                                        [{'起':5,'止':5.8}], 1, secondary=[])
    assert backend.calls == [(4, 6.8, 0)]
    assert rows[-1]['text'] == '慧芝，人呢？'
    assert rows[-1]['speaker'] != previous[0]['speaker']
    assert checks[0]['新增'][0]['text'] == '慧芝，人呢？'
    assert backend.settings == {} and len(previous) == 1


def test_recovery_existing_long_sentence_not_reinserted(monkeypatch):
    module = cli()
    monkeypatch.setattr(module.audio, 'duration', lambda _: 10)
    backend = RecoveryBackend([])
    sounds = [word('第一句第二句第三句第四句第五句。',1,8)]
    rows, checks = module.recover_missing(backend, 'audio.wav', sounds,
                                        [{'起':2,'止':6}], 1, secondary=[])
    assert rows == sounds and not backend.calls and not checks


def test_recovery_rejects_context_and_restores_vad(monkeypatch):
    module = cli()
    monkeypatch.setattr(module.audio, 'duration', lambda _: 10)
    backend = RecoveryBackend([word('幼儿园门口。',6,6.5)])
    backend.settings['vad_threshold'] = .2
    previous = [word('幼儿园门口。',6,6.5)]
    rows, checks = module.recover_missing(backend, 'audio.wav', previous,
                                        [{'起':5,'止':5.8}], 1, secondary=[])
    assert rows == previous and not checks[0]['新增']
    assert backend.settings['vad_threshold'] == .2


def test_recovery_failure_does_not_discard_full_transcript(monkeypatch):
    from s2s.asr.common import ASRError
    module = cli()
    monkeypatch.setattr(module.audio, 'duration', lambda _: 10)
    backend = RecoveryBackend([])
    def fail(*args):
        raise ASRError('grok HTTP 503；请检查 asr_raw')
    backend.transcribe = fail
    previous = [word('你好。',1,2)]
    rows, checks = module.recover_missing(backend, 'audio.wav', previous,
                                        [{'起':5,'止':5.8}], 1, secondary=[])
    assert rows == previous and '503' in checks[0]['错误'] and backend.settings == {}


def test_recovery_rejects_subtitles(monkeypatch):
    module = cli()
    monkeypatch.setattr(module.audio, 'duration', lambda _: 10)
    backend = RecoveryBackend([])
    with pytest.raises(ValueError, match='字幕'):
        module.recover_missing(backend, 'audio.wav', [], [subtitle()], 1, secondary=[])
    assert not backend.calls


def test_erzi_is_not_an_er_suffix():
    rows = words_to_sentences([word('我的话', 0, 1), word('儿子呢？', 2, 2.2, 1)], 0, 3, 'c')
    assert [s.text for s in rows] == ['我的话', '儿子呢？']
    assert rows[0].speaker != rows[1].speaker


def test_cli_automatic_recovery_saves_diagnostics(tmp_path, monkeypatch):
    import numpy as np
    module = cli()
    dest = tmp_path / 'E01'
    dest.mkdir()
    (dest / '字幕记录.json').write_text('这不是有效字幕；自动流程不得读取')
    backend = RecoveryBackend([word('慧芝，人呢？',5,5.8)])
    calls = []
    def transcribe(path, start=None, end=None):
        calls.append((start,end))
        backend.last_words = [word('你好。',1,2)] if start is None else backend.response
        return []
    backend.transcribe = transcribe
    secondary = write_json(tmp_path, {'words':[]})
    monkeypatch.setattr(module,'get_backend',lambda _:backend)
    monkeypatch.setattr(module.audio,'duration',lambda _:10)
    monkeypatch.setattr(module.audio,'extract_audio',lambda *args:Path('audio.wav'))
    monkeypatch.setattr(module,'detect',lambda *args:[{'起':1,'止':2},{'起':5,'止':5.8}])
    monkeypatch.setattr(module,'waveform',lambda *args:np.zeros(160000))
    assert module.main(['video.mp4',str(dest),'--secondary-words',str(secondary)]) == 0
    assert calls == [(None,None),(4,6.8)]
    rows = json.loads((dest / '声音记录.json').read_text())
    assert rows[-1]['文本'] == '慧芝，人呢？' and rows[-1]['听辨'] == '听不清'
    checks = json.loads((dest / '素材/asr_raw/recovery.json').read_text())
    assert len(checks[0]['新增']) == 1
