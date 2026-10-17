import importlib.util
import json
from pathlib import Path

import pytest

from s2s.asr.grok import GrokBackend, keyterms_from, request_fields
from s2s.asr.postprocess import seconds, words_to_sentences


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
def test_cli_rejects_remote_keyterm_sources(tmp_path, name):
    module = cli(name)
    args = ['video.mp4', str(tmp_path / 'E01')] if name == 'asr' else ['video.mp4', '0', '1']
    with pytest.raises(SystemExit):
        module.main(args + ['--keyterm-from', '角色卡.json'])


def subtitle(text='桂芝人呢', start='00:05.0', end='00:05.8', **extra):
    return dict(文字=text, 起=start, 止=end, 类型='台词字幕', **extra)


def test_erzi_is_not_an_er_suffix():
    rows = words_to_sentences([word('我的话', 0, 1), word('儿子呢？', 2, 2.2, 1)], 0, 3, 'c')
    assert [s.text for s in rows] == ['我的话', '儿子呢？']
    assert rows[0].speaker != rows[1].speaker
