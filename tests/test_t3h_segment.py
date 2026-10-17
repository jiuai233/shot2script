"""共享音频句界、词归属、静音补查与本地检测协议。"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2s.asr.segment import boundaries, load_words, missing_voice, prepare_words, segment_pair
from s2s.asr.crosscheck import priority
from s2s.vad import waveform


def word(text, start, end, **kwargs):
    return dict(text=text, start=start, end=end, **kwargs)


def wave(total=5):
    samples = np.ones(round(total * 16000), dtype=np.float32) * .2
    samples[round(1.8 * 16000):round(2.2 * 16000)] = 0
    return samples


def test_shared_boundaries_do_not_follow_old_word_pause_or_punctuation():
    left = [word('奶奶，', .1, .2), word('奶奶，你等', 1, 1.7), word('等我。', 2.2, 2.3)]
    right = [word('奶奶奶奶你等等我！', .1, 2.4)]
    original = deepcopy(left)
    rows, spans = segment_pair(left, right, [{"起": .06, "止": 2.4}], wave(3))
    assert len(rows) == 1 and len(spans) == 1
    assert rows[0]['文本'] == '奶奶，奶奶，你等等我。'
    assert rows[0]['核对']['状态'] == '一致'
    assert left == original


def test_only_one_lane_kept_and_silent_vad_not_fabricated():
    rows, spans = segment_pair([word('甲', .2, .9)], [word('乙', 3.2, 3.8)],
                              [{'起': .2, '止': 1}, {'起': 2, '止': 2.5}, {'起': 3, '止': 4}], wave())
    assert len(rows) == 2 and len(spans) == 3
    assert [r['核对']['状态'] for r in rows] == ['仅主路', '仅副路']
    assert rows[0]['声线'] != rows[1]['声线']


def test_min_pause_configurable_and_point_is_quiet():
    samples = np.ones(64000) * .3
    samples[round(1.2*16000):round(1.3*16000)] = 0
    voices = [{'起': .2, '止': 1}, {'起': 1.6, '止': 2}]
    assert len(boundaries(voices, [[], []], samples, min_pause=.7)) == 1
    spans = boundaries(voices, [[], []], samples, min_pause=.5)
    assert len(spans) == 2
    assert 1 <= spans[0][1] <= 1.6


def test_long_sentence_splits_at_longest_actual_pause():
    samples = np.ones(160000) * .2
    left = [word('甲，', .2, 2), word('乙，', 2.2, 4), word('丙。', 4.4, 5), word('丁', 5.4, 9)]
    voices = [{'起': .2, '止': 2}, {'起': 2.2, '止': 4}, {'起': 4.4, '止': 5}, {'起': 5.4, '止': 9}]
    rows, spans = segment_pair(left, [], voices, samples)
    assert len(rows) >= 2
    assert any(5 <= a <= 5.4 for a, _ in spans[1:])
    assert ''.join(r['文本'] for r in rows) == '甲，乙，丙。丁'


def test_no_pause_never_cuts_long_word_for_duration_limit():
    rows, spans = segment_pair([word('不能硬切的长词。', 0, 9)], [], [{'起': 0, '止': 9}], np.ones(160000))
    assert len(rows) == 1 and rows[0]['截断'] is True
    assert float(rows[0]['止'][3:]) >= 9


def test_word_crossing_short_vad_pause_still_prevents_cut():
    left = [word('跨界', .8, 1.6)]
    rows, spans = segment_pair(left, [word('跨界', .9, 1.7)],
                              [{'起': .5, '止': 1}, {'起': 1.4, '止': 2.5}], wave())
    assert len(rows) == 1
    assert all(not (.8 < edge < 1.6) for span in spans for edge in span)


@pytest.mark.parametrize('swap', [False, True])
def test_word_fully_outside_voice_does_not_create_or_extend_sentence(swap):
    left = [word('赶紧', .2, .8), word('的。', 2.2, 2.4), word('来了。', 3.2, 3.8)]
    right = [word('赶紧的。', .2, .9), word('来了。', 3.2, 3.8)]
    original = deepcopy(left)
    rows, spans = segment_pair(*( (right, left) if swap else (left, right)),
                              [{'起': .1, '止': 1}, {'起': 3, '止': 4}], wave())
    assert len(spans) == len(rows) == 2
    assert rows[0]['核对']['主路'] == rows[0]['核对']['副路'] == '赶紧的。'
    assert rows[1]['核对']['状态'] == '一致'
    assert left == original


def test_unreliable_leading_word_follows_next_text_not_its_wrong_time():
    rows, spans = segment_pair([word('快', .1, .2), word('来了。', 2.2, 2.8)], [],
                              [{'起': 2, '止': 3}], wave())
    assert len(rows) == len(spans) == 1
    assert rows[0]['文本'] == '快来了。'
    assert rows[0]['起'] >= '00:01.8'


@pytest.mark.parametrize('crossing,expected', [((.2, 2.1), 0), ((.9, 2.9), 1)])
@pytest.mark.parametrize('pause', [.5, 1.0])
def test_word_spanning_long_pause_goes_to_side_with_more_time(crossing, expected, pause):
    a, b = crossing
    right_start = 1 + pause
    samples = np.ones(64000) * .2
    samples[16000:round(right_start * 16000)] = 0
    left = [word('甲', .2, .5), word('等', a, b), word('乙', 2.5, 2.9)]
    rows, spans = segment_pair(left, [word('甲', .2, .5), word('乙', 2.5, 2.9)],
                              [{'起': .1, '止': 1}, {'起': right_start, '止': 3}], samples)
    assert len(rows) == len(spans) == 2
    assert '等' in rows[expected]['文本']
    assert sum(r['文本'].count('等') for r in rows) == 1
    assert float(rows[0]['止'][3:]) < right_start


def test_two_lanes_same_word_same_time_preserve_vad_missed_speech():
    voices = [{'起': .06, '止': .49}, {'起': 3.2, '止': 4}]
    left = [word('奶奶，', .16, .24), word('奶', .96, 1.04), word('奶，', 1.2, 1.28),
            word('你等', 1.5, 1.7), word('等我。', 2.24, 2.32)]
    right = [word('奶', 1.02, 1.18), word('奶', 1.26, 1.34), word('你', 1.46, 1.56),
             word('等', 1.66, 1.88), word('等', 1.96, 2.2), word('我。', 2.26, 2.36)]
    rows, _ = segment_pair(left, right, voices, wave())
    assert rows[0]['文本'] == '奶奶，奶奶，你等等我。'
    assert float(rows[0]['止'][3:]) >= 2.4


def test_fully_unreliable_lane_keeps_text_with_other_lane_voice_sentence():
    rows, _ = segment_pair([word('不丢', 2, 2.2), word('字。', 2.3, 2.4)],
                          [word('不丢字。', .2, .9)], [{'起': .1, '止': 1}], wave())
    assert len(rows) == 1 and rows[0]['核对']['状态'] == '一致'


def test_both_edges_truncated_without_adding_words():
    rows, _ = segment_pair([word('已有字', 0, 1.95)], [], [{'起': 0, '止': 2}], np.ones(32000))
    assert rows[0]['截断'] is True and rows[0]['文本'] == '已有字'
    assert rows[0]['听辨'] == '清楚'


def test_fractional_episode_end_stays_within_diarize_tolerance():
    total = 1.938
    rows, _ = segment_pair([word('尾句', 1.7, 1.93)], [], [{'起': 1.7, '止': total}],
                          np.ones(round(total * 16000)))
    assert rows[-1]['止'] == '00:01.9'
    assert rows[-1]['截断'] is True
    assert abs(float(rows[-1]['止'][3:]) - total) <= .05


def test_punctuation_attaches_without_empty_sentence():
    words = prepare_words([word('你', .1, .4), word('。', .4, .4)], 2)
    assert len(words) == 1 and words[0]['text'] == '你。'
    with pytest.raises(ValueError, match='零长实字'):
        prepare_words([word('你', .4, .4)], 2)


@pytest.mark.parametrize('setting,value', [('min_pause', 0), ('min_pause', True), ('max_sentence', float('nan'))])
def test_bad_parameters(setting, value):
    with pytest.raises(ValueError):
        boundaries([], [], wave(), **{setting: value})


def test_words_window_offset_and_missing_raw_rejected(tmp_path):
    path = tmp_path/'raw.json'
    path.write_text(json.dumps({'window': [10, 20], 'response': {'words': [word('字', 1, 2)]}}))
    assert load_words(path) == [word('字', 11, 12)]
    path.write_text(json.dumps({'文本': '只剩句级'}))
    with pytest.raises(ValueError, match='不能从句级'):
        load_words(path)


def test_missing_audio_trigger_requires_both_lanes_absent():
    voices = [{'起': 1, '止': 2}, {'起': 3, '止': 4}, {'起': 5, '止': 6}]
    missing = missing_voice(voices, [word('甲', 1, 2)], [word('乙', 3, 4)])
    assert missing == [{'起': 5, '止': 6}]


@pytest.mark.parametrize('sub', ['', '这份字幕正确', '这份字幕完全不同'])
def test_priority_cannot_depend_on_subtitles(sub):
    assert priority('不一致', 'abcdefghij', 'abcdefghiK', sub) == 3
    assert priority('仅主路', '甲', '', sub) == 1
    assert priority('一致', '甲', '甲', sub, duration=.1) == 1
    assert priority('一致', '甲', '甲', sub, truncated=True) == 1


def test_vad_subprocess_runs_only_vad_and_times_out_without_retry(tmp_path, monkeypatch):
    from s2s import vad
    model = tmp_path/'model'; model.mkdir()
    for name in ['model_quant.onnx', 'config.yaml', 'am.mvn']: (model/name).touch()
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        assert args[1:3] == ['-m', 's2s.vad']
        assert 'local_runtime' not in ' '.join(args)
        return SimpleNamespace(returncode=0, stdout='[{"起":0.1,"止":0.9}]')
    monkeypatch.setattr(vad.subprocess, 'run', run)
    monkeypatch.setattr(vad, 'waveform', lambda _: np.zeros(16000))
    assert vad.detect('audio.wav', {'model_path': model}) == [{'起': .1, '止': .9}]
    def timeout(*args, **kwargs):
        calls.append(args)
        raise vad.subprocess.TimeoutExpired('vad', 1)
    monkeypatch.setattr(vad.subprocess, 'run', timeout)
    with pytest.raises(ValueError, match='超时'):
        vad.detect('audio.wav', {'model_path': model, 'timeout_seconds': 1})
    assert len(calls) == 2


def test_local_keeps_real_words_on_disk(tmp_path, monkeypatch):
    from s2s.asr.local import LocalBackend
    backend = LocalBackend({'model': 'qwen3-asr-0.6b', 'raw_dir': str(tmp_path/'raw')})
    monkeypatch.setattr('s2s.asr.local.audio.duration', lambda _: 20)
    monkeypatch.setattr('s2s.asr.local.audio.extract_audio', lambda *args: args[1])
    monkeypatch.setattr(backend, '_words', lambda _: [word('你好', .1, 1)])
    backend.transcribe('audio.wav', 10, 12)
    assert backend.last_words == [word('你好', 10.1, 11)]
    assert load_words(next((tmp_path/'raw').glob('*.json'))) == backend.last_words


def test_subtitles_cannot_masquerade_as_vad():
    with pytest.raises(ValueError, match='不接受字幕'):
        segment_pair([word('音频', .2, .8)], [], [{'起':.1,'止':1,'文字':'字幕','类型':'台词字幕'}], wave())


def test_missing_original_words_fails_before_backend_or_audio(tmp_path, monkeypatch):
    from s2s.asr import crosscheck
    monkeypatch.setattr(crosscheck, 'load_config', lambda: {'asr':{'cross_check':{'enabled':True,'primary':'local','secondary':'grok'}}})
    monkeypatch.setattr(crosscheck, 'get_backend', lambda *args: pytest.fail('不得识别'))
    monkeypatch.setattr(crosscheck.audio, 'extract_audio', lambda *args: pytest.fail('缺词时不读取音频'))
    with pytest.raises(ValueError, match='不会重新识别补齐'):
        crosscheck.run('missing.mp4', tmp_path/'E02', primary_path=tmp_path/'声音记录.json', secondary_path=tmp_path/'副路.json')
    assert not (tmp_path/'E02').exists()
