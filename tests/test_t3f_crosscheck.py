"""双路对齐、优先级、复用与黑板备份。"""
import json
from pathlib import Path

import pytest

from s2s.asr import crosscheck as cross
from s2s.listen.storage import read, write
from s2s.validate import validate_file


def sound(text, start=0, end=1, ident=1, voice=1):
    def t(v):
        return f"{int(v)//60:02d}:{v%60:04.1f}"
    return {'id': f'E01-S{ident:03d}', '起': t(start), '止': t(end), '文本': text,
            '声线': f'E01-V{voice}', '听辨': '清楚'}


@pytest.mark.parametrize('left,right,state', [('你好。','你好','一致'), ('老师','老是','不一致'),
    ('电话是一百二十三号。','电话是１２３号','一致'), ('到3:40开会','到三点四十分开会','一致')])
def test_normalization(left, right, state):
    rows = cross.merge([sound(left)], [sound(right)])
    assert len(rows) == 1
    assert rows[0]['文本'] == left
    assert rows[0]['核对']['状态'] == state


def test_primary_secondary_only_and_voice_scope():
    rows = cross.merge([sound('甲', 1, 2)], [sound('乙', 3, 4)])
    assert [r['核对']['状态'] for r in rows] == ['仅主路', '仅副路']
    assert [r['id'] for r in rows] == ['E01-S001', 'E01-S002']
    assert rows[1]['声线'] != rows[0]['声线']
    assert rows[1]['核对']['优先级'] == 1


@pytest.mark.parametrize('reverse', [False, True])
def test_one_to_many(reverse):
    a = [sound('今天必须说清楚。', 1, 4)]
    b = [sound('今天必须', 1, 2), sound('说清楚', 2, 4, 2)]
    rows = cross.merge(b, a) if reverse else cross.merge(a, b)
    assert len(rows) == (2 if reverse else 1)
    assert all(r['核对']['状态'] == '一致' for r in rows)
    assert rows[0]['起'] == '00:01.0' and rows[-1]['止'] == '00:04.0'


def test_different_primary_voices_not_joined():
    rows = cross.merge([sound('甲', 0, 1), sound('乙', 1, 2, 2, 2)], [sound('甲乙', 0, 2)])
    assert len(rows) == 2
    assert {r['声线'] for r in rows} == {'E01-V1', 'E01-V2'}
    assert all(r['核对']['状态'] == '一致' for r in rows)
    assert [r['核对']['副路'] for r in rows] == ['甲','乙']


def test_keep_adjacent_sentences():
    a = [sound('甲乙', 1, 2), sound('丙丁', 2, 3, 2)]
    assert len(cross.merge(a, a)) == 2


@pytest.mark.parametrize('state,a,b,sub,expected', [
    ('不一致','abcdefghij','abcdefghiK','abcdefghij',3),
    ('不一致','甲乙丙','天地人','天地人',1),
    ('不一致','甲乙丙丁','甲乙丁丁','甲乙丙丁',2),
    ('仅主路','你好','','你好',1), ('仅副路','','你好','你好',1)])
def test_priorities(state,a,b,sub,expected):
    assert cross.priority(state,a,b,sub) == expected


def test_subtitle_reference_does_not_choose_text():
    rows = cross.merge([sound('甲')], [sound('乙')], [
        {'起':'00:00.0','止':'00:01.0','文字':'乙','类型':'台词字幕'},
        {'起':'00:00.0','止':'00:01.0','文字':'人名','类型':'人物字卡'}])
    assert rows[0]['文本'] == '甲'
    assert rows[0]['核对']['字幕'] == ''
    assert rows[0]['核对']['状态'] == '不一致'


def test_pending_update_no_duplicates():
    rows = cross.merge([sound('甲')], [sound('乙')])
    first = cross.update_pending(rows, [], 'E01')
    second = cross.update_pending(rows, first, 'E01')
    assert second == first and len(first) == 1
    assert '主路：甲\n副路：乙' == first[0]['依据']
    assert first[0]['级别'] == '阻断'
    rows[0]['核对']['状态'] = '一致'
    assert cross.update_pending(rows, second, 'E01')[0]['状态'] == '已裁定'


def fake_audio(monkeypatch):
    import numpy as np
    from s2s import vad
    monkeypatch.setattr(cross.audio, 'extract_audio', lambda _source, target: target)
    monkeypatch.setattr(vad, 'waveform', lambda _: np.zeros(64000))
    monkeypatch.setattr(vad, 'detect', lambda *args: [])


def raw(tmp_path, name, rows):
    path = tmp_path/name
    write(path, {'words':[{'text':r['文本'],'start':float(r['起'][3:]),'end':float(r['止'][3:])} for r in rows]})
    return path


def config(monkeypatch):
    monkeypatch.setattr(cross, 'load_config', lambda: {'asr': {'provider':'other', 'cross_check':
        {'enabled':True, 'primary':'local', 'secondary':'grok'}}})


def test_reuse_never_calls_backends(tmp_path, monkeypatch):
    config(monkeypatch)
    monkeypatch.setattr(cross, 'get_backend', lambda *_: pytest.fail('结果已提供，不得识别'))
    episode=tmp_path/'E01'
    episode.mkdir()
    write(episode/'声音记录.json', [sound('原始')])
    write(tmp_path/'a.json',[sound('甲',.2,.8)]); write(tmp_path/'b.json',[sound('乙',.2,.8)])
    fake_audio(monkeypatch)
    result=cross.run('不存在的视频',episode,tmp_path/'a.json',tmp_path/'b.json',
                     primary_words=raw(tmp_path,'aw.json',read(tmp_path/'a.json')),
                     secondary_words=raw(tmp_path,'bw.json',read(tmp_path/'b.json')))
    assert result['主路']=='local' and result['统计']['不一致']==1
    assert read(Path(result['备份'])/'声音记录.json')[0]['文本']=='原始'
    assert all(validate_file(episode/name)['通过'] for name in ('声音记录.json','待核对.json'))


def test_missing_result_calls_configured_backend_once(tmp_path, monkeypatch):
    from s2s.asr.base import Sentence
    config(monkeypatch)
    calls=[]
    class Backend:
        settings={}
        def transcribe(self, wav):
            calls.append(str(wav)); self.last_words=[{'text':'甲','start':.2,'end':.8}]; return [Sentence('00:00.2','00:00.8','甲','')]
    providers=[]
    def backend(name):
        providers.append(name); return Backend()
    monkeypatch.setattr(cross,'get_backend',backend)
    monkeypatch.setattr(cross.audio,'extract_audio',lambda _source,target: target)
    write(tmp_path/'b.json',[sound('甲',.2,.8)])
    fake_audio(monkeypatch)
    cross.run('input.mp4',tmp_path/'E01',secondary_path=tmp_path/'b.json',secondary_words=raw(tmp_path,'bw.json',read(tmp_path/'b.json')))
    assert providers==['local'] and len(calls)==1


def test_disabled_crosscheck_fails_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(cross, 'load_config', lambda: {'asr': {'provider':'grok'}})
    with pytest.raises(ValueError, match='enabled'):
        cross.run('video',tmp_path/'E01')


def test_long_different_secondary_does_not_swallow_exact_match():
    rows = cross.merge([sound('你好',0,2)], [sound('你好',0,1),sound('别的很长内容'*20,1,2,2)])
    assert len(rows)==2
    assert rows[0]['核对']['状态']=='一致'
    assert rows[1]['核对']['状态']=='仅副路'


def test_split_preserves_original_secondary_characters():
    primary=[sound('我提醒一句。',0,1),sound('接个孩子而已，',1,2,2,2)]
    reference='我提醒一句，接个孩子而已。'
    rows=cross.merge(primary,[sound(reference,0,2)])
    assert len(rows)==2 and all(r['核对']['状态']=='一致' for r in rows)
    assert ''.join(r['核对']['副路'] for r in rows)==reference
    assert [(r['起'],r['止'],r['声线']) for r in rows]==[(r['起'],r['止'],r['声线']) for r in primary]


def test_simultaneous_sentences_have_separate_pending_items():
    rows=cross.merge([sound('甲',0,1,1),sound('乙',0,1,2,2)],[])
    first=cross.update_pending(rows,[],'E01')
    assert len(first)==2
    assert cross.update_pending(rows,first,'E01')==first


def test_inserting_secondary_remaps_existing_non_cross_issue(tmp_path,monkeypatch):
    config(monkeypatch)
    episode=tmp_path/'E01'; episode.mkdir()
    old=[sound('原句',2,3)]
    write(episode/'声音记录.json',old)
    pending=cross.update_pending(cross.merge(old,[]),[],'E01')
    pending[0]['来源']='验收'; pending[0]['处理职责']='成稿'
    write(episode/'待核对.json',pending)
    write(tmp_path/'a.json',old)
    write(tmp_path/'b.json',[sound('新增',0,1),sound('原句',2,3,2)])
    fake_audio(monkeypatch)
    cross.run('不存在的视频',episode,tmp_path/'a.json',tmp_path/'b.json',
              primary_words=raw(tmp_path,'aw.json',read(tmp_path/'a.json')),
              secondary_words=raw(tmp_path,'bw.json',read(tmp_path/'b.json')))
    issue=next(r for r in read(episode/'待核对.json') if r['来源']=='验收')
    assert issue['位置']['句']=='E01-S002'


def test_point_subtitle_is_included_as_reference():
    rows=cross.merge([sound('你好',0,2)],[sound('你好',0,2)],[
        {'起':'00:01.0','止':'00:01.0','文字':'你好','类型':'台词字幕'}])
    assert rows[0]['核对']['字幕']==''
