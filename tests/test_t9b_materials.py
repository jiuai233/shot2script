import json

import pytest

from test_t8_dispatch import board, action_result, save
from s2s.dispatch.materials import MATERIALS, assemble
from s2s.evidence import sound_snapshot


def authorized_data(package):
    return json.JSONDecoder().raw_decode(package.prompt.split('## 本次授权资料\n', 1)[1])[0]


def test_role_card_includes_voice_evidence_not_library(board, action_result):
    save(board / '派发结果/动作识别/E01-P01.json', action_result)
    sounds = json.loads((board / '声音记录.json').read_text())
    stamp = {k: sounds[0][k] for k in ('起', '止', '文本', '声线')}
    evidence = {
        '口型记录': {'句': [{'句': 'E01-S001', **stamp, '结论': '存疑', '复核': None, '轨迹': []}], '声线配对': []},
        '声纹聚类': {'句列表': [{'句': 'E01-S001', **stamp, '待定': True, '合并依据': ['max_speakers']}],
                 '约束': {'冲突': ['聚类冲突哨兵']}},
        '声纹候选': {'声音摘要': sound_snapshot(sounds), '仅供参考': True, '声线': [{'标签': 'E01-V1', '候选': [{'角色': 'C003', '余弦相似度': 0.9}]}]},
    }
    for name, value in evidence.items():
        save(board / (name + '.json'), value)
    library = board.parent / '声纹/C003.json'
    save(library, {'向量': ['原始向量不得泄漏'], '来源': [{'均值': ['集均值不得泄漏']}]})
    package = assemble('角色卡', board, {}, mode='建立')
    data = authorized_data(package)
    for name, value in evidence.items():
        assert data[name] == value
        assert board / (name + '.json') in package.files
    assert library not in package.files
    assert '原始向量不得泄漏' not in package.prompt
    assert '集均值不得泄漏' not in package.prompt
    assert '动作隔离哨兵' not in package.prompt
    assert '声纹候选:存在时' in MATERIALS['角色卡']


def test_lipsync_projection_preserves_conflicts_without_curves(board, action_result):
    save(board / '派发结果/动作识别/E01-P01.json', action_result)
    track = {'id': 'E01-K001-F1', '轨迹组': 'G1', '框': {'x': 0.1},
             '状态': '存疑', '说话分数': 0.6, '模型分数': 0.8,
             '评分依据': {'评分冲突': True}, '不可用原因': ['遮挡'],
             '依据帧': ['K001_01.jpg'], '曲线': [{'音量': 0.01}],
             '模型曲线': [{'说话概率': 0.9}]}
    row = {'句': 'E01-S001', '声线': 'E01-V1', '起': '00:00.1', '止': '00:00.9', '文本': '对白隔离哨兵',
           '结论': '存疑', '轨迹': [track], '复核': {'画面结论': '画内', '与代码一致': False}}
    pairs = [{'声线': 'E01-V1', '轨迹组': 'G1', '支持句数': 1, '反例句数': 1}]
    save(board / '口型记录.json', {'句': [row], '声线配对': pairs, '统计': {'耗时秒': 5}})
    data = authorized_data(assemble('角色卡', board, {}))['口型记录']
    assert data['句'][0] == {**row, '轨迹': [{k: v for k, v in track.items() if k not in ('曲线', '模型曲线')}]}
    assert data['声线配对'] == pairs
    assert '统计' not in data


def test_missing_optional_voice_evidence_stays_absent(board, action_result):
    save(board / '派发结果/动作识别/E01-P01.json', action_result)
    package = assemble('角色卡', board, {}, mode='建立')
    data = authorized_data(package)
    for name in ('口型记录', '声纹聚类', '声纹候选'):
        assert name not in data
        assert board / (name + '.json') not in package.files


@pytest.mark.parametrize('role', ['动作识别', '成稿', '验收'])
def test_voice_evidence_not_exposed_to_other_roles(board, role):
    for name in ('口型记录', '声纹聚类', '声纹候选'):
        save(board / (name + '.json'), {'哨兵': '角色卡专用证据哨兵'})
    package = assemble(role, board, {'片段号': 'E01-P01'} if role == '动作识别' else {})
    assert '角色卡专用证据哨兵' not in package.prompt
    assert not any(p.stem in ('口型记录', '声纹聚类', '声纹候选') for p in package.files)
