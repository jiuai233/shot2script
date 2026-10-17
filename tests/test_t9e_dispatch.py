import json

import pytest

from test_t8_dispatch import board, action_result, save
from s2s.dispatch.materials import assemble
from s2s.dispatch.results import validate_result
from s2s.dispatch.roles import dispatch
from s2s import agent


def review():
    return {'模式': '全文审查', '集': 1, '审查范围': {'行': '1-4', '场景': ['场景1-1'], '镜头': 'E01-K001', '对白句': 'E01-S001'},
            '问题': [], '复核': [], '统计': {'阻断': 0, '规范': 0}, '结论': '通过'}


def events(result, paths):
    rows = []
    for i, path in enumerate(paths):
        rows.extend([{'type': 'tool_execution_start', 'toolCallId': str(i), 'toolName': 'read', 'args': {'path': str(path)}},
                     {'type': 'tool_execution_end', 'toolCallId': str(i), 'isError': False}])
    rows.append({'type': 'message_end', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps(result)}],
                                                   'usage': {'totalTokens': 10}}})
    return '\n'.join(json.dumps(row) for row in rows)


def test_override_prompt(board, tmp_path):
    file = tmp_path / 'prompt.md'
    file.write_text('对照提示词 {{集号}}')
    p = assemble('动作识别', board, {'片段号': 'E01-P01'}, prompt_file=file)
    assert p.prompt.startswith('对照提示词 01') and file in p.files
    assert assemble('动作识别', board, {'片段号': 'E01-P01'}).prompt.startswith('# 动作识别')


def test_mouth_visibility_and_clipped_time(board, action_result):
    p = assemble('动作识别', board, {'片段号': 'E01-P01'})
    row = action_result['镜头'][0]
    row['口型'] = [{'人声时间段': '00:00.1-00:00.9', '人物状态': [{'人物': 'E01-P01-T1', '状态': '可见'}]}]
    assert not validate_result('动作识别', action_result, board, p)
    row['口型'][0]['人声时间段'] = '00:00.1-00:01.1'
    assert '人声时间段越出当前镜头' in validate_result('动作识别', action_result, board, p)
    row['口型'][0]['人声时间段'] = '00:00.1-00:00.9'
    row['口型'][0]['人物状态'][0]['状态'] = '开口'
    assert validate_result('动作识别', action_result, board, p)


def test_text_and_visual_projection(board):
    p = assemble('验收', board, {}, review_stage='文本')
    assert not p.images and '对白隔离哨兵' in p.prompt
    save(board / '对照表.json', [{'行': 4, '类型': '△', '场景': '场景1-1', '句': [], '镜头': ['E01-K001'], '动作': []}])
    p = assemble('验收', board, {'片段号': 'E01-P01'}, review_stage='画面')
    assert len(p.images) == 2 and p.segment['id'] == 'E01-P01'
    assert '"行": 4' in p.prompt and '只报告本段问题' in p.prompt
    assert '动作隔离哨兵' not in p.prompt


def test_review_missing_reads_never_completes(board, monkeypatch):
    calls = []
    def fake(role, prompt, **kwargs):
        calls.append(prompt)
        return events(review(), [])
    monkeypatch.setattr(agent, 'run', fake)
    output = board / 'review.json'
    r = dispatch('验收', board, {}, output)
    assert len(calls) == 7  # 一次文本 + 每段三次画面
    assert not r['全文审查完成']
    coverage = json.loads(output.with_suffix('.coverage.json').read_text())
    assert all(s['状态'] == '画面未完整审查' for s in coverage['片段'])
    assert json.loads(output.read_text())['结论'] == '不通过'
    rows = [json.loads(line) for line in (board / '派发日志.jsonl').read_text().splitlines()]
    assert len(rows[1]['尝试记录']) == 3
    assert all(a['读取图片数'] == 0 and a['token用量']['totalTokens'] == 10 for a in rows[1]['尝试记录'])
    assert '重新读取全部授权帧' in calls[2]


def test_review_full_coverage_and_issue_merge(board, monkeypatch):
    result = review()
    issue = {'位置': {'场景': '场景1-1', '行': 4, '镜头': 'E01-K001', '时间': '00:00.1'}, '原文': '△ 她站着。',
             '类型': '人物指代与角色卡', '级别': '规范', '相关人物': [], '依据': '正文', '修正要求': '使用具体名'}
    result.update({'问题': [issue], '统计': {'阻断': 0, '规范': 1}, '结论': '不通过'})
    images = list((board / '素材/frames').glob('*.jpg'))
    monkeypatch.setattr(agent, 'run', lambda *a, **k: events(result, images))
    output = board / 'review.json'
    r = dispatch('验收', board, {}, output)
    assert r['全文审查完成']
    assert len(json.loads(output.read_text())['问题']) == 1
    c = json.loads(output.with_suffix('.coverage.json').read_text())
    assert c['实际读取不同帧数'] == 2


def test_action_retry_must_reread_full_set(board, action_result, monkeypatch):
    images = sorted((board / '素材/frames').glob('*.jpg'))
    responses = iter([events(action_result, images[:1]), events(action_result, images[1:]), events(action_result, images)])
    monkeypatch.setattr(agent, 'run', lambda *a, **k: next(responses))
    r = dispatch('动作识别', board, {'片段号': 'E01-P01'}, board / 'result.json')
    assert r['重答次数'] == 2 and r['画面完整']
    row = json.loads((board / '派发日志.jsonl').read_text().splitlines()[-1])
    assert [a['读取图片数'] for a in row['尝试记录']] == [1, 1, 2]


def test_merge_different_wording_keeps_evidence():
    from s2s.dispatch.roles import merge_review_issues
    issue = {'位置': {'行': 15}, '类型': '动作与对白关联', '级别': '规范', '原文': '人物（画外，低头）：对白',
             '依据': '文本检查', '修正要求': '删除低头'}
    other = {**issue, '依据': 'K007_01.jpg', '修正要求': '仅保留画外标记'}
    result = merge_review_issues([{'问题': [issue]}, {'问题': [other]}])
    assert len(result) == 1
    assert '文本检查' in result[0]['依据'] and 'K007_01.jpg' in result[0]['依据']
    assert '删除低头' in result[0]['修正要求'] and '仅保留画外标记' in result[0]['修正要求']


def test_merge_location_without_line():
    from s2s.dispatch.roles import merge_review_issues
    issue = {'位置': {'镜头': 'E01-K013', '时间': '00:16.6'}, '类型': '具体动作', '级别': '阻断',
             '原文': '缺少掉牌动作', '依据': 'K013_01.jpg', '修正要求': '补写动作'}
    assert merge_review_issues([{'问题': [issue]}, {'问题': [issue]}]) == [issue]


def test_merge_missing_sentence_neighbor_boundaries():
    from s2s.dispatch.roles import merge_review_issues
    issue = {'位置': {'行': 40, '镜头': 'E01-K023', '时间': '00:40.6'}, '类型': '全文完整性与一致性', '级别': '阻断',
             '原文': '本段缺少对应句', '依据': '声音记录S019', '修正要求': '补入台词'}
    other = {**issue, '位置': {**issue['位置'], '行': 41}, '原文': '第41行前漏S019', '依据': 'K023_06.jpg'}
    result = merge_review_issues([{'问题': [issue]}, {'问题': [other]}])
    assert len(result) == 1 and 'K023_06.jpg' in result[0]['依据']
    assert '第41行' in result[0]['原文']


def test_merge_severity_conflict_preserves_higher_level():
    from s2s.dispatch.roles import merge_review_issues
    issue = {'位置': {'行': 13}, '类型': '全文完整性与一致性', '级别': '规范',
             '原文': '资料存在冲突', '依据': '主副路不一致', '修正要求': '回听'}
    result = merge_review_issues([{'问题': [issue]}, {'问题': [{**issue, '级别': '阻断'}]}])
    assert len(result) == 1 and result[0]['级别'] == '阻断'


def test_custom_review_prompt_is_not_formal_acceptance(board, monkeypatch, tmp_path):
    prompt = tmp_path / 'review.md'
    prompt.write_text('自定义对比试验提示词')
    images = list((board / '素材/frames').glob('*.jpg'))
    monkeypatch.setattr(agent, 'run', lambda *a, **k: events(review(), images))
    output = board / 'custom-review.json'
    result = dispatch('验收', board, {}, output, prompt_file=prompt)
    assert result['全文审查完成']
    assert json.loads(output.with_suffix('.coverage.json').read_text())['模拟调用'] is True
