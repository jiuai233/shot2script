import copy
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


def test_review_picture_stage_sends_all_frames_directly(board, monkeypatch):
    calls = []
    def fake(role, prompt, **kwargs):
        calls.append((prompt, kwargs))
        return events(review(), [])
    monkeypatch.setattr(agent, 'run', fake)
    output = board / 'review.json'
    r = dispatch('验收', board, {}, output)
    assert len(calls) == 3 and r['全文审查完成']
    text, pictures = calls[0], calls[1:]
    assert text[1]['runner'] == 'pi' and text[1]['images'] == [] and text[1]['labels'] is None
    for prompt, kwargs in pictures:
        assert kwargs['runner'] == 'deepseek' and kwargs['events']
        assert len(kwargs['images']) == len(kwargs['labels']) == 2
        assert '查看全部附图' in prompt and '只能用 read' not in prompt
    coverage = json.loads(output.with_suffix('.coverage.json').read_text())
    assert all(s['状态'] == '完成' and s['已读帧数'] == s['授权帧数'] == 2 for s in coverage['片段'])


def test_recheck_sends_only_changed_scene_frames_directly(board, monkeypatch):
    save(board / '对照表.json', [{'行': 2, '类型': '场景行', '场景': '场景1-1', '句': [], '镜头': ['E01-K001'], '动作': []},
                                {'行': 4, '类型': '△', '场景': '场景1-1', '句': [], '镜头': ['E01-K001'], '动作': []}])
    save(board / '问题清单.json', [{'id': 'E01-Q001', '来源': '验收', '位置': {'行': 4}, '原文': '△ 穿紫衣的小女孩站着。',
                                  '类型': '具体动作', '级别': '阻断', '相关人物': [], '依据': 'K001_01.jpg',
                                  '修正要求': '补写动作', '处理职责': '成稿', '状态': '已修正', '复核结果': None}])
    calls = []
    def fake(role, prompt, **kwargs):
        calls.append((prompt, kwargs))
        return events({**review(), '模式': '复核'}, [])
    monkeypatch.setattr(agent, 'run', fake)
    try:
        dispatch('验收', board, {}, board / 'recheck.json', mode='复核', scope='第4行')
    except RuntimeError:
        pass
    prompt, kwargs = calls[0]
    assert kwargs['runner'] == 'deepseek' and kwargs['labels'] == ['K001_01.jpg｜镜头 E01-K001｜00:00.1']
    assert [p.name for p in kwargs['images']] == ['K001_01.jpg'] and '只能用 read' not in prompt


def test_recheck_shots_adds_neighbor_scenes():
    from s2s.dispatch.materials import recheck_shots
    mapping = [{'行': n, '场景': f'场景1-{n}', '镜头': [f'E01-K00{n}']} for n in range(1, 5)]
    shots = [{'id': f'E01-K00{n}'} for n in range(1, 5)]
    issues = [{'状态': '已修正', '位置': {'镜头': 'E01-K003'}}, {'状态': '待修正', '位置': {'行': 1}}]
    assert [s['id'] for s in recheck_shots(mapping, shots, issues, '')] == ['E01-K002', 'E01-K003', 'E01-K004']
    assert [s['id'] for s in recheck_shots(mapping, shots, [], '第 1 至 1 行')] == ['E01-K001', 'E01-K002']
    with pytest.raises(ValueError, match='未能'):
        recheck_shots(mapping, shots, issues[1:], '全文')


def test_static_rules_move_to_system_and_convention_is_trimmed(board):
    p = assemble('动作识别', board, {'片段号': 'E01-P01'})
    assert '## 数据约定' not in p.prompt and '## 返回包精确 Schema' in p.system
    assert '### 4.4 动作条目' in p.system and '## 2. 黑板目录' not in p.system and '### 4.15' not in p.system
    assert p.signature()['系统'] == p.system



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


def test_action_sends_one_frame_per_second(board, action_result):
    times = ['00:00.1', '00:00.5', '00:01.0', '00:01.6', '00:02.0', '00:02.4', '00:02.9']
    shots = [{'id': 'E01-K001', '起': '00:00.0', '止': '00:03.0',
              '帧': [{'文件': f'K001_{i:02d}.jpg', '时间': t} for i, t in enumerate(times, 1)]},
             {'id': 'E01-K002', '起': '00:03.0', '止': '00:03.4',
              '帧': [{'文件': f'K002_{i:02d}.jpg', '时间': t} for i, t in enumerate(('00:03.1', '00:03.2', '00:03.3'), 1)]}]
    save(board / '镜头表.json', {'集': 1, '视频': 'video.mp4', '时长': '00:03.4', '镜头': shots})
    save(board / '片段表.json', {'集': 1, '片段': [
        {'id': 'E01-P01', '镜头': ['E01-K001'], '起': '00:00.0', '止': '00:03.0', '衔接前': None, '衔接后': 'E01-K002'}]})
    for shot in shots:
        for frame in shot['帧']:
            (board / '素材/frames' / frame['文件']).write_bytes(b'fake-frame')
    p = assemble('动作识别', board, {'片段号': 'E01-P01'})
    assert [path.name for path in p.images] == ['K001_02.jpg', 'K001_04.jpg', 'K001_06.jpg', 'K002_02.jpg']
    assert p.labels[-1] == 'K002_02.jpg｜镜头 E01-K002｜00:03.2｜衔接参考'
    assert 'K001_01.jpg' not in p.prompt
    action_result['镜头'][0].update({'起': '00:00.0', '止': '00:03.0', '动作': [
        {'起': '00:00.5', '止': '00:01.6', '执行者': 'E01-P01-T1', '动作': '向前走', '对象': None, '物件': None,
         '结果': None, '关键': False, '关键类别': None, '依据帧': ['K001_02.jpg', 'K001_04.jpg'], '确定度': '确认', '说明': ''}]})
    person = action_result['镜头'][0]['画面人物'][0]
    person['依据帧'] = ['K001_02.jpg']
    assert not validate_result('动作识别', action_result, board, p)
    person['依据帧'] = ['K001_01.jpg']
    assert validate_result('动作识别', action_result, board, p) == ['E01-K001 引用了非本镜头依据帧']
    person['依据帧'] = ['K001_02.jpg']
    action_result['镜头'][0]['动作'][0]['依据帧'] = ['K001_01.jpg']
    assert validate_result('动作识别', action_result, board, p) == ['E01-K001 引用了非本镜头依据帧']


def test_action_rejects_mostly_idle_shots(board, action_result):
    shots = [{'id': f'E01-K{i:03d}', '起': f'00:0{i - 1}.0', '止': f'00:0{i}.0',
              '帧': [{'文件': f'K{i:03d}_01.jpg', '时间': f'00:0{i - 1}.5'}]} for i in range(1, 9)]
    save(board / '镜头表.json', {'集': 1, '视频': 'video.mp4', '时长': '00:08.0', '镜头': shots})
    save(board / '片段表.json', {'集': 1, '片段': [
        {'id': 'E01-P01', '镜头': [s['id'] for s in shots], '起': '00:00.0', '止': '00:08.0', '衔接前': None, '衔接后': None}]})
    for shot in shots:
        (board / '素材/frames' / shot['帧'][0]['文件']).write_bytes(b'fake-frame')
    p = assemble('动作识别', board, {'片段号': 'E01-P01'})
    template = action_result['镜头'][0]
    action_result['镜头'] = []
    for shot in shots:
        row = copy.deepcopy(template)
        row.update({'镜头': shot['id'], '起': shot['起'], '止': shot['止'], '动作': []})
        row['画面人物'][0]['依据帧'] = [shot['帧'][0]['文件']]
        action_result['镜头'].append(row)
    errors = validate_result('动作识别', action_result, board, p)
    assert len(errors) == 1 and '8 个镜头中 8 个没有动作条目' in errors[0]
    for row, shot in zip(action_result['镜头'][:3], shots):
        row['动作'] = [{'起': shot['起'], '止': shot['止'], '执行者': 'E01-P01-T1', '动作': '向前走', '对象': None,
                       '物件': None, '结果': None, '关键': False, '关键类别': None,
                       '依据帧': [shot['帧'][0]['文件']], '确定度': '确认', '说明': ''}]
    assert not validate_result('动作识别', action_result, board, p)


def test_action_retry_resends_all_frames(board, action_result, monkeypatch):
    images = [p.resolve() for p in sorted((board / '素材/frames').glob('*.jpg'))]
    responses = iter(['不是 JSON', events(action_result, [])])
    calls = []
    def fake(role, prompt, **kwargs):
        calls.append((prompt, kwargs))
        return next(responses)
    monkeypatch.setattr(agent, 'run', fake)
    r = dispatch('动作识别', board, {'片段号': 'E01-P01'}, board / 'result.json')
    assert r['重答次数'] == 1 and r['画面完整'] and r['已读帧数'] == 2
    assert all(kwargs['images'] == images for _, kwargs in calls)
    assert calls[0][1]['labels'] == ['K001_01.jpg｜镜头 E01-K001｜00:00.1', 'K002_01.jpg｜镜头 E01-K002｜00:01.1｜衔接参考']
    assert '必须重新查看全部附图' in calls[1][0]


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
