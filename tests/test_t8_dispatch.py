import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from s2s.dispatch.controllers import episode_prompt, mark_step, start_controller
from s2s.dispatch.materials import assemble, render
from s2s.dispatch.results import extract_json, extract_result, validate_result
from s2s.dispatch.roles import DispatchError, dispatch
from s2s.dispatch.transport import decode_response, pi_options


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


@pytest.fixture
def board(tmp_path):
    episode = tmp_path / '测试剧/E01'
    episode.mkdir(parents=True)
    save(episode.parent / '角色卡.json', [])
    save(episode / '镜头表.json', {'集': 1, '视频': 'video.mp4', '时长': '00:02.0', '镜头': [
        {'id': 'E01-K001', '起': '00:00.0', '止': '00:01.0', '帧': [{'文件': 'K001_01.jpg', '时间': '00:00.1'}]},
        {'id': 'E01-K002', '起': '00:01.0', '止': '00:02.0', '帧': [{'文件': 'K002_01.jpg', '时间': '00:01.1'}]}]})
    save(episode / '片段表.json', {'集': 1, '片段': [
        {'id': 'E01-P01', '镜头': ['E01-K001'], '起': '00:00.0', '止': '00:01.0', '衔接前': None, '衔接后': 'E01-K002'},
        {'id': 'E01-P02', '镜头': ['E01-K002'], '起': '00:01.0', '止': '00:02.0', '衔接前': 'E01-K001', '衔接后': None}]})
    save(episode / '声音记录.json', [{'id': 'E01-S001', '起': '00:00.1', '止': '00:00.9', '文本': '对白隔离哨兵', '声线': 'E01-V1', '听辨': '清楚'}])
    save(episode / '字幕记录.json', [{'起': '00:00.1', '止': '00:00.9', '文字': '台词字幕隔离哨兵', '类型': '台词字幕'},
                                  {'起': '00:00.1', '止': '00:00.9', '文字': '地点文字', '类型': '地点字幕'}])
    save(episode / '动作记录.json', [{'动作': '动作隔离哨兵', '确定度': '确认'}])
    save(episode / '归属表.json', [{'依据': '归属隔离哨兵', '确定度': '确认'}])
    for name in ('对照表', '待核对', '问题清单'):
        save(episode / f'{name}.json', [])
    (episode / '工作稿.md').write_text('第1集\n场景1-1：门口，日，外\n人物：穿紫衣的小女孩\n△ 穿紫衣的小女孩站着。', encoding='utf-8')
    for name in ('K001_01.jpg', 'K002_01.jpg'):
        file = episode / '素材/frames' / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'fake-frame')
    return episode


@pytest.fixture
def action_result():
    return {'片段': 'E01-P01', '模式': '片段识别', '镜头': [
        {'镜头': 'E01-K001', '起': '00:00.0', '止': '00:01.0',
         '场景信息': {'地点': '门口', '日夜': '日', '内外': '外', '闪回迹象': None},
         '画面人物': [{'人物': 'E01-P01-T1', '外观': '紫衣双马尾小女孩', '位置': '左侧'}],
         '动作': [], '神态': [], '口型': [], '画面文字': [], '说明': '无明显动作'}],
        '衔接': {'承接前段': '', '延续到后段': ''}, '加密取帧请求': [], '待核对': []}


def test_render():
    assert render('{{剧名}} {{ 集号 }}', {'剧名': '测试', '集号': '01'}) == '测试 01'
    with pytest.raises(ValueError, match='未替换占位'):
        render('{{缺失}}', {})
    with pytest.raises(ValueError, match='仍含占位'):
        render('{{值}}', {'值': '{{未填}}'})


def test_action_isolation_and_segment(board):
    p = assemble('动作识别', board, {'片段号': 'E01-P01'})
    assert '对白隔离哨兵' not in p.prompt
    assert '台词字幕隔离哨兵' not in p.prompt
    assert '动作隔离哨兵' not in p.prompt
    assert '地点文字' in p.prompt
    assert len(p.images) == 2  # 本段与后一段衔接。
    assert p.segment['镜头'] == ['E01-K001']
    (board / '声音记录.json').unlink()
    assert '"人声时间段": []' in assemble('动作识别', board, {'片段号': 'E01-P01'}).prompt


def test_review_isolation(board):
    p = assemble('验收', board, {})
    assert '动作隔离哨兵' not in p.prompt and '归属隔离哨兵' not in p.prompt
    assert all(path.name not in ('动作记录.json', '归属表.json') for path in p.files)
    assert '对白隔离哨兵' in p.prompt and '台词字幕隔离哨兵' in p.prompt
    assert len(p.images) == 2


def test_draft_has_no_frames(board):
    p = assemble('成稿', board, {})
    assert not p.images
    assert 'K001_01.jpg' not in p.prompt.split('## 本次授权资料')[1]
    assert '动作隔离哨兵' in p.prompt
    assert '归属隔离哨兵' in p.prompt


def test_role_card_projects_no_actions(board, action_result):
    save(board / '派发结果/动作识别/E01-P01.json', action_result)
    p = assemble('角色卡', board, {})
    assert '人物与口型' in p.prompt and '紫衣双马尾小女孩' in p.prompt
    assert '动作隔离哨兵' not in p.prompt
    assert '无明显动作' not in p.prompt


@pytest.mark.parametrize('text,expected', [('{"x": 1}', {'x': 1}),
    ('说明\n```json\n[{"x":1}]\n```', [{'x': 1}]),
    ('结果：{"x":{"y":1}} 完成', {'x': {'y': 1}})])
def test_extract(text, expected):
    assert extract_json(text) == expected


@pytest.mark.parametrize('text', ['不是JSON', '{} {}', '```json\n{}\n```\n```json\n[]\n```'])
def test_extract_bad(text):
    with pytest.raises(ValueError):
        extract_json(text)


def test_draft_two_parts():
    result = extract_result('成稿', '**工作稿**\n```markdown\n第1集\n场景1-1：门口，日，外\n```\n**对照表**\n```json\n[]\n```')
    assert result['工作稿'].startswith('第1集') and result['对照表'] == []


def test_validation(board, action_result):
    p = assemble('动作识别', board, {'片段号': 'E01-P01'})
    assert validate_result('动作识别', action_result, board, p) == []
    broken = copy.deepcopy(action_result)
    broken['镜头'][0]['动作'] = [{'执行者': 'E01-P01-T1', '动作': '站着', '对象': None,
         '物件': None, '结果': None, '起': '00:00.1', '止': '00:00.9', '关键': False,
         '关键类别': None, '依据帧': ['K001_01.jpg'], '确定度': '确认', '说明': ''}]
    assert validate_result('动作识别', broken, board, p) == []
    broken['镜头'][0]['动作'][0]['依据帧'] = ['K002_01.jpg']
    assert any('非本镜头依据帧' in e for e in validate_result('动作识别', broken, board, p))
    broken = copy.deepcopy(action_result)
    broken['镜头'][0]['镜头'] = 'E01-K002'
    assert validate_result('动作识别', broken, board, p)
    assert validate_result('动作识别', {'片段': 'E01-P01'}, board, p)


def test_retry_and_log(board, action_result):
    calls = []
    def fake(role, prompt, **kwargs):
        calls.append(prompt)
        return '{}' if len(calls) == 1 else json.dumps(action_result)
    output = board / '派发结果/动作识别/E01-P01.json'
    r = dispatch('动作识别', board, {'片段号': 'E01-P01'}, output, call=fake)
    assert r['通过'] and r['重答次数'] == 1
    assert '校验错误' in calls[1]
    log = json.loads((board / '派发日志.jsonl').read_text().splitlines()[-1])
    assert log['成功'] and log['重答次数'] == 1 and log['耗时秒'] >= 0
    assert log['输入文件'] and log['结果文件'] == str(output)
    assert log['时间'].endswith('+08:00')
    assert Path(log['失败返回'][0]).exists()


def test_retry_limit_preserves_old_result(board):
    output = board / 'result.json'
    save(output, {'原结果': True})
    calls = []
    def fake(*args, **kwargs):
        calls.append(1)
        return '{}'
    with pytest.raises(DispatchError, match='已重答 2 次'):
        dispatch('动作识别', board, {'片段号': 'E01-P01'}, output, call=fake)
    assert len(calls) == 3 and json.loads(output.read_text()) == {'原结果': True}
    assert not json.loads((board / '派发日志.jsonl').read_text().splitlines()[-1])['成功']


def test_resume_skips_completed_segment_and_changed_input_invalidates(board, action_result):
    calls = []
    def fake(*args, **kwargs):
        calls.append(1)
        return json.dumps(action_result)
    output = board / '派发结果/动作识别/E01-P01.json'
    first = dispatch('动作识别', board, {'片段号': 'E01-P01'}, output, call=fake)
    second = dispatch('动作识别', board, {'片段号': 'E01-P01'}, output, call=fake)
    assert not first['跳过'] and second['跳过'] and len(calls) == 1
    with (board / '素材/frames/K001_01.jpg').open('ab') as stream:
        stream.write(b'changed')
    third = dispatch('动作识别', board, {'片段号': 'E01-P01'}, output, call=fake)
    assert not third['跳过'] and len(calls) == 2


def test_runtime_error_is_not_retried(board):
    calls = []
    def fake(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('超时')
    with pytest.raises(RuntimeError, match='超时'):
        dispatch('动作识别', board, {'片段号': 'E01-P01'}, board / 'result.json', call=fake)
    assert len(calls) == 1
    assert not json.loads((board / '派发日志.jsonl').read_text().splitlines()[-1])['成功']


def test_pi_json_usage():
    raw = '\n'.join(json.dumps(e) for e in [
        {'type': 'message_end', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': '中途'}], 'usage': {'input': 10, 'output': 3}}},
        {'type': 'tool_execution_start', 'toolCallId': 'read1', 'toolName': 'read', 'args': {'path': '/tmp/frame.jpg'}},
        {'type': 'tool_execution_end', 'toolCallId': 'read1', 'isError': False},
        {'type': 'message_end', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': '{}'}], 'usage': {'input': 20, 'output': 4}}}])
    r = decode_response(raw)
    assert r.text == '{}' and r.usage == {'input': 30, 'output': 7} and r.reads == ['/tmp/frame.jpg']


def test_pi_options_tools_and_sessions(tmp_path):
    binary = tmp_path / 'pi-fake'
    binary.write_text(f'#!{sys.executable}\nimport json,sys,os\nprint(json.dumps({{"args":sys.argv[1:],"paths":json.loads(os.environ["S2S_DISPATCH_READ_PATHS"])}}))')
    binary.chmod(0o700)
    image = tmp_path / 'frame.jpg'
    image.write_bytes(b'image')
    with pi_options({'bin': str(binary)}, read_paths=[image]) as options:
        command = [options['bin'], '-p', '--no-session', '--', 'hello']
        r = json.loads(subprocess.check_output(command, text=True))
        assert r['args'][r['args'].index('--tools')+1] == 'read'
        assert '--no-extensions' in r['args'] and '--no-mcp' in r['args']
        assert r['paths'] == [str(image)]
    with pi_options({'bin': str(binary)}, controller=True, session_dir=tmp_path / '会话') as options:
        r = json.loads(subprocess.check_output([options['bin'], '-p', '--no-session', '--', 'hello'], text=True))
        assert '--no-session' not in r['args'] and '--session-dir' in r['args']
        assert 'read,bash,edit,write' in r['args']


def test_controller_checkpoint_and_prompt(tmp_path):
    root = tmp_path / '剧目'
    episode, _ = episode_prompt('续做剧', 'E01', root=root)
    for step in range(1, 6):
        mark_step('续做剧', 1, step, root=root)
    _, prompt = episode_prompt('续做剧', 1, root=root)
    assert '"5": true' in prompt and '"6": false' in prompt
    assert '状态.json' in prompt and '从状态中尚未完成' in prompt
    assert '{{' not in prompt
    calls = []
    def fake(role, prompt, **kwargs):
        calls.append((role, kwargs))
        return '主控结束'
    result = start_controller('单集主控', prompt, episode / '会话', call=fake)
    assert result['通过'] and calls[0][0] == '单集主控'
    assert (episode / '会话').exists()
    with pytest.raises(ValueError, match='复核要求'):
        episode_prompt('续做剧', 1, mode='定向复核', root=root)


def test_wrong_episode_or_unknown_mode(board):
    with pytest.raises(ValueError, match='片段号'):
        assemble('动作识别', board, {})
    with pytest.raises(ValueError, match='模式无效'):
        assemble('验收', board, {}, mode='建立')
    with pytest.raises(ValueError, match='必须填写'):
        assemble('验收', board, {}, mode='复核')


def test_failed_image_reads_not_counted():
    raw = '\n'.join(json.dumps(e) for e in [
        {'type': 'tool_execution_start', 'toolCallId': 'bad', 'toolName': 'read', 'args': {'path': '/not/allowed'}},
        {'type': 'tool_execution_end', 'toolCallId': 'bad', 'isError': True},
        {'type': 'message_end', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': '{}'}]}}])
    assert decode_response(raw).reads == []


def test_dense_frames_and_path_boundary(board, action_result):
    dense = board / '素材/frames_dense/00-00.0-00-00.5/帧.json'
    save(dense, {'视频': 'video.mp4', '起': '00:00.0', '止': '00:00.5', '帧率': 2,
                 '帧': [{'文件': 't00-00.2.jpg', '时间': '00:00.2'}]})
    (dense.parent / 't00-00.2.jpg').write_bytes(b'image')
    p = assemble('动作识别', board, {'片段号': 'E01-P01', '加密帧索引': [str(dense.relative_to(board))]}, mode='定向复看', scope='检查姿势')
    assert len(p.images) == 3 and p.dense[0]['镜头'] == 'E01-K001'
    with pytest.raises(ValueError, match='必须位于本集'):
        assemble('动作识别', board, {'片段号': 'E01-P01', '加密帧索引': ['../../secret.json']})
    shots = json.loads((board / '镜头表.json').read_text())
    shots['镜头'][0]['帧'][0]['文件'] = '../../../角色卡.json'
    save(board / '镜头表.json', shots)
    with pytest.raises(ValueError, match='路径越界'):
        assemble('动作识别', board, {'片段号': 'E01-P01'})


def test_read_guard_blocks_other_files_and_tools(board):
    import os
    import shutil
    node = shutil.which('node')
    if not node:
        pytest.skip('需要 pi 的 Node 运行环境')
    guard = Path(__file__).parents[1] / 'tools/s2s/dispatch/read_guard.ts'
    image = board / '素材/frames/K001_01.jpg'
    script = '''
      const module = await import(process.argv[1]);
      let handler;
      module.default({on: (event, fn) => { handler = fn; }});
      const ctx = {cwd: process.argv[2]};
      const good = await handler({toolName:'read',input:{path:'素材/frames/K001_01.jpg'}},ctx);
      const bad = await handler({toolName:'read',input:{path:'声音记录.json'}},ctx);
      const write = await handler({toolName:'write',input:{path:'工作稿.md'}},ctx);
      console.log(JSON.stringify({good:good || null,bad,write}));
    '''
    env = {**os.environ, 'S2S_DISPATCH_READ_PATHS': json.dumps([str(image.resolve())])}
    result = json.loads(subprocess.check_output([node, '--experimental-strip-types', '--input-type=module', '-e', script, guard.as_uri(), str(board)], env=env, text=True))
    assert result['good'] is None and result['bad']['block'] and result['write']['block']


def test_review_cross_episode_package(board):
    (board / '正式稿.md').write_text('第1集\n场景1-1：门口，日，外\n人物：女孩\n△ 女孩站着。', encoding='utf-8')
    save(board / '状态.json', {'结尾衔接': '留在门口'})
    p = assemble('验收', board, {'跨集集号': [1]}, mode='跨集')
    materials = p.prompt.split('## 本次授权资料')[1].split('## 返回包精确')[0]
    assert '跨集资料' in materials and '留在门口' in materials
    assert '对白隔离哨兵' not in materials


def test_other_role_validation(board, action_result):
    save(board / '派发结果/动作识别/E01-P01.json', action_result)
    role_packet = assemble('角色卡', board, {})
    roles = {'模式': '建立', '临时人物映射': [], '新增人物': [], '补充资料': [],
             '对白归属': [], '变更申请': [], '待核对': [], '性格总结': []}
    assert any('覆盖全部对白句' in e for e in validate_result('角色卡', roles, board, role_packet))
    draft_packet = assemble('成稿', board, {})
    draft = {'工作稿': (board / '工作稿.md').read_text(), '对照表': []}
    assert validate_result('成稿', draft, board, draft_packet) == []
    review_packet = assemble('验收', board, {})
    review = {'模式': '全文审查', '集': 1, '审查范围': {'行': '1-4', '场景': ['场景1-1'], '镜头': 'E01-K001', '对白句': 'E01-S001'},
              '问题': [], '复核': [], '统计': {'阻断': 0, '规范': 0}, '结论': '通过'}
    assert validate_result('验收', review, board, review_packet) == []
    review['统计']['阻断'] = 1
    assert any('问题统计不一致' in e for e in validate_result('验收', review, board, review_packet))


def test_token_usage_logged(board, action_result):
    def fake(*args, **kwargs):
        return json.dumps({'type': 'message_end', 'message': {'role': 'assistant',
            'content': [{'type': 'text', 'text': json.dumps(action_result)}],
            'usage': {'input': 100, 'output': 20, 'totalTokens': 120}}})
    dispatch('动作识别', board, {'片段号': 'E01-P01'}, board / 'result.json', call=fake)
    log = json.loads((board / '派发日志.jsonl').read_text().splitlines()[-1])
    assert log['token用量']['totalTokens'] == 120
