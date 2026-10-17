"""生产剧本规则的离线检查，不调用模型。"""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from s2s.dispatch.materials import Package
from s2s.dispatch.results import validate_result
from s2s.dispatch.roles import dispatch, DispatchError, finish_review
from s2s.lint_script import lint_text
from s2s.quality import check_script, review_version
from s2s.validate import read_json, validate_file

FIXTURE = Path(__file__).parent / 'fixtures/t5/完整黑板'


@pytest.fixture
def board(tmp_path):
    root = tmp_path / '剧目/测试剧'
    shutil.copytree(FIXTURE, root)
    episode = root / 'E01'
    actions = read_json(episode / '动作记录.json')
    for action, sentence in zip(actions, (None, 'E01-S001', None, 'E01-S003')):
        action['排布'] = {'位置': '对白括号' if sentence else '独立△', '句': sentence,
                          '依据': '按人物、动作过程和实际对白时序审核'}
    save(episode / '动作记录.json', actions)
    return episode


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def change(board, name, edit):
    path = board / (name + '.json')
    value = read_json(path)
    edit(value)
    save(path, value)
    return path


def rewrite(board, old, new):
    path = board / '工作稿.md'
    path.write_text(path.read_text(encoding='utf-8').replace(old, new), encoding='utf-8')


def rules(board, **kwargs):
    return {error['检查项'] for error in check_script(board, **kwargs)['错误']}


def issue(*, position=None):
    return {'id': 'E01-Q001', '来源': '归属', '位置': position or {'句': 'E01-S001'},
            '原文': '存疑', '类型': '声音与画面归属', '级别': '阻断', '相关人物': [],
            '依据': '证据不足', '修正要求': '回听核对', '处理职责': '角色卡',
            '状态': '待核对', '复核结果': None}


def accept_review(board, *, version=None, simulated=False):
    """离线构造审查回执，只测试正式稿规则。"""
    result = {'审查范围': {'场景': ['场景1-1'], '对白句': 'E01-S001 至 E01-S003'},
              '问题': [], '结论': '通过'}
    coverage = [{'片段': row['id'], '状态': '完成', '授权帧数': 1, '已读帧数': 1,
                 '已读帧': ['测试帧'], '结果文件': '测试结果'}
                for row in read_json(board / '片段表.json')['片段']]
    output = board / '派发结果/验收.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    finish_review(board, output, [result], coverage, version=version or review_version(board), simulated=simulated)
    return output


def test_confirmed_fixture_passes_and_reports_semantic_review_boundary(board):
    assert check_script(board)['通过'], check_script(board)
    accept_review(board)
    assert check_script(board, formal=True)['通过'], check_script(board, formal=True)
    assert check_script(board)['需素材验收']


@pytest.mark.parametrize('name', ['男人', '女人', '小男孩', '她', '对方', '', ' 林舟'])
def test_role_card_cannot_register_illegal_name(board, name):
    path = change(board.parent, '角色卡', lambda rows: rows[0].update({'名称': name}))
    assert not validate_file(path)['通过']
    assert not lint_text(f'人物：{name}\n△ {name}站起身。', read_json(path))['通过']


def test_duplicate_names_and_ids_are_rejected(board):
    path = change(board.parent, '角色卡', lambda rows: rows[1].update({'名称': rows[0]['名称']}))
    assert not validate_file(path)['通过']
    assert '角色卡名称' in rules(board)
    change(board.parent, '角色卡', lambda rows: rows[1].update({'名称': '沈宁', 'id': 'C001'}))
    assert not validate_file(path)['通过']


@pytest.mark.parametrize('line', ['△ 林总站起身。', '林舟（看向沈小姐）：台词。'])
def test_registered_aliases_are_not_display_names(board, line):
    change(board.parent, '角色卡', lambda rows: (rows[0]['别名称呼'].append('林总'), rows[1]['别名称呼'].append('沈小姐')))
    roles = read_json(board.parent / '角色卡.json')
    assert any(e['检查项'] == '角色别称漂移' for e in lint_text(line, roles)['错误'])
    assert lint_text('林舟：林总和沈小姐来了，她站起身。', roles)['通过']


@pytest.mark.parametrize('text', ['眼中闪过一丝怒意', '下意识地举手', '镜头推近', '画面切向门口'])
def test_forbidden_style_only_in_description(text):
    roles = [{'名称': '林舟'}]
    assert not lint_text('△ 林舟' + text + '。', roles)['通过']
    assert not lint_text('林舟（' + text + '）：台词。', roles)['通过']
    assert lint_text('林舟：' + text + '，她和男人来了。', roles)['通过']
    assert lint_text('△ 画面字幕：' + text, roles)['通过']


def test_legal_description_intransitive_action_and_static_scene_are_allowed():
    roles = [{'名称': '穿红衣服的小男孩'}, {'名称': '沈宁'}]
    assert lint_text('△ 穿红衣服的小男孩站起身。\n△ 杯子掉在地上。\n沈宁：她来了。', roles)['通过']


@pytest.mark.parametrize('field,value', [('动作', ' '), ('依据帧', []), ('依据帧', ['K002_01.jpg']),
                                       ('起', '00:06.0'), ('止', '00:06.0')])
def test_action_requires_content_time_and_own_shot_evidence(board, field, value):
    path = change(board, '动作记录', lambda rows: rows[0].update({field: value}))
    assert not validate_file(path)['通过']
    assert not check_script(board)['通过']


def test_dense_evidence_is_accepted_only_in_matching_shot_and_video(board):
    index = board / '素材/frames_dense/window/帧.json'
    dense = {'视频': 'videos/sample/E01.mp4', '起': '00:01.0', '止': '00:02.0', '帧率': 2,
             '帧': [{'文件': 't00-01.2.jpg', '时间': '00:01.2'}]}
    save(index, dense)
    path = change(board, '动作记录', lambda rows: rows[0].update({'依据帧': ['t00-01.2.jpg']}))
    assert validate_file(path)['通过'], validate_file(path)
    assert check_script(board)['通过'], check_script(board)
    dense['视频'] = 'other.mp4'
    save(index, dense)
    assert not validate_file(path)['通过']


def test_duplicate_action_reference_is_blocked(board):
    change(board, '对照表', lambda rows: rows[3]['动作'].append('E01-A0001'))
    assert '重复动作' in rules(board)


def test_changed_speaker_and_changed_shape_are_blocked(board):
    rewrite(board, '林舟（画外）', '沈宁（画外）')
    assert '说话人错归属' in rules(board)
    rewrite(board, '沈宁（画外）', '林舟（心声）')
    assert '声音形态不一致' in rules(board)


def test_listener_action_in_speaker_parenthesis_is_blocked(board):
    change(board, '动作记录', lambda rows: rows[1].update({'执行者': 'C002'}))
    assert '括号动作错归属' in rules(board)


def test_action_after_dialogue_cannot_be_moved_into_parenthesis(board):
    change(board, '动作记录', lambda rows: rows[1].update({'起': '00:09.0', '止': '00:09.5'}))
    assert '动作对白时序' in rules(board)


def test_wrong_action_subject_and_missing_target_are_blocked(board):
    rewrite(board, '△ 林舟走到', '△ 沈宁走到')
    assert '动作执行者不一致' in rules(board)
    rewrite(board, '从门口走到沈宁身旁，把信封递给沈宁。', '从门口走到桌旁，递出信封。')
    assert '动作对象未呈现' in rules(board)


def test_action_in_empty_parenthesis_or_without_record_is_blocked(board):
    rewrite(board, '林舟（右掌拍在桌面上，抬高声音）', '林舟')
    assert '动作未呈现' in rules(board)
    rewrite(board, '林舟（画外）', '林舟（画外，抬头）')
    assert {'画外形态动作', '括号动作缺少记录'} <= rules(board)


def test_events_cannot_be_moved_before_earlier_events(board):
    change(board, '动作记录', lambda rows: rows[2].update({'镜头': 'E01-K001', '起': '00:00.1', '止': '00:00.5',
                                                        '依据帧': ['K001_01.jpg']}))
    change(board, '对照表', lambda rows: rows[6].update({'镜头': ['E01-K001']}))
    assert '事件时序错误' in rules(board)


def test_key_action_cannot_disappear_by_removing_mapping(board):
    change(board, '对照表', lambda rows: rows[2]['动作'].clear())
    assert '关键动作未覆盖' in rules(board)


def test_uncertain_key_action_stays_in_issue_list_not_confirmed_script(board):
    change(board, '动作记录', lambda rows: rows[0].update({'确定度': '存疑'}))
    change(board, '对照表', lambda rows: rows[2]['动作'].clear())
    rewrite(board, '△ 林舟走到会议桌旁，拉开椅子，把文件放到桌上。', '△ 画面字幕：办公室')
    assert '关键动作未确认' in rules(board)
    save(board / '待核对.json', [issue(position={'镜头': 'E01-K001'})])
    assert check_script(board)['通过'], check_script(board)
    assert {'关键动作未确认', '未决问题'} <= rules(board, formal=True)


def test_draft_can_preserve_unknown_speaker_but_cannot_guess(board):
    change(board, '归属表', lambda rows: rows[1].update({'确定度': '存疑'}))
    save(board / '待核对.json', [issue(position={'句': 'E01-S002'})])
    assert '存疑归属冒充确认' in rules(board)
    rewrite(board, '林舟（画外）', '【说话人待核对 E01-Q001】')
    assert check_script(board)['通过'], check_script(board)
    assert not check_script(board, formal=True)['通过']


def test_note_does_not_modify_dialogue_and_must_reference_pending_issue(board):
    save(board / '待核对.json', [issue()])
    rewrite(board, '今天必须说清楚。', '今天必须说清楚。【待核对 E01-Q001】')
    assert check_script(board)['通过'], check_script(board)
    save(board / '待核对.json', [])
    assert '待核对引用' in rules(board)


def test_fixed_issue_without_recheck_still_blocks_formal(board):
    row = issue()
    row['状态'] = '已修正'
    save(board / '问题清单.json', [row])
    assert '未决问题' in rules(board, formal=True)
    row['复核结果'] = '已对照素材复核，通过'
    save(board / '问题清单.json', [row])
    accept_review(board)
    assert check_script(board, formal=True)['通过'], check_script(board, formal=True)


def test_formal_rejects_unconfirmed_identity(board):
    change(board.parent, '角色卡', lambda rows: rows[0].update({'状态': '待确认'}))
    assert not check_script(board, formal=True)['通过']


def test_draft_result_runs_all_checks_before_accepting(board):
    packet = Package('', mode='全集成稿')
    draft = {'工作稿': (board / '工作稿.md').read_text(), '对照表': read_json(board / '对照表.json')}
    assert validate_result('成稿', draft, board, packet) == []
    broken = copy.deepcopy(draft)
    broken['工作稿'] = broken['工作稿'].replace('林舟（画外）', '沈宁（画外）')
    assert any('说话人错归属' in e for e in validate_result('成稿', broken, board, packet))
    assert (board / '工作稿.md').read_text() == draft['工作稿']


def test_invalid_draft_retry_preserves_previous_output(board):
    draft = {'工作稿': (board / '工作稿.md').read_text().replace('林舟（画外）', '沈宁（画外）'),
             '对照表': read_json(board / '对照表.json')}
    output = board / 'draft.json'
    save(output, {'旧结果': True})
    calls = []
    def fake(*args, **kwargs):
        calls.append(1)
        return json.dumps(draft, ensure_ascii=False)
    with pytest.raises(DispatchError, match='说话人错归属'):
        dispatch('成稿', board, {}, output, call=fake)
    assert len(calls) == 3
    assert read_json(output) == {'旧结果': True}


def test_new_role_name_and_mapping_are_checked(board):
    packet = Package('', mode='建立')
    data = {'模式': '建立', '临时人物映射': [], '新增人物': [], '补充资料': [],
            '对白归属': read_json(board / '归属表.json'), '变更申请': [], '待核对': [], '性格总结': []}
    assert validate_result('角色卡', data, board, packet) == []
    data['临时人物映射'] = [{'临时': 'E01-P01-T1', '映射': 'C999', '依据': '外观连续', '确定度': '确认'}]
    assert any('未登记编号' in e for e in validate_result('角色卡', data, board, packet))
    data['临时人物映射'][0]['映射'] = 'C001'
    assert validate_result('角色卡', data, board, packet) == []
    data['临时人物映射'].append(dict(data['临时人物映射'][0]))
    assert any('重复映射' in e for e in validate_result('角色卡', data, board, packet))


def test_quality_cli_returns_json(board):
    command = [sys.executable, 'tools/check_script.py', str(board)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['通过']
    rewrite(board, '林舟（画外）', '沈宁（画外）')
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert not json.loads(result.stdout)['通过']


def test_formal_requires_current_successful_review(board):
    assert '独立验收缺失或过期' in rules(board, formal=True)
    accept_review(board)
    assert check_script(board, formal=True)['通过']
    rewrite(board, '今天必须说清楚。', '今天必须说清楚！')
    assert '独立验收缺失或过期' in rules(board, formal=True)


@pytest.mark.parametrize('name,edit', [
    ('动作记录', lambda rows: rows[0].update({'说明': '新增证据说明'})),
    ('归属表', lambda rows: rows[0].update({'依据': '新的归属依据'})),
    ('声音记录', lambda rows: rows[0].update({'声线': 'E01-V2'})),
])
def test_changed_evidence_invalidates_old_review(board, name, edit):
    accept_review(board)
    change(board, name, edit)
    assert '独立验收缺失或过期' in rules(board, formal=True)


def test_changed_frames_invalidate_old_review(board):
    path = board / '素材/frames/K001_01.jpg'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'frame')
    accept_review(board)
    path.write_bytes(b'changed frame')
    assert '独立验收缺失或过期' in rules(board, formal=True)


def test_role_summary_does_not_invalidate_identity_but_rename_does(board):
    accept_review(board)
    change(board.parent, '角色卡', lambda rows: rows[0]['性格总结'].append({'集': 1, '内容': '坚决', '事件依据': '拍桌'}))
    assert check_script(board, formal=True)['通过']
    change(board.parent, '角色卡', lambda rows: rows[0]['别名称呼'].append('林总'))
    assert '独立验收缺失或过期' in rules(board, formal=True)


def test_changes_during_review_prevent_pass(board):
    version = review_version(board)
    rewrite(board, '今天必须说清楚。', '今天必须说清楚！')
    path = accept_review(board, version=version)
    assert read_json(path)['结论'] == '不通过'
    assert '独立验收缺失或过期' in rules(board, formal=True)


def test_simulated_review_cannot_authorize_formal_delivery(board):
    accept_review(board, simulated=True)
    assert '独立验收缺失或过期' in rules(board, formal=True)


@pytest.mark.parametrize('mode', ['failed', 'missing_segment', 'missing_reads', 'altered_result'])
def test_incomplete_or_failed_review_never_authorizes_formal(board, mode):
    path = accept_review(board)
    receipt_path = path.with_suffix('.coverage.json')
    receipt = read_json(receipt_path)
    if mode == 'failed':
        result = read_json(path)
        result['结论'] = '不通过'
        save(path, result)
        receipt['验收结果'] = result
    elif mode == 'missing_segment':
        receipt['片段'].pop()
    elif mode == 'missing_reads':
        receipt['片段'][0]['已读帧数'] = 0
    else:
        result = read_json(path)
        result['审查范围']['行'] = '1-1'
        save(path, result)
    save(receipt_path, receipt)
    assert '独立验收缺失或过期' in rules(board, formal=True)


def test_formal_must_equal_reviewed_working_copy(board):
    accept_review(board)
    formal = board / '正式稿.md'
    formal.write_text((board / '工作稿.md').read_text().replace('今天必须说清楚。', '今天必须说清楚！'))
    assert '交付版本不一致' in rules(board, script=formal, formal=True)
