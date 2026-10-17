import copy
import json
import subprocess
import sys

import pytest

from s2s.dispatch.rolecard import apply_rolecard_result
from s2s.dispatch.roles import dispatch
from s2s.quality import review_version, check_script, check_review
from s2s.dispatch.roles import finish_review
from s2s.lint_script import lint_text
from s2s.dispatch.materials import assemble
from s2s.dispatch.results import validate_result
from s2s.validate import validate_file
from test_rolecard_two_stage import (episode, save, merge_result, attribution_result, board_bytes, confirmed_role)


def establish(ep, output):
    dispatch('角色卡', ep, {}, output, mode='建立', call=lambda role, prompt, **kw:
             json.dumps(merge_result() if '本次阶段：人物合并' in prompt else attribution_result()))
    return json.loads(output.read_text())


def approval(result, approve=True):
    return {'结果版本': 2, '输入版本': result['输入版本'],
            '批准': [{'人物': 'E01-N001', '依据': '主控审核代表帧，描述名可区分'}] if approve else []}


@pytest.mark.parametrize('approve', [False, True])
def test_apply_requires_explicit_approval_and_keeps_n_provenance(episode, tmp_path, approve):
    result = establish(episode, tmp_path / 'result.json')
    before = review_version(episode)
    applied = apply_rolecard_result(episode, result, approval(result, approve))
    assert applied['通过']
    roles = json.loads((episode.parent / '角色卡.json').read_text())
    table = json.loads((episode / '本集人物.json').read_text())
    assert table['人物'][0]['审核C'] == ('C001' if approve else None)
    assert len(roles) == int(approve)
    expected = 'C001' if approve else 'E01-N001'
    assert json.loads((episode / '临时人物.json').read_text())[0]['映射'] == expected
    assert json.loads((episode / '归属表.json').read_text())[0]['说话人'] == expected
    for kind in ('本集人物', '临时人物', '归属表', '待核对', '状态'):
        assert validate_file(episode / f'{kind}.json')['通过'], validate_file(episode / f'{kind}.json')
    assert review_version(episode) != before
    state = json.loads((episode / '状态.json').read_text())
    assert state['步骤']['3'] and not any(state['步骤'][str(n)] for n in range(4, 10))
    if not approve:
        assert any(r['位置'].get('句') == 'E01-S001' for r in json.loads((episode / '待核对.json').read_text()))


@pytest.mark.parametrize('bad', ['missing', 'old_approval', 'old_input', 'changed_confirmed_c', 'unregistered', 'tampered'])
def test_bad_apply_leaves_board_unchanged(episode, tmp_path, bad):
    result = establish(episode, tmp_path / 'result.json')
    decisions = approval(result)
    if bad == 'missing':
        decisions = None
    elif bad == 'old_approval':
        decisions['输入版本'] = {}
    elif bad == 'old_input':
        save(episode / '声音记录.json', [])
    elif bad == 'changed_confirmed_c':
        save(episode.parent / '角色卡.json', [confirmed_role('C099')])
    elif bad == 'unregistered':
        decisions['批准'][0]['人物'] = 'E02-N001'
    else:
        result['对白归属'][0]['说话人'] = 'C999'
    before = board_bytes(episode)
    with pytest.raises((ValueError, TypeError)):
        apply_rolecard_result(episode, result, decisions)
    assert board_bytes(episode) == before


@pytest.mark.parametrize('target', ['归属表', '本集人物', '待核对', '状态'])
def test_apply_rejects_edited_overwrite_target_without_losing_correction(episode, tmp_path, target):
    result = establish(episode, tmp_path / 'result.json')
    path = episode / f'{target}.json'
    if target == '归属表':
        corrected = attribution_result(None, True)['对白归属']
        corrected[0]['依据'] = '建立后主控回听修正'
        save(path, corrected)
    else:
        # 即使仅格式改变，覆盖目标的size/mtime基线也已不一致。
        path.write_bytes(path.read_bytes() + b'\n ')
    before = board_bytes(episode)
    with pytest.raises(ValueError, match='覆盖目标版本过期'):
        apply_rolecard_result(episode, result, approval(result))
    assert board_bytes(episode) == before


def test_apply_rechecks_overwrite_targets_after_staging_validation(episode, tmp_path, monkeypatch):
    from s2s.dispatch import rolecard
    result = establish(episode, tmp_path / 'result.json')
    original_validate = rolecard.validate_file
    corrected = attribution_result(None, True)['对白归属']
    expected = {}

    def validate_and_correct(path, kind):
        checked = original_validate(path, kind)
        if kind == '状态':
            save(episode / '归属表.json', corrected)
            expected.update(board_bytes(episode))
        return checked

    monkeypatch.setattr(rolecard, 'validate_file', validate_and_correct)
    with pytest.raises(ValueError, match='覆盖目标版本过期'):
        apply_rolecard_result(episode, result, approval(result))
    assert expected and board_bytes(episode) == expected


def test_apply_requires_overwrite_target_snapshot(episode, tmp_path):
    result = establish(episode, tmp_path / 'result.json')
    result.pop('覆盖目标版本')
    before = board_bytes(episode)
    with pytest.raises(ValueError, match='结果字段不完整'):
        apply_rolecard_result(episode, result, approval(result))
    assert board_bytes(episode) == before


def test_existing_cli_can_apply_without_model(episode, tmp_path):
    output = tmp_path / 'result.json'
    result = establish(episode, output)
    approvals = tmp_path / 'approval.json'
    save(approvals, approval(result))
    run = subprocess.run([sys.executable, 'tools/dispatch_role.py', '角色卡', '--集', str(episode),
                          '--应用结果', str(output), '--审核', str(approvals)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)['获批映射'] == {'E01-N001': 'C001'}


def test_old_no_n_board_and_missing_n_registry(episode):
    (episode / '本集人物.json').unlink()
    assert validate_file(episode / '临时人物.json')['通过']
    temporary = json.loads((episode / '临时人物.json').read_text())
    temporary[0]['映射'] = 'E01-N001'
    save(episode / '临时人物.json', temporary)
    assert not validate_file(episode / '临时人物.json')['通过']


def draft_for(ep, speaker, *, name='穿紫衣的双马尾女孩'):
    text = f'第1集\n场景1-1：门口，日，外\n人物：{name if not speaker.startswith("【") else ""}\n{speaker}：声音原文哨兵\n'
    (ep / '工作稿.md').write_text(text, encoding='utf-8')
    save(ep / '对照表.json', [
        {'行': 2, '类型': '场景行', '场景': '场景1-1', '句': [], '镜头': ['E01-K001'], '动作': []},
        {'行': 3, '类型': '人物行', '场景': '场景1-1', '句': [], '镜头': ['E01-K001'], '动作': []},
        {'行': 4, '类型': '对白', '场景': '场景1-1', '句': ['E01-S001'], '镜头': ['E01-K001', 'E01-K002'], '动作': []}])


def test_pending_n_draft_and_formal_identity_gate(episode, tmp_path):
    result = establish(episode, tmp_path / 'result.json')
    apply_rolecard_result(episode, result, approval(result, False))
    pending = json.loads((episode / '待核对.json').read_text())
    question = next(r['id'] for r in pending if r['位置'].get('句') == 'E01-S001')
    draft_for(episode, f'【说话人待核对 {question}】')
    checked = check_script(episode)
    assert checked['通过'], checked['错误']
    assert not check_script(episode, formal=True)['通过']
    assert lint_text('第1集\nE01-N001：声音原文哨兵', formal=True)['错误']


def test_downstream_resolves_t_n_approved_c_and_registry_change_expires_review(episode, tmp_path):
    result = establish(episode, tmp_path / 'result.json')
    apply_rolecard_result(episode, result, approval(result))
    draft_for(episode, '穿紫衣的双马尾女孩')
    before = review_version(episode)
    temporary = json.loads((episode / '临时人物.json').read_text())
    temporary[0]['映射'] = 'E01-N001'
    save(episode / '临时人物.json', temporary)
    attribution = json.loads((episode / '归属表.json').read_text())
    for speaker in ('E01-N001', 'E01-P01-T1'):
        attribution[0]['说话人'] = speaker
        save(episode / '归属表.json', attribution)
        checked = check_script(episode)
        assert checked['通过'], checked['错误']
    assert review_version(episode) != before
    table = json.loads((episode / '本集人物.json').read_text())
    table['人物'][0]['审核依据'] += '，补充审核'
    save(episode / '本集人物.json', table)
    assert review_version(episode) != before
    table['人物'][0]['审核C'] = 'E01-N001'
    save(episode / '本集人物.json', table)
    assert not validate_file(episode / '本集人物.json')['通过']


def test_registry_rejects_stale_source_and_tampered_frozen_person(episode, tmp_path):
    result = establish(episode, tmp_path / 'result.json')
    apply_rolecard_result(episode, result, approval(result, False))
    table = json.loads((episode / '本集人物.json').read_text())
    save(episode.parent / '角色卡.json', [confirmed_role('C099')])
    assert validate_file(episode / '本集人物.json')['通过']  # 后续集追加无关C不改历史冻结身份。
    save(episode.parent / '角色卡.json', [])
    table['人物'][0]['名称'] = '穿白衣的长发女孩'
    save(episode / '本集人物.json', table)
    assert not validate_file(episode / '本集人物.json')['通过']
    save(episode / '本集人物.json', result['本集人物'])
    (episode / '素材/frames/K001_01.jpg').write_bytes(b'updated')
    assert not validate_file(episode / '归属表.json')['通过']


def test_targeted_and_summary_bind_current_registry_without_regrouping(episode, tmp_path):
    result = establish(episode, tmp_path / 'result.json')
    apply_rolecard_result(episode, result, approval(result, False))
    targeted = assemble('角色卡', episode, {}, mode='定向核对', scope='E01-S001')
    assert 'E01-N001' in targeted.prompt and '台词字幕隔离哨兵' not in targeted.prompt
    summary = assemble('角色卡', episode, {}, mode='总结')
    assert not summary.images
    assert all(s not in summary.prompt for s in ('声音原文哨兵', '台词字幕隔离哨兵', '动作叙述隔离哨兵'))
    data = {'模式': '定向核对', '临时人物映射': [], '新增人物': [], '补充资料': [],
            '对白归属': attribution_result()['对白归属'], '变更申请': [], '待核对': [], '性格总结': []}
    assert not validate_result('角色卡', data, episode, targeted)
    data['临时人物映射'] = [{'临时': 'E01-P01-T1', '映射': 'C001', '依据': '重分组', '确定度': '确认'}]
    assert validate_result('角色卡', data, episode, targeted)


@pytest.mark.parametrize('position', [{'镜头': 'E01-K001'}, {'句': 'E01-S001'},
                                     {'句': 'E01-S001', '镜头': 'E01-K001'}])
def test_targeted_material_question_without_attribution_change(episode, tmp_path, position):
    result = establish(episode, tmp_path / 'result.json')
    apply_rolecard_result(episode, result, approval(result, False))
    targeted = assemble('角色卡', episode, {}, mode='定向核对', scope='E01-K001')
    data = {'模式': '定向核对', '临时人物映射': [], '新增人物': [], '补充资料': [],
            '对白归属': [], '变更申请': [], '待核对': [
                {'位置': position, '问题': '衣着资料与代表帧冲突，需主控核对', '级别': '阻断'}], '性格总结': []}
    assert not validate_result('角色卡', data, episode, targeted)


@pytest.mark.parametrize('position', [{}, {'场景': '场景1-1'}, {'镜头': 'E02-K001'},
                                     {'镜头': 'E01-K999'}, {'句': 'E02-S001'},
                                     {'句': 'E01-S999'}, {'镜头': None}, {'句': []},
                                     {'镜头': 'E01-K001', '未知': 'E01-S001'}])
def test_targeted_material_question_rejects_bad_positions(episode, tmp_path, position):
    result = establish(episode, tmp_path / 'result.json')
    apply_rolecard_result(episode, result, approval(result, False))
    targeted = assemble('角色卡', episode, {}, mode='定向核对', scope='E01-K001')
    data = {'模式': '定向核对', '临时人物映射': [], '新增人物': [], '补充资料': [],
            '对白归属': [], '变更申请': [], '待核对': [
                {'位置': position, '问题': '资料冲突', '级别': '阻断'}], '性格总结': []}
    assert validate_result('角色卡', data, episode, targeted)


@pytest.mark.parametrize('blocker', ['same_sentence', 'shot_only', 'other_sentence', 'non_blocking'])
def test_targeted_uncertain_attribution_still_requires_same_sentence_blocker(episode, tmp_path, blocker):
    sounds = json.loads((episode / '声音记录.json').read_text())
    sounds.append({**sounds[0], 'id': 'E01-S002'})
    save(episode / '声音记录.json', sounds)
    targeted = assemble('角色卡', episode, {}, mode='定向核对', scope='E01-S001')
    position = {'句': 'E01-S001'} if blocker in ('same_sentence', 'non_blocking') else (
        {'镜头': 'E01-K001'} if blocker == 'shot_only' else {'句': 'E01-S002'})
    data = {'模式': '定向核对', '临时人物映射': [], '新增人物': [], '补充资料': [],
            '对白归属': attribution_result(None, True)['对白归属'], '变更申请': [], '待核对': [
                {'位置': position, '问题': '身份需核对', '级别': '规范' if blocker == 'non_blocking' else '阻断'}],
            '性格总结': []}
    errors = validate_result('角色卡', data, episode, targeted)
    assert bool(errors) == (blocker != 'same_sentence')


def test_approval_is_bound_to_frozen_merge_not_just_source_files(episode, tmp_path):
    result = establish(episode, tmp_path / 'result.json')
    decisions = approval(result)
    changed = copy.deepcopy(result)
    changed['合并结果']['分组'][0]['依据'] = '不同合并结论'
    changed['输入版本']['冻结合并'] = changed['合并结果']
    changed['本集人物']['输入版本'] = changed['输入版本']
    changed['本集人物']['人物'][0]['依据'] = '不同合并结论'
    changed['临时人物映射'][0]['依据'] = '不同合并结论'
    before = board_bytes(episode)
    with pytest.raises(ValueError, match='审核输入版本过期'):
        apply_rolecard_result(episode, changed, decisions)
    assert board_bytes(episode) == before


def test_current_approval_change_expires_an_independent_review_receipt(episode, tmp_path):
    video = tmp_path / 'mock.mp4'
    video.write_bytes(b'mock-video')
    shots = json.loads((episode / '镜头表.json').read_text())
    shots['视频'] = str(video)
    save(episode / '镜头表.json', shots)
    result = establish(episode, tmp_path / 'result.json')
    apply_rolecard_result(episode, result, approval(result))
    draft_for(episode, '穿紫衣的双马尾女孩')
    paths = [str(p.resolve()) for p in sorted((episode / '素材/frames').glob('*.jpg'))]
    coverage = [{'片段': f'E01-P{n:02d}', '状态': '完成', '授权帧数': 2, '已读帧数': 2,
                 '已读帧': paths, '结果文件': 'mock-review'} for n in (1, 2)]
    review_path = episode / '派发结果/验收.json'
    review_path.parent.mkdir(exist_ok=True)
    # 离线构造回执，只测试版本门禁，不代表真实审查。
    finish_review(episode, review_path, [{'审查范围': {'场景': ['场景1-1']}, '问题': [], '结论': '通过'}],
                  coverage, version=review_version(episode))
    assert check_review(episode)
    assert check_script(episode, formal=True)['通过']
    table = json.loads((episode / '本集人物.json').read_text())
    table['人物'][0]['审核依据'] += '，新审核'
    save(episode / '本集人物.json', table)
    assert not check_review(episode)
    assert not check_script(episode, formal=True)['通过']


def test_apply_preserves_series_role_references_to_other_episodes(episode, tmp_path):
    role = confirmed_role()
    role['首次出场'] = 'E02-K001'
    save(episode.parent / '角色卡.json', [role])
    save(episode.parent / 'E02/镜头表.json', {'镜头': [{'id': 'E02-K001'}]})
    result = establish(episode, tmp_path / 'result.json')
    decisions = approval(result)
    decisions['批准'][0]['已有角色'] = 'C007'
    assert apply_rolecard_result(episode, result, decisions)['获批映射'] == {'E01-N001': 'C007'}
    assert json.loads((episode.parent / '角色卡.json').read_text())[0]['首次出场'] == 'E02-K001'
