"""E0证据降级、形态解耦与发布恢复边界，不使用真实素材。"""

import copy
from pathlib import Path

import pytest

from s2s.board import initialize_board
from s2s.dispatch.rolecard import _declarations, _validate_attribution
from s2s.fixed_pipeline import run_fixed_episode, run_lock
from s2s.fixed_steps import finish_episode, merge_actions
from s2s.quality import check_script
from s2s.run_record import RunRecord, load, save
from test_fixed_episode import FIXTURES, VIDEO
from test_production_pipeline import accepted, board, workspace, mock_phases


def custom_run(tmp_path, data):
    fixture = tmp_path / 'case.json'; save(fixture, data)
    return run_fixed_episode('测试', 1, video=VIDEO, fixture=fixture, run_root=tmp_path / 'run')


def test_identity_is_retained_when_sentence_shape_changes(tmp_path):
    data = load(FIXTURES / 'pass.json')
    attribution = data['模型返回']['对白归属/E01-P01']
    attribution['对白归属'][0].update(形态=None, 形态待核对='镜头切到听者，身份保持C001，缺可靠词界；核对整句画内/画外变化')
    attribution['待核对'] = [{'位置': {'句': 'E01-S001'}, '问题': '同句形态变化，身份证据保留', '级别': '阻断'}]
    data['模型返回']['动作识别/E01-P01']['镜头'][0]['动作'] = []
    draft = data['模型返回']['成稿']
    draft['动作排布'] = []
    draft['对照表'][-1]['动作'] = []
    draft['工作稿'] = '第1集\n场景1-1：门口，日，外\n人物：林舟\n林舟：等一下。【待核对 E01-Q001】\n'
    result = custom_run(tmp_path, data)
    episode = Path(result['目录'])
    row = load(episode / '归属表.json')[0]
    assert row['说话人'] == 'C001' and row['确定度'] == '确认' and row['形态'] is None
    assert check_script(episode)['通过']
    assert '声音形态未确认' in {e['检查项'] for e in check_script(episode, formal=True)['错误']}
    sounds = load(episode / '声音记录.json')
    assert '词' not in sounds[0] and sounds[0]['文本'] == '等一下。'


def test_shape_pending_requires_same_sentence_blocker_and_preserves_identity():
    row = {'句': 'E01-S001', '说话人': 'C001', '形态': None, '形态待核对': '画内转画外', '依据': '身份连续', '确定度': '确认'}
    data = {'对白归属': [row], '待核对': [], '补充建议': []}
    with pytest.raises(ValueError, match='形态未定须有同句阻断'):
        _validate_attribution(data, [{'id': 'E01-S001'}], {'人物': []}, {'已确认角色': [{'id': 'C001'}]}, {'E01-K001'})


def test_mixed_speaker_sentence_cannot_be_confirmed_for_one_person():
    data = {'对白归属': [{'句': 'E01-S001', '说话人': 'C001', '形态': '画内', '依据': '仅候选证据', '确定度': '确认'}],
            '补充建议': [], '待核对': [{'位置': {'句': 'E01-S001'}, '问题': '句内换人，须回听拆句', '级别': '阻断'}]}
    batch = [{'id': 'E01-S001', '核对': {'异常': ['句内换人或重叠']}}]
    with pytest.raises(ValueError, match='不能整句确认一个人'):
        _validate_attribution(data, batch, {'人物': []}, {'已确认角色': [{'id': 'C001'}]}, {'E01-K001'})
    data['对白归属'][0]['确定度'] = '存疑'
    _validate_attribution(data, batch, {'人物': []}, {'已确认角色': [{'id': 'C001'}]}, {'E01-K001'})


def test_missing_lipsync_is_reported_without_fake_negative(tmp_path):
    data = load(FIXTURES / 'pass.json')
    result = custom_run(tmp_path, data)
    episode = Path(result['目录'])
    assert not (episode / '口型记录.json').exists()
    stage = load(episode / '派发结果/角色卡/建立-阶段/对白归属-E01-P01.json')
    assert '缺失；不能作为否定证据' in stage['输入版本']['资料']['提示词']
    assert load(episode / '候选证据状态.json')['口型'] == '缺失；不能作为否定证据'


def test_optional_preprocessor_failure_keeps_lipsync_absent(workspace, monkeypatch):
    from s2s import production as p
    episode, video = board(workspace)
    mock_phases(monkeypatch)
    phases = list(p.PHASES)
    def unavailable(*args):
        raise RuntimeError('缺少本地口型模型')
    phases[4] = ('口型', unavailable, phases[4][2], phases[4][3])
    monkeypatch.setattr(p, 'PHASES', tuple(phases))
    job = p.prepare_episode(video, episode, optional_lipsync=True)
    assert job['阶段']['口型']['状态'] == '降级'
    assert job['阶段']['口型']['证据状态'] == '缺失；不能作为否定证据'
    assert not (episode / '口型记录.json').exists()
    assert job['状态'] == '素材就绪'


def test_missing_combined_placement_retries_before_draft_publication(tmp_path):
    data = load(FIXTURES / 'pass.json')
    valid = data['模型返回']['成稿']
    invalid = copy.deepcopy(valid); invalid.pop('动作排布')
    data['模型返回']['成稿'] = [invalid, valid]
    result = custom_run(tmp_path, data)
    ledger = load(result['调用账本'])
    attempts = [r for r in ledger['请求'] if r['阶段'].startswith('成稿/')]
    assert len(attempts) == 2 and attempts[-1]['阶段'].endswith('格式尝试2')
    assert result['离线影子终态'] == '正式交付候选'


def test_unapproved_critical_action_retains_event_evidence_and_blocker(tmp_path):
    data = load(FIXTURES / 'pending.json')
    action = load(FIXTURES / 'pass.json')['模型返回']['动作识别/E01-P01']['镜头'][0]['动作'][0]
    data['模型返回']['动作识别/E01-P01']['镜头'][0]['动作'] = [action]
    result = custom_run(tmp_path, data)
    episode = Path(result['目录'])
    actions = load(episode / '动作记录.json')
    assert len(actions) == 1 and actions[0]['动作'] == '举起右手' and actions[0]['确定度'] == '确认'
    assert len(actions[0]['依据帧']) == 2
    assert any('关键动作事实保留' in q['依据'] for q in load(episode / '待核对.json'))
    assert check_script(episode)['通过']
    assert not check_script(episode, formal=True)['通过']


def test_transaction_recovery_refuses_unrelated_edit(tmp_path, monkeypatch):
    initialize_board('测试', 1, root=tmp_path)
    episode = tmp_path / '测试/E01'
    record = RunRecord(episode, {'版本': 1})
    original = RunRecord._apply_transaction
    def interrupt(transaction):
        original({'文件': transaction['文件'][:1]})
        raise RuntimeError('模拟崩溃')
    monkeypatch.setattr(RunRecord, '_apply_transaction', staticmethod(interrupt))
    with pytest.raises(RuntimeError, match='崩溃'):
        record.stage('多文件', lambda: record.publish([(episode / '工作稿.md', 'A'), (episode / '验收报告.md', 'B')], {'通过': True}))
    (episode / '工作稿.md').write_text('人工后续编辑')
    monkeypatch.setattr(RunRecord, '_apply_transaction', staticmethod(original))
    with pytest.raises(ValueError, match='独立修改'):
        RunRecord(episode, {'版本': 1})
    assert (episode / '工作稿.md').read_text() == '人工后续编辑'


def test_same_run_cannot_execute_concurrently(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match='正在执行'):
            with run_lock(tmp_path):
                pytest.fail('不应并行持有同一锁')


def test_actual_publication_branch_uses_existing_test_only_acceptance(workspace):
    # 这是显式既有验收夹具，不是把本次模拟审查的标记改成False。
    episode, _ = board(workspace)
    accepted(episode)
    review = episode / '派发结果/验收/全文.json'
    receipt = load(review.with_suffix('.coverage.json'))
    receipt['输入未变'] = True
    save(review.with_suffix('.coverage.json'), receipt)
    def publish(writes, result):
        for path, value in writes:
            path.write_text(value if isinstance(value, str) else __import__('json').dumps(value, ensure_ascii=False))
    result = finish_episode(episode, review, check_script(episode), publish)
    assert result['正式交付'] is True
    assert check_script(episode, '正式稿.md', formal=True)['通过']


def test_merge_keeps_stable_id_when_later_shot_has_first_appearance(tmp_path):
    initialize_board('测试', 1, root=tmp_path)
    episode = tmp_path / '测试/E01'
    data = load(FIXTURES / 'pending.json')['模型返回']['动作识别/E01-P01']
    first = data['镜头'][0]
    first['画面人物'][0].pop('外观')
    second = copy.deepcopy(first); second['镜头'] = 'E01-K002'
    second['画面人物'][0]['外观'] = '换姿势后的蓝外套短发'
    data['镜头'].append(second)
    save(episode / '片段表.json', {'片段': [{'id': 'E01-P01', '镜头': ['E01-K001', 'E01-K002']}]})
    result_path = episode / 'result.json'; save(result_path, data)
    captured = {}
    def publish(writes, result):
        captured.update({p.name: value for p, value in writes})
    merge_actions(episode, {'E01-P01': result_path}, publish)
    people = captured['临时人物.json']
    assert len(people) == 1 and people[0]['外观'] == '换姿势后的蓝外套短发'
    assert people[0]['出现镜头'] == ['E01-K001', 'E01-K002']
    save(episode / '派发结果/动作识别/E01-P01.json', data)
    _declarations(episode, people, [{'id': 'E01-K001'}, {'id': 'E01-K002'}])
