"""E0入口合同：所有视频/识别/模型均显式模拟，不能据此授权正式稿。"""

import copy
import json
from pathlib import Path
import subprocess

import pytest

from s2s import agent, config, production
from s2s.dispatch import controllers
from s2s.fixed_pipeline import run_fixed_episode
from s2s.offline_episode import OfflineEpisode
from s2s.quality import check_review, check_script
from s2s.run_record import RunRecord, load, save

FIXTURES = Path(__file__).parent / 'fixtures/fixed_episode'
VIDEO = FIXTURES / 'input.mp4'


def run(tmp_path, name='pass', **kwargs):
    return production.run_episode('离线剧', 1, video=VIDEO, fixture=FIXTURES / (name + '.json'),
                                  run_root=tmp_path / 'run', scheduler='固定', **kwargs)


@pytest.mark.parametrize('name,shadow', [('pass', '正式交付候选'), ('pending', '待人工裁定')])
def test_entry_to_both_shadow_terminals_cannot_publish(tmp_path, monkeypatch, name, shadow):
    def forbidden(*args, **kwargs):
        pytest.fail('离线合同不得调用真实模型或长Pi主控')
    monkeypatch.setattr(agent, 'run', forbidden)
    monkeypatch.setattr(controllers, 'start_controller', forbidden)
    result = run(tmp_path, name)
    assert result['状态'] == '待人工裁定' and result['离线影子终态'] == shadow
    assert result['模拟运行'] and not result['正式交付']
    episode = Path(result['目录'])
    assert not (episode / '正式稿.md').read_text()
    assert (episode / '工作稿.md').read_text().strip()
    receipt = load(episode / '派发结果/验收/全文.coverage.json')
    assert receipt['模拟调用'] is True and receipt['全文审查完成'] is True
    assert not check_review(episode)
    assert not check_script(episode, formal=True)['通过']
    ledger = load(result['调用账本'])
    assert ledger['真实远程请求数'] == 0 and ledger['模拟调用数'] >= 6
    assert ledger['已结算美元'] is None and not ledger['金额完整']
    assert all(row['官方usage'] is None and row['实际美元'] is None for row in ledger['请求'])
    stages = load(result['阶段账本'])['阶段']
    assert all(row['状态'] == '完成' for row in stages.values())
    assert {'角色卡两步', '人物审核apply', '成稿与排布', '机械检查', '独立全文验收', '终态发布'} <= stages.keys()


def test_same_run_reuses_completed_stages(tmp_path, monkeypatch):
    first = run(tmp_path)
    before = load(first['调用账本'])
    monkeypatch.setattr(OfflineEpisode, 'call', lambda *a, **k: pytest.fail('完成阶段不应重跑'))
    assert run(tmp_path) == first
    assert load(first['调用账本']) == before


@pytest.mark.parametrize('kind', ['fixture', 'artifact', 'video'])
def test_content_hashes_reject_changed_input_or_artifact(tmp_path, kind):
    video = tmp_path / 'input.mp4'; video.write_bytes(VIDEO.read_bytes())
    fixture = tmp_path / 'case.json'; fixture.write_bytes((FIXTURES / 'pass.json').read_bytes())
    kwargs = {'video': video, 'fixture': fixture, 'run_root': tmp_path / 'run'}
    result = run_fixed_episode('离线剧', 1, **kwargs)
    if kind == 'fixture':
        fixture.write_text(fixture.read_text() + ' ')
    elif kind == 'video':
        video.write_bytes(b'changed-input')
    else:
        (Path(result['目录']) / '动作记录.json').write_text('[]')
    with pytest.raises(ValueError, match='版本已变化|外部修改'):
        run_fixed_episode('离线剧', 1, **kwargs)


@pytest.mark.parametrize('filename', ['tools/ocr/vocr.swift', 'pyproject.toml', 'uv.lock'])
def test_runtime_ocr_and_dependency_changes_reject_resume(tmp_path, monkeypatch, filename):
    from s2s import run_record
    runtime = tmp_path / 'runtime'
    for name in ('tools/ocr/vocr.swift', 'pyproject.toml', 'uv.lock'):
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen-runtime-input\n')
    monkeypatch.setattr(run_record, 'RUNTIME_ROOT', runtime)
    result = run(tmp_path)
    frozen = load(result['冻结'])
    assert len(frozen['代码SHA256']['tools/ocr/vocr.swift']) == 64
    assert set(frozen['依赖文件SHA256']) == {'pyproject.toml', 'uv.lock'}
    assert all(len(value) == 64 for value in frozen['依赖文件SHA256'].values())
    (runtime / filename).write_text('changed-runtime-input\n')
    with pytest.raises(ValueError, match='代码版本已变化'):
        run(tmp_path)


def test_live_preflight_fails_before_preprocessing_or_remote_request(tmp_path, monkeypatch):
    monkeypatch.setattr('s2s.fixed_pipeline.prepare_episode', lambda *a, **k: pytest.fail('不应开始预处理'))
    monkeypatch.setattr(agent, 'run', lambda *a, **k: pytest.fail('不应发出付费请求'))
    with pytest.raises(RuntimeError, match='真实调用预检阻断.*E1'):
        production.run_episode('离线剧', 1, video=VIDEO, run_root=tmp_path / 'run', scheduler='固定')
    assert not (tmp_path / 'run').exists()


def test_live_preflight_cli_is_explicit(tmp_path, capsys):
    code = controllers.episode_main(['离线剧', '1', '--调度', '固定', '--视频', str(VIDEO), '--运行目录', str(tmp_path / 'run'), '--预检'])
    result = json.loads(capsys.readouterr().out)
    assert code == 1 and '真实调用预检阻断' in result['错误']


def test_offline_cli_single_command(tmp_path):
    command = [str(Path(__import__('sys').executable)), 'tools/run_episode.py', '离线剧', '1', '--调度', '固定', '--视频', str(VIDEO),
               '--离线夹具', str(FIXTURES / 'pass.json'), '--运行目录', str(tmp_path / 'run')]
    result = subprocess.run(command, cwd=FIXTURES.parents[2], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)['离线影子终态'] == '正式交付候选'


def test_failed_draft_resumes_without_redoing_upstream(tmp_path, monkeypatch):
    original = OfflineEpisode.call
    calls = []
    fail = True
    def call(self, role, prompt, **kwargs):
        calls.append(role)
        if role == '成稿' and fail:
            raise RuntimeError('模拟成稿中断')
        return original(self, role, prompt, **kwargs)
    monkeypatch.setattr(OfflineEpisode, 'call', call)
    with pytest.raises(RuntimeError, match='模拟成稿中断'):
        run(tmp_path)
    fail = False
    calls.clear()
    result = run(tmp_path)
    assert result['离线影子终态'] == '正式交付候选'
    assert '动作识别' not in calls and '角色卡' not in calls
    row = load(result['阶段账本'])['阶段']['成稿与排布']
    assert row['尝试'] == 2 and row['历史'][0]['状态'] == '失败'


@pytest.mark.parametrize('failed_key,reused_key', [('角色卡/对白归属', '角色卡/人物合并'), ('验收/全文审查/E01-P01', '验收/全文审查/整集')])
def test_internal_stage_resume_reuses_successful_semantic_parts(tmp_path, monkeypatch, failed_key, reused_key):
    original = OfflineEpisode.call
    calls, fail = [], True
    def call(self, role, prompt, **kwargs):
        stage = kwargs['request_budget'].stage
        calls.append(stage)
        if fail and stage.startswith(failed_key):
            raise RuntimeError('模拟分阶段中断')
        return original(self, role, prompt, **kwargs)
    monkeypatch.setattr(OfflineEpisode, 'call', call)
    with pytest.raises(RuntimeError, match='分阶段中断'):
        run(tmp_path)
    assert any(stage.startswith(reused_key) for stage in calls)
    fail = False; calls.clear()
    result = run(tmp_path)
    assert result['离线影子终态'] == '正式交付候选'
    assert any(stage.startswith(failed_key) for stage in calls)
    assert not any(stage.startswith(reused_key) for stage in calls)


def test_unrecorded_added_audit_artifact_is_rejected(tmp_path):
    result = run(tmp_path)
    save(Path(result['目录']) / '派发结果/验收/外部.coverage.json', {'模拟调用': False})
    with pytest.raises(ValueError, match='未记账产物'):
        run(tmp_path)


def test_interrupted_preprocessing_revalidates_before_any_semantic_stage(tmp_path, monkeypatch):
    original = OfflineEpisode.prepare
    def interrupted(self, video, episode):
        original(self, video, episode)
        raise KeyboardInterrupt('模拟预处理进程中断')
    monkeypatch.setattr(OfflineEpisode, 'prepare', interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path)
    monkeypatch.setattr(OfflineEpisode, 'prepare', original)
    result = run(tmp_path)
    row = load(result['阶段账本'])['阶段']['固定预处理']
    assert row['尝试'] == 2 and row['历史'][0]['状态'] == '中断'
    assert '输入校验前不派发模型' in row['历史'][0]['恢复要求']
    assert result['离线影子终态'] == '正式交付候选'


def test_missing_approval_preserves_n_and_no_confirmed_c(tmp_path):
    result = run(tmp_path, 'pending')
    episode = Path(result['目录'])
    assert load(episode.parent / '角色卡.json') == []
    assert load(episode / '归属表.json')[0]['说话人'] == 'E01-N001'
    assert load(episode / '本集人物.json')['人物'][0]['状态'] == '待审核'
    assert load(result['阶段账本'])['阶段']['人物审核apply']['结果']['获批映射'] == {}
    assert '没有批准' in load(result['阶段账本'])['阶段']['人物审核apply']['结果']['审核状态']


def approved_new_person_fixture(tmp_path):
    data = load(FIXTURES / 'pending.json')
    data['模型返回']['成稿']['工作稿'] = '第1集\n场景1-1：门口，日，外\n人物：穿蓝外套的短发青年\n穿蓝外套的短发青年：等一下。\n'
    path = tmp_path / 'approved-case.json'; save(path, data)
    return path


@pytest.mark.parametrize('boundary', range(1, 7))
def test_interrupted_apply_rolls_forward_without_duplicate_ids(tmp_path, monkeypatch, boundary):
    fixture = approved_new_person_fixture(tmp_path)
    audit = tmp_path / 'audit.json'
    kwargs = {'video': VIDEO, 'fixture': fixture, 'run_root': tmp_path / 'run', 'approvals': audit}
    first = run_fixed_episode('离线剧', 1, **kwargs)
    assert first['状态'] == '等待显式人物审核'
    template = load(first['审核模板'])
    template['批准'] = [{'人物': 'E01-N001', '依据': '仅离线测试的显式审核，不是真实人物确认'}]
    save(audit, template)
    original = RunRecord._apply_transaction
    active = False
    original_publish = RunRecord.publish
    def publish(self, writes, result):
        nonlocal active
        active = self.data['活动阶段'] == '人物审核apply'
        try:
            return original_publish(self, writes, result)
        finally:
            active = False
    def interrupted(transaction):
        if active:
            original({'文件': transaction['文件'][:boundary]})
            raise RuntimeError('模拟apply发布中断')
        return original(transaction)
    monkeypatch.setattr(RunRecord, 'publish', publish)
    monkeypatch.setattr(RunRecord, '_apply_transaction', staticmethod(interrupted))
    with pytest.raises(RuntimeError, match='发布中断'):
        run_fixed_episode('离线剧', 1, **kwargs)
    monkeypatch.setattr(RunRecord, '_apply_transaction', staticmethod(original))
    result = run_fixed_episode('离线剧', 1, **kwargs)
    episode = Path(result['目录'])
    assert [r['id'] for r in load(episode.parent / '角色卡.json')] == ['C001']
    assert load(episode / '归属表.json')[0]['说话人'] == 'C001'
    assert load(result['阶段账本'])['阶段']['人物审核apply']['中断恢复'] is True
    assert result['离线影子终态'] == '正式交付候选'


def test_unfrozen_existing_directory_is_not_overwritten(tmp_path):
    episode = tmp_path / 'run/离线剧/E01'; episode.mkdir(parents=True)
    old = episode / '工作稿.md'; old.write_text('既有稿件')
    with pytest.raises(ValueError, match='拒绝采纳或覆盖'):
        run(tmp_path)
    assert old.read_text() == '既有稿件'


def test_legacy_scheduler_is_default_and_fixed_is_explicit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(production, '_run_episode_controller', lambda *a, **k: {'通过': True, '旧主控': True})
    assert production.run_episode('测试', 1)['旧主控']
    assert production.run_episode('测试', 1, scheduler='主控')['旧主控']
    assert controllers.episode_main(['测试', '1']) == 0
    assert json.loads(capsys.readouterr().out)['旧主控'] is True
    with pytest.raises(ValueError, match='固定首阶段'):
        production.run_episode('测试', 1, scheduler='固定', mode='定向复核', review='核对')
    with pytest.raises(ValueError, match='旧主控不接受'):
        production.run_episode('测试', 1, scheduler='主控', fixture=FIXTURES / 'pass.json')
