"""生产入口离线验证；模型和识别替换为夹具，Word 使用真实生成器。"""

import json
from pathlib import Path

import pytest

from s2s import config, production as p
from s2s.board import initialize_board
from s2s.dispatch import controllers
from s2s.quality import review_version

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'PROJECT_ROOT', tmp_path)
    (tmp_path / 'prompts').symlink_to(ROOT / 'prompts', target_is_directory=True)
    return tmp_path


def board(root, number=1):
    initialize_board('测试剧', number, root=root / '剧目')
    episode = root / '剧目/测试剧' / f'E{number:02d}'
    video = root / f'E{number:02d}.mp4'
    video.write_bytes(b'video')
    return episode, video


def accepted(episode):
    number = int(episode.name[1:])
    prefix = episode.name
    text = f'第{number}集\n场景{number}-1：门口，日，外\n人物：林舟\n林舟：等一下。\n'
    for name in ('工作稿.md', '正式稿.md'):
        (episode / name).write_text(text, encoding='utf-8')
    (episode / '验收报告.md').write_text('全文独立验收通过。', encoding='utf-8')
    p.write(episode.parent / '角色卡.json', [{
        'id': 'C001', '名称': '林舟', '名称类型': '姓名', '状态': '已确认',
        '别名称呼': [], '外观': [], '声线': [], '关系': [], '身份依据': ['人物字卡'],
        '性格总结': [], '首次出场': 'E01-K001', '待核对': []}])
    video = episode.parents[2] / f'{prefix}.mp4'
    if not video.exists():
        video.write_bytes(b'fixture-video')
    p.write(episode / '镜头表.json', {'集': number, '视频': str(video), '时长': '00:02.0', '镜头': [
        {'id': f'{prefix}-K001', '起': '00:00.0', '止': '00:02.0', '帧': [{'文件': 'K001_01.jpg', '时间': '00:00.1'}]}]})
    p.write(episode / '声音记录.json', [{'id': f'{prefix}-S001', '起': '00:00.1', '止': '00:01.0',
                                          '文本': '等一下。', '声线': f'{prefix}-V1', '听辨': '清楚'}])
    p.write(episode / '归属表.json', [{'句': f'{prefix}-S001', '说话人': 'C001', '形态': '画内',
                                   '依据': '口型及人物字卡', '确定度': '确认'}])
    p.write(episode / '对照表.json', [
        {'行': line, '类型': kind, '句': [f'{prefix}-S001'] if line == 4 else [], '动作': [],
         '场景': f'场景{number}-1', '镜头': [f'{prefix}-K001']}
        for line, kind in ((2, '场景行'), (3, '人物行'), (4, '对白'))])
    p.write(episode / '片段表.json', {'集': number, '片段': [
        {'id': f'{prefix}-P01', '镜头': [f'{prefix}-K001'], '起': '00:00.0', '止': '00:02.0',
         '衔接前': None, '衔接后': None}]})
    frame = episode / '素材/frames/K001_01.jpg'
    frame.write_bytes(b'fixture-frame')
    review = {'模式': '全文审查', '集': number,
              '审查范围': {'行': '1-4', '场景': [f'场景{number}-1'],
                           '镜头': f'{prefix}-K001 至 {prefix}-K001', '对白句': f'{prefix}-S001 至 {prefix}-S001'},
              '统计': {'阻断': 0, '规范': 0}, '结论': '通过', '问题': [], '复核': []}
    p.write(episode / '派发结果/验收/全文.json', review)
    p.write(episode / '派发结果/验收/全文.coverage.json', {
        '全文审查完成': True, '片段': [{'片段': f'{prefix}-P01', '状态': '完成', '授权帧数': 1, '已读帧数': 1, '已读帧': [str(frame.resolve())]}],
        '模拟调用': False, '输入版本': review_version(episode), '验收结果': review})
    state = p.read(episode / '状态.json')
    state.update(状态='验收通过')
    p.write(episode / '状态.json', state)
    return text


def mock_phases(monkeypatch):
    calls = []
    original = p.PHASES
    phases = []
    for name, _, inputs, outputs in original:
        def operation(video, episode, name=name, outputs=outputs):
            calls.append(name)
            for output in outputs:
                if output == '镜头表.json':
                    p.write(episode / output, {'视频': str(video), '镜头': [{'帧': [{'文件': 'f.jpg'}]}]})
                    frame = episode / '素材/frames/f.jpg'
                    frame.write_bytes(b'frame')
                elif output == '声音记录.json':
                    p.write(episode / output, [{'文本': '等一下。', '声线': 'E01-V1'}])
                elif output.endswith('.json'):
                    p.write(episode / output, [])
                else:
                    (episode / output).write_text('回听页', encoding='utf-8')
        phases.append((name, operation, inputs, outputs))
    monkeypatch.setattr(p, 'PHASES', tuple(phases))
    monkeypatch.setattr(p, 'validate_file', lambda *a, **k: {'通过': True})
    return calls


def test_discover_order_duplicates_and_missing(tmp_path):
    for name in ('E02.MP4', '01.mov', 'README.txt'):
        (tmp_path / name).write_bytes(b'input')
    assert list(sorted(p.discover_videos(tmp_path, 2))) == [1, 2]
    with pytest.raises(ValueError, match='缺失视频：E03'):
        p.discover_videos(tmp_path, 3)
    (tmp_path / 'E01.mp4').write_bytes(b'input')
    with pytest.raises(ValueError, match='多个视频'):
        p.discover_videos(tmp_path, 2)


def test_prepare_order_and_resume(workspace, monkeypatch):
    episode, video = board(workspace)
    calls = mock_phases(monkeypatch)
    p.prepare_episode(video, episode)
    assert calls == ['镜头', '分段', '语音', '字幕', '口型', '声纹', '回听']
    calls.clear()
    p.prepare_episode(video, episode)
    assert not calls
    (episode / '口型记录.json').unlink()
    p.prepare_episode(video, episode)
    assert calls == ['口型', '声纹']


def test_audio_edits_refresh_downstream_without_overwriting_human(workspace, monkeypatch):
    episode, video = board(workspace)
    calls = mock_phases(monkeypatch)
    p.prepare_episode(video, episode)
    calls.clear()
    p.write(episode / '声音记录.json', [{'文本': '人工改字', '声线': 'E01-V1'}])
    p.prepare_episode(video, episode)
    assert calls == ['口型', '声纹', '回听']
    assert p.read(episode / '声音记录.json')[0]['文本'] == '人工改字'


def test_phase_failure_stops_and_resume_does_not_repeat_upstream(workspace, monkeypatch):
    episode, video = board(workspace)
    calls = mock_phases(monkeypatch)
    phases = list(p.PHASES)
    def fail(*args):
        calls.append('字幕失败')
        raise RuntimeError('OCR 不可用')
    phases[3] = ('字幕', fail, phases[3][2], phases[3][3])
    saved = p.PHASES
    monkeypatch.setattr(p, 'PHASES', tuple(phases))
    with pytest.raises(RuntimeError, match='OCR 不可用'):
        p.prepare_episode(video, episode)
    assert calls == ['镜头', '分段', '语音', '字幕失败']
    job = p.read(episode / '生产任务.json')
    assert job['状态'] == '失败' and job['当前阶段'] == '字幕'
    monkeypatch.setattr(p, 'PHASES', saved)
    calls.clear()
    p.prepare_episode(video, episode)
    assert calls == ['字幕', '口型', '声纹', '回听']


def test_changed_video_rejected_before_processing(workspace, monkeypatch):
    episode, video = board(workspace)
    calls = mock_phases(monkeypatch)
    p.prepare_episode(video, episode)
    calls.clear()
    video.write_bytes(b'changed-video')
    with pytest.raises(ValueError, match='其他视频版本'):
        p.prepare_episode(video, episode)
    assert not calls


def test_controller_exit_is_not_acceptance(workspace):
    episode, _ = board(workspace)
    with pytest.raises(ValueError, match='主控已退出'):
        p.check_episode_delivery(episode)


def test_pending_human_is_draft_not_formal(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    state = p.read(episode / '状态.json')
    state['状态'] = '待人工裁定'
    p.write(episode / '状态.json', state)
    result = p.check_episode_delivery(episode)
    assert result['通过'] and not result['正式交付']
    assert 'Markdown' not in result and '草稿' in result


def test_formal_requires_independent_coverage_and_current_problems_resolved(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    assert p.check_episode_delivery(episode)['正式交付']
    p.write(episode / '待核对.json', [{'状态': '待人工裁定'}])
    with pytest.raises(ValueError, match='待核对'):
        p.check_episode_delivery(episode)
    p.write(episode / '待核对.json', [])
    p.write(episode / '派发结果/验收/全文.coverage.json', {'全文审查完成': False})
    with pytest.raises(ValueError, match='独立验收'):
        p.check_episode_delivery(episode)


def test_bad_formal_dialogue_blocked(workspace):
    episode, _ = board(workspace)
    text = accepted(episode)
    (episode / '正式稿.md').write_text(text.replace('等一下。', '改写的台词。'), encoding='utf-8')
    with pytest.raises(ValueError, match='台词不一致'):
        p.check_episode_delivery(episode)


def test_real_markdown_and_word_export_excludes_pending_episode(workspace):
    episode, _ = board(workspace, 2)
    first = episode.parent / 'E01'
    text = accepted(first)
    state = p.read(episode / '状态.json')
    state['状态'] = '待人工裁定'
    p.write(episode / '状态.json', state)
    result = p.export_series(episode.parent, 2)
    assert result['通过'] and not result['正式交付'] and result['未收录集'] == [2]
    assert Path(result['Markdown']).read_text().endswith(text)
    assert '第2集' not in Path(result['Markdown']).read_text()
    assert p.check_docx(result['Word'], 2, missing={2})['通过']


def test_all_pending_no_formal_export(workspace):
    episode, _ = board(workspace)
    state = p.read(episode / '状态.json')
    state['状态'] = '待人工裁定'
    p.write(episode / '状态.json', state)
    result = p.export_series(episode.parent, 1)
    assert result['状态'] == '待人工裁定' and 'Word' not in result


def test_episode_runs_prepare_before_controller_and_reuses_terminal(workspace, monkeypatch):
    episode, video = board(workspace)
    events = []
    def prepare(source, directory):
        events.append('素材')
        for name in ('口型记录', '声纹候选'):
            if not (directory / f'{name}.json').exists():
                p.write(directory / f'{name}.json', {})
        return p.read(directory / '生产任务.json') if (directory / '生产任务.json').exists() else {}
    def controller(role, prompt, session, **kwargs):
        events.append('主控')
        assert role == '单集主控' and '从状态中尚未完成' in prompt
        accepted(episode)
        return {'通过': True}
    monkeypatch.setattr(p, 'prepare_episode', prepare)
    monkeypatch.setattr(controllers, 'start_controller', controller)
    assert p.run_episode('测试剧', 1, video=video)['正式交付']
    assert events == ['素材', '主控']
    events.clear()
    assert p.run_episode('测试剧', 1, video=video)['正式交付']
    assert events == ['素材']


def test_series_video_mapping_and_real_export(workspace, monkeypatch):
    videos = workspace / '素材'
    videos.mkdir()
    for number in (1, 2):
        (videos / f'E{number:02d}.mp4').write_bytes(b'video')
    def controller(role, prompt, session, **kwargs):
        assert role == '全剧主控' and '上一集角色变更审核完毕' in prompt
        series = workspace / '剧目/测试剧'
        for number in (1, 2):
            episode = series / f'E{number:02d}'
            source = p.resolve_video(series, number)
            assert source == videos / f'E{number:02d}.mp4'
            accepted(episode)
            p.write(episode / '生产任务.json', {'视频': p.stamp(source), '状态': '验收通过'})
        p.write(series / '派发结果/跨集验收.json', {'模式': '跨集', '结论': '通过', '问题': []})
        state = p.read(series / '全剧状态.json')
        state['状态'] = '验收通过'
        p.write(series / '全剧状态.json', state)
    monkeypatch.setattr(controllers, 'start_controller', controller)
    result = p.run_series('测试剧', 2, directory=videos)
    assert result['正式交付'] and p.check_docx(result['Word'], 2)['通过']
    assert p.read(workspace / '剧目/测试剧/生产任务.json')['状态'] == '验收通过'


def test_cli_returns_json_for_missing_source_and_no_model_call(workspace, capsys):
    assert controllers.episode_main(['测试剧', '1', '--视频', str(workspace / 'missing.mp4')]) == 1
    assert not json.loads(capsys.readouterr().out)['通过']
    assert controllers.series_main(['测试剧', '2', '--素材目录', str(workspace / 'missing')]) == 1
    assert not json.loads(capsys.readouterr().out)['通过']
