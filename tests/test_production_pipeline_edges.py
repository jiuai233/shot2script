"""生产入口的恢复、环境选择和跨集失败路径。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from s2s import production as p
from s2s.dispatch import controllers
from test_production_pipeline import workspace, board, accepted, mock_phases


def test_initial_failure_record_can_resume(workspace, monkeypatch):
    episode, video = board(workspace)
    p.write(episode / '生产任务.json', {'状态': '失败', '错误': '初始输入不可用'})
    calls = mock_phases(monkeypatch)
    assert p.prepare_episode(video, episode)['状态'] == '素材就绪'
    assert len(calls) == 7


def test_single_default_input_does_not_require_earlier_episodes(workspace):
    directory = workspace / 'videos/测试剧'
    directory.mkdir(parents=True)
    video = directory / 'E03.mp4'
    video.write_bytes(b'video')
    assert p.resolve_video(workspace / '剧目/测试剧', 3) == video


def test_checkpoint_command_needs_no_video(workspace, capsys):
    assert controllers.episode_main(['测试剧', '1', '--完成步骤', '3']) == 0
    assert p.read(workspace / '剧目/测试剧/E01/状态.json')['步骤']['3']


def test_missing_frame_rebuilds_shots_and_dependent_materials(workspace, monkeypatch):
    episode, video = board(workspace)
    calls = mock_phases(monkeypatch)
    p.prepare_episode(video, episode)
    (episode / '素材/frames/f.jpg').unlink()
    calls.clear()
    p.prepare_episode(video, episode)
    assert '镜头' in calls and '分段' in calls and '字幕' in calls and '语音' not in calls


def test_lipsync_uses_configured_local_python_and_keeps_evidence(workspace, monkeypatch):
    episode, video = board(workspace)
    monkeypatch.setattr(p.config, 'get_environment', lambda: {'S2S_LIPSYNC_PYTHON': '/local/mediapipe/python'})
    def run(command, **kwargs):
        assert command[0] == '/local/mediapipe/python'
        assert kwargs['timeout'] == 1200
        output = Path(command[4])
        assert output.parent.parent == episode / '素材/口型'
        p.write(output, {'句': [], '证据目录': str(output.parent)})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess, 'run', run)
    p.prepare_lipsync(video, episode)
    assert p.read(episode / '口型记录.json')['证据目录']


def test_controller_failure_saved_once_without_retry(workspace, monkeypatch):
    videos = workspace / '输入'
    videos.mkdir()
    (videos / 'E01.mp4').write_bytes(b'video')
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('运行器超时')
    monkeypatch.setattr(controllers, 'start_controller', fail)
    with pytest.raises(RuntimeError, match='运行器超时'):
        p.run_series('测试剧', 1, directory=videos)
    job = p.read(workspace / '剧目/测试剧/生产任务.json')
    assert calls == [1] and job['状态'] == '失败' and '超时' in job['错误']
    assert not list((workspace / '剧目/测试剧/合并').iterdir())


def test_unfinished_cross_review_cannot_export_formal_series(workspace, monkeypatch):
    videos = workspace / '输入'
    videos.mkdir()
    for number in (1, 2):
        (videos / f'E{number:02d}.mp4').write_bytes(b'video')
    def controller(*args, **kwargs):
        series = workspace / '剧目/测试剧'
        for number in (1, 2):
            episode = series / f'E{number:02d}'
            accepted(episode)
            p.write(episode / '生产任务.json', {'视频': p.stamp(videos / f'E{number:02d}.mp4'), '状态': '验收通过'})
        p.write(series / '全剧状态.json', {'状态': '验收通过', '集': []})
        p.write(series / '派发结果/跨集验收.json', {'模式': '跨集', '结论': '不通过', '问题': ['名称冲突']})
    monkeypatch.setattr(controllers, 'start_controller', controller)
    with pytest.raises(ValueError, match='跨集验收尚未通过'):
        p.run_series('测试剧', 2, directory=videos)
    assert not list((workspace / '剧目/测试剧/合并').iterdir())


def test_current_formal_with_wrong_attribution_is_blocked(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    row = p.read(episode / '归属表.json')[0]
    row['确定度'] = '存疑'
    p.write(episode / '归属表.json', [row])
    with pytest.raises(ValueError, match='对白归属未确认'):
        p.check_episode_delivery(episode)
