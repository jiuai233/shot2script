import pytest
from test_t2_segments import table
from s2s.segments import build_segments, per_second
from s2s.timefmt import fmt


def long_shots(count, seconds=4):
    """每镜头 seconds 秒、按 0.5 秒取的本地帧；动作识别每秒只送 1 帧。"""
    shots = []
    for i in range(count):
        start = i * seconds
        shots.append({'id': f'E01-K{i + 1:03d}', '起': fmt(start), '止': fmt(start + seconds),
                      '帧': [{'文件': f'K{i + 1:03d}_{j + 1:02d}.jpg', '时间': fmt(start + 0.1 + j * 0.5)}
                            for j in range(seconds * 2)]})
    return {'集': 1, '镜头': shots}


def test_per_second_picks_one_frame_each_second_and_keeps_short_shots():
    shot = long_shots(1)['镜头'][0]
    assert [f['时间'] for f in per_second(shot)] == ['00:00.6', '00:01.6', '00:02.6', '00:03.6']
    short = {'起': '00:03.0', '止': '00:03.4', '帧': [{'时间': t} for t in ('00:03.1', '00:03.2', '00:03.3')]}
    assert per_second(short) == [{'时间': '00:03.2'}]


def test_frame_budgets_count_sent_frames_and_cover_all_shots():
    source = long_shots(30)
    result = build_segments(source)
    assert [x for s in result['片段'] for x in s['镜头']] == [s['id'] for s in source['镜头']]
    for segment in result['片段']:
        assert len(segment['镜头']) * 4 <= 40
        assert (len(segment['镜头']) + bool(segment['衔接前']) + bool(segment['衔接后'])) * 4 <= 48
    # 本地帧 8 张/镜头时 10 个镜头已有 80 张，仍按送出的 4 张/镜头放入同一段。
    assert len(result['片段'][0]['镜头']) == 10


def test_impossible_budget():
    with pytest.raises(ValueError, match='单镜头'):
        build_segments(long_shots(1, seconds=10), frame_budget=5)


@pytest.mark.parametrize('kw', [{'frame_budget': 0}, {'context_budget': -1}, {'target': 9, 'soft_limit': 8}])
def test_invalid_limits(kw):
    with pytest.raises(ValueError):
        build_segments(table(1), **kw)
