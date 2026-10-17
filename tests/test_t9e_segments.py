import pytest
from test_t2_segments import table
from s2s.segments import build_segments


def test_frame_budgets_and_complete_coverage():
    source = table(20)
    for shot in source['镜头']:
        shot['帧'] = [{}] * 12
    result = build_segments(source)
    assert [x for s in result['片段'] for x in s['镜头']] == [s['id'] for s in source['镜头']]
    for segment in result['片段']:
        assert len(segment['镜头']) * 12 <= 36
        assert (len(segment['镜头']) + bool(segment['衔接前']) + bool(segment['衔接后'])) * 12 <= 48


def test_impossible_budget():
    source = table(1)
    source['镜头'][0]['帧'] = [{}] * 12
    with pytest.raises(ValueError, match='单镜头'):
        build_segments(source, frame_budget=5)


@pytest.mark.parametrize('kw', [{'frame_budget': 0}, {'context_budget': -1}, {'target': 9, 'soft_limit': 8}])
def test_invalid_limits(kw):
    with pytest.raises(ValueError):
        build_segments(table(1), **kw)
