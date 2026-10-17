"""正式稿与导出入口必须使用当前完整验收，不靠状态或计数放行。"""

import pytest

from s2s import production as p
from s2s.dispatch import controllers
from s2s.quality import check_review, check_script
from test_production_pipeline import workspace, board, accepted


def receipt(episode):
    return episode / '派发结果/验收/全文.coverage.json'


def test_review_counts_without_actual_frame_reads_are_not_evidence(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    path = receipt(episode)
    data = p.read(path)
    data['片段'][0].pop('已读帧', None)
    p.write(path, data)
    assert not check_review(episode)


def test_repeating_same_frame_does_not_count_as_full_coverage(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    path = receipt(episode)
    data = p.read(path)
    data['片段'][0].update(已读帧=['同一帧', '同一帧'], 已读帧数=2, 授权帧数=2)
    p.write(path, data)
    assert not check_review(episode)


def test_partial_review_cannot_authorize_full_script(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    path = receipt(episode)
    data = p.read(path)
    review = data['验收结果']
    review['审查范围'] = {'行': '1-1', '镜头': '无', '场景': [], '对白句': '无'}
    p.write(path, data)
    p.write(path.with_name('全文.json'), review)
    assert not check_review(episode)


def test_formal_step_eight_is_not_marked_when_validation_fails(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    (episode / '正式稿.md').write_text('第1集\n林舟：改写对白。', encoding='utf-8')
    with pytest.raises(ValueError):
        controllers.mark_step('测试剧', 1, 8, root=workspace / '剧目')
    assert not p.read(episode / '状态.json')['步骤']['8']


def test_formal_step_eight_can_be_completed_only_after_current_acceptance(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    assert controllers.mark_step('测试剧', 1, 8, root=workspace / '剧目')['步骤']['8']


def test_invalid_formal_blocks_all_word_and_markdown_export(workspace):
    episode, _ = board(workspace)
    text = accepted(episode)
    (episode / '正式稿.md').write_text(text.replace('等一下。', '你给我站住！'), encoding='utf-8')
    before = (episode / '正式稿.md').read_bytes()
    with pytest.raises(ValueError):
        p.export_series(episode.parent, 1)
    assert not list((episode.parent / '合并').iterdir())
    assert (episode / '正式稿.md').read_bytes() == before


def test_solved_status_without_recheck_never_releases_formal(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    p.write(episode / '待核对.json', [{
        'id': 'E01-Q001', '来源': '识别', '位置': {'句': 'E01-S001'}, '原文': '等一下。',
        '类型': '全文完整性与一致性', '级别': '阻断', '相关人物': [], '依据': '待回听',
        '修正要求': '回听', '处理职责': '回听', '状态': '已裁定', '复核结果': None}])
    with pytest.raises(ValueError, match='未决问题'):
        p.check_episode_delivery(episode)


def test_missing_frame_makes_old_acceptance_unusable(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    frame = episode / '素材/frames/K001_01.jpg'
    frame.unlink()
    with pytest.raises(ValueError, match='独立验收'):
        p.check_episode_delivery(episode)


def test_simulation_flag_cannot_be_overridden_by_delivery_state(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    path = receipt(episode)
    data = p.read(path)
    data['模拟调用'] = True
    p.write(path, data)
    assert p.read(episode / '状态.json')['状态'] == '验收通过'
    assert not check_script(episode, '正式稿.md', formal=True)['通过']


def test_source_video_replaced_after_review_blocks_delivery(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    video = workspace / 'E01.mp4'
    video.write_bytes(b'replaced-video')
    with pytest.raises(ValueError, match='独立验收'):
        p.check_episode_delivery(episode)


def test_pending_human_cannot_mark_formal_step_complete(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    state = p.read(episode / '状态.json')
    state['状态'] = '待人工裁定'
    p.write(episode / '状态.json', state)
    with pytest.raises(ValueError, match='第 8 步'):
        controllers.mark_step('测试剧', 1, 8, root=workspace / '剧目')
    assert not p.read(episode / '状态.json')['步骤']['8']


@pytest.mark.parametrize('field,value', [('镜头', '无'), ('场景', []), ('对白句', '无对白')])
def test_review_scope_cannot_omit_current_material(workspace, field, value):
    episode, _ = board(workspace)
    accepted(episode)
    path = receipt(episode)
    data = p.read(path)
    review = data['验收结果']
    review['审查范围'][field] = value
    p.write(path, data)
    p.write(path.with_name('全文.json'), review)
    assert not check_review(episode)


def test_newer_failed_review_does_not_reuse_older_success(workspace):
    episode, _ = board(workspace)
    accepted(episode)
    data = p.read(receipt(episode))
    review = dict(data['验收结果'])
    review['结论'] = '不通过'
    data['验收结果'] = review
    target = receipt(episode).with_name('最新.coverage.json')
    p.write(target.with_name('最新.json'), review)
    p.write(target, data)
    assert not check_review(episode)


def test_direct_full_series_export_requires_cross_episode_acceptance(workspace):
    last, _ = board(workspace, 2)
    accepted(last.parent / 'E01')
    accepted(last)
    with pytest.raises((ValueError, OSError), match='跨集|cross'):
        p.export_series(last.parent, 2)
    assert not list((last.parent / '合并').iterdir())
