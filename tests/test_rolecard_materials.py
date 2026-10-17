import json
from pathlib import Path
import subprocess
import sys

import pytest

from s2s.dispatch.rolecard_materials import build_rolecard_materials


ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


@pytest.fixture
def snapshot(tmp_path):
    episode = tmp_path / '测试剧/E01'

    def create(count=2, *, shared=True):
        people, shots, results = [], [], []
        for index in range(count):
            person = f'E01-P01-T{index + 1}'
            shot_id = 'E01-K001' if shared else f'E01-K{index + 1:03d}'
            filename = 'K001_01.jpg' if shared else f'K{index + 1:03d}_01.jpg'
            people.append({'id': person, '片段': 'E01-P01', '外观': f'外观{index}',
                           '出现镜头': [shot_id], '映射': None, '映射依据': '映射隔离哨兵'})
            if not shared or not shots:
                shots.append({'id': shot_id, '起': '00:00.0', '止': '00:01.0',
                              '帧': [{'文件': filename, '时间': '00:00.1'}]})
                results.append({'镜头': shot_id, '画面人物': [], '动作': [], '神态': []})
                frame = episode / '素材/frames' / filename
                frame.parent.mkdir(parents=True, exist_ok=True)
                frame.write_bytes(b'fake-frame')
            row = results[0] if shared else results[-1]
            row['画面人物'].append({'人物': person, '外观': f'外观{index}'})
            row['动作'].append({'执行者': person, '依据帧': [filename],
                                '确定度': '确认', '动作': '动作全文隔离哨兵'})
        save(episode / '临时人物.json', people)
        save(episode / '镜头表.json', {'镜头': shots})
        save(episode / '派发结果/动作识别/E01-P01.json', {'片段': 'E01-P01', '镜头': results})
        return episode

    return create


def update_result(episode, change):
    path = episode / '派发结果/动作识别/E01-P01.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    change(data)
    save(path, data)


def test_shared_frame_keeps_all_labels_and_is_deterministic(snapshot):
    episode = snapshot()
    data = build_rolecard_materials(episode, max_images=64)
    assert data == build_rolecard_materials(episode, max_images=64)
    assert len(data['images']) == len(data['labels']) == 1
    assert all(p['id'] in data['labels'][0] for p in data['临时人物'])
    assert data['统计']['候选关联人数'] == 2
    assert data['统计']['容量状态'] == '候选容量内'
    assert data['临时人物'][0]['候选帧'][0]['选择依据']['人物'] == 'E01-P01-T1'


def test_68_declarations_are_not_68_images(snapshot):
    data = build_rolecard_materials(snapshot(68), max_images=64)
    assert len(data['临时人物']) == data['统计']['候选关联人数'] == 68
    assert data['统计']['实际唯一图数U'] == 1
    assert data['统计']['缺失人数'] == 0
    assert data['统计']['容量状态'] == '候选容量内'
    assert all(p['id'] + '｜' in data['labels'][0] for p in data['临时人物'])


def test_65_distinct_images_report_over_limit_without_truncation(snapshot):
    data = build_rolecard_materials(snapshot(65, shared=False), max_images=64)
    assert len(data['images']) == len(data['labels']) == 65
    assert data['统计']['候选关联人数'] == 65
    assert data['统计']['容量状态'] == '候选超限：需后续分批'
    assert data['统计']['超过图片上限'] is True


def test_missing_person_evidence_is_not_filled_from_other_person(snapshot):
    episode = snapshot()
    update_result(episode, lambda d: d['镜头'][0]['动作'].pop(0))
    data = build_rolecard_materials(episode, max_images=64)
    assert data['临时人物'][0]['候选帧'] == []
    assert len(data['临时人物']) == 2
    assert data['统计']['候选关联人数'] == data['统计']['缺失人数'] == 1
    assert data['缺失项'][0]['人物'] == 'E01-P01-T1'
    assert data['统计']['容量状态'] == '未满足：候选材料不完整'
    assert data['统计']['材料状态'] == '候选材料不完整'


def test_no_evidence_does_not_count_zero_images_as_satisfied(snapshot):
    episode = snapshot()
    update_result(episode, lambda d: d['镜头'][0].update(动作=[]))
    data = build_rolecard_materials(episode, max_images=64)
    assert data['images'] == data['labels'] == []
    assert data['统计']['缺失人数'] == 2
    assert data['统计']['容量状态'] != '候选容量内'


def test_expression_is_eligible_and_earliest_referenced_frame_wins(snapshot):
    episode = snapshot(1)
    path = episode / '镜头表.json'
    shots = json.loads(path.read_text(encoding='utf-8'))
    shots['镜头'][0]['帧'].append({'文件': 'K001_02.jpg', '时间': '00:00.8'})
    save(path, shots)
    (episode / '素材/frames/K001_02.jpg').write_bytes(b'fake-frame')

    def change(data):
        row = data['镜头'][0]
        row['动作'][0]['依据帧'] = ['K001_02.jpg']
        row['神态'] = [{'人物': 'E01-P01-T1', '时间': '00:00.1',
                      '神态': '神态全文隔离哨兵', '依据帧': ['K001_01.jpg']}]
    update_result(episode, change)
    data = build_rolecard_materials(episode, max_images=64)
    chosen = data['临时人物'][0]['候选帧'][0]
    assert chosen['文件'] == 'K001_01.jpg'
    assert chosen['选择依据']['类型'] == '神态依据帧'
    assert chosen['时间'] == '00:00.1'
    assert '神态全文隔离哨兵' not in json.dumps(data, ensure_ascii=False)


def test_first_shot_frame_is_not_used_without_person_reference(snapshot):
    episode = snapshot(1)
    path = episode / '镜头表.json'
    shots = json.loads(path.read_text(encoding='utf-8'))
    shots['镜头'][0]['帧'].append({'文件': 'K001_02.jpg', '时间': '00:00.8'})
    save(path, shots)
    (episode / '素材/frames/K001_02.jpg').write_bytes(b'fake-frame')
    update_result(episode, lambda d: d['镜头'][0]['动作'][0].update(依据帧=['K001_02.jpg']))
    data = build_rolecard_materials(episode, max_images=64)
    assert data['临时人物'][0]['候选帧'][0]['文件'] == 'K001_02.jpg'


@pytest.mark.parametrize('mismatch', ['人物', '片段', '出现镜头'])
def test_person_evidence_must_match_local_declaration(snapshot, mismatch):
    episode = snapshot(2, shared=False)
    if mismatch == '人物':
        update_result(episode, lambda d: d['镜头'][0]['画面人物'].clear())
    elif mismatch == '片段':
        update_result(episode, lambda d: d.update(片段='E01-P02'))
    else:
        path = episode / '临时人物.json'
        people = json.loads(path.read_text(encoding='utf-8'))
        people[0]['出现镜头'] = ['E01-K002']
        save(path, people)
    with pytest.raises(ValueError, match='人物依据与片段/本镜头声明不符'):
        build_rolecard_materials(episode, max_images=64)


def test_frame_from_other_shot_is_rejected(snapshot):
    episode = snapshot(2, shared=False)
    update_result(episode, lambda d: d['镜头'][0]['动作'][0].update(依据帧=['K002_01.jpg']))
    with pytest.raises(ValueError, match='非本镜头授权帧'):
        build_rolecard_materials(episode, max_images=64)


@pytest.mark.parametrize('broken', ['missing', 'traversal', 'symlink'])
def test_missing_or_escaped_frames_fail_explicitly(snapshot, tmp_path, broken):
    episode = snapshot(1)
    frame = episode / '素材/frames/K001_01.jpg'
    if broken == 'missing':
        frame.unlink()
        message = '图片不存在'
    else:
        outside = tmp_path / 'outside.jpg'
        outside.write_bytes(b'fake-frame')
        if broken == 'traversal':
            update_result(episode, lambda d: d['镜头'][0]['动作'][0].update(依据帧=[str(outside)]))
        else:
            frame.unlink()
            frame.symlink_to(outside)
        message = '路径越界'
    with pytest.raises(ValueError, match=message):
        build_rolecard_materials(episode, max_images=64)


def test_legal_frame_root_symlink_and_normalized_path_dedup(snapshot, tmp_path):
    episode = snapshot()
    root = episode / '素材/frames'
    real_root = tmp_path / 'source-frames'
    root.rename(real_root)
    root.symlink_to(real_root, target_is_directory=True)
    (real_root / 'K001_02.jpg').symlink_to(real_root / 'K001_01.jpg')
    path = episode / '镜头表.json'
    shots = json.loads(path.read_text(encoding='utf-8'))
    shots['镜头'][0]['帧'].append({'文件': 'K001_02.jpg', '时间': '00:00.2'})
    save(path, shots)
    update_result(episode, lambda d: d['镜头'][0]['动作'][1].update(依据帧=['K001_02.jpg']))
    data = build_rolecard_materials(episode, max_images=64)
    assert data['images'] == [str((real_root / 'K001_01.jpg').resolve())]
    assert all(person['id'] in data['labels'][0] for person in data['临时人物'])
    assert '00:00.2' in data['labels'][0]


def test_identity_only_and_visual_suitability_unconfirmed(snapshot):
    episode = snapshot()
    save(episode / '字幕记录.json', [
        {'起': '00:00.0', '止': '00:00.9', '文字': text, '类型': kind}
        for kind, text in [('人物字卡', '人物姓名'), ('地点字幕', '学校门口'),
                           ('台词字幕', '台词字幕隔离哨兵'), ('画面文字', '其他文字隔离哨兵')]])
    save(episode / '声音记录.json', [{'文本': '声音全文隔离哨兵'}])
    data = build_rolecard_materials(episode, max_images=64)
    encoded = json.dumps(data, ensure_ascii=False)
    assert all(s not in encoded for s in ('台词字幕隔离哨兵', '其他文字隔离哨兵',
                                         '声音全文隔离哨兵', '动作全文隔离哨兵', '映射隔离哨兵'))
    assert [s['类型'] for s in data['身份证据字幕']] == ['人物字卡', '地点字幕']
    assert data['统计']['候选关联覆盖'] == '完整'
    assert data['统计']['视觉覆盖'] == '未确认'
    assert '未读图' in data['视觉适用性']
    assert all('未确认' in p['视觉适用性'] for p in data['临时人物'])


def test_default_limit_uses_vision_configuration(snapshot, monkeypatch):
    monkeypatch.setattr('s2s.dispatch.rolecard_materials.get_vision', lambda: {'max_images_per_call': 1})
    data = build_rolecard_materials(snapshot(2, shared=False))
    assert data['统计']['图片上限'] == 1
    assert data['统计']['超过图片上限'] is True
    assert data['统计']['上限来源'] == 'vision.max_images_per_call'


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_limit(snapshot, limit):
    with pytest.raises(ValueError, match='正整数'):
        build_rolecard_materials(snapshot(1), max_images=limit)


def test_missing_action_results_fail_without_preparing(snapshot):
    episode = snapshot(1)
    (episode / '派发结果/动作识别/E01-P01.json').unlink()
    with pytest.raises(ValueError, match='缺少派发结果'):
        build_rolecard_materials(episode, max_images=64)


def test_cli_writes_new_output_only_and_never_inside_snapshot(snapshot, tmp_path):
    episode = snapshot()
    before = {str(p.relative_to(episode)): p.read_bytes() for p in episode.rglob('*') if p.is_file()}
    output = tmp_path / 'output/materials.json'
    command = [sys.executable, str(ROOT / 'tools/rolecard_materials.py'), str(episode)]
    result = subprocess.run([*command, str(output)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['统计']['候选关联人数'] == 2
    written = output.read_bytes()
    result = subprocess.run([*command, str(output)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 1 and '拒绝覆盖' in result.stderr
    assert output.read_bytes() == written
    result = subprocess.run([*command, str(episode / 'new.json')], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 1 and '快照之外' in result.stderr
    after = {str(p.relative_to(episode)): p.read_bytes() for p in episode.rglob('*') if p.is_file()}
    assert before == after
