import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

from jsonschema import Draft202012Validator
import pytest

from s2s.board import initialize_board
from s2s.validate import SCHEMA_DIR, read_json, validate_file

FIXTURES = Path(__file__).parent / 'fixtures/t5'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return path


@pytest.fixture
def board(tmp_path):
    root = tmp_path / '黑板'
    shutil.copytree(FIXTURES / '完整黑板', root)
    episode = root / 'E01'
    shutil.copy(FIXTURES / 'T5b/角色变更申请.json', episode)
    dense = episode / '素材/frames_dense/00-11.0-00-12.0/帧.json'
    save(dense, read_json(FIXTURES / 'T5b/加密帧索引.json'))
    state = read_json(episode / '状态.json')
    state['角色变更申请'] = read_json(episode / '角色变更申请.json')
    save(episode / '状态.json', state)
    return episode


def change(board, name, edit):
    path = board / f'{name}.json'
    data = read_json(path)
    edit(data)
    return save(path, data)


def assert_bad(path, message=None, kind=None):
    result = validate_file(path, kind)
    assert not result['通过'], result
    if message:
        assert any(message in error['问题'] for error in result['错误']), result
    return result


NEW_RECORDS = {
    '临时人物': FIXTURES / '完整黑板/E01/临时人物.json',
    '角色变更申请': FIXTURES / 'T5b/角色变更申请.json',
    '加密帧索引': FIXTURES / 'T5b/加密帧索引.json',
}


@pytest.mark.parametrize('kind', NEW_RECORDS)
def test_new_schemas_valid_required_and_typed(kind):
    schema = read_json(SCHEMA_DIR / f'{kind}.schema.json')
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    data = read_json(NEW_RECORDS[kind])
    assert validator.is_valid(data)
    record = data[0] if isinstance(data, list) else data
    for field in record:
        bad = copy.deepcopy(data)
        target = bad[0] if isinstance(bad, list) else bad
        del target[field]
        assert not validator.is_valid(bad), field
        target[field] = {}
        assert not validator.is_valid(bad), field
    bad = copy.deepcopy(data)
    (bad[0] if isinstance(bad, list) else bad)['额外字段'] = True
    assert not validator.is_valid(bad)
    assert not validator.is_valid({} if isinstance(data, list) else [])


def test_all_fixture_files_and_embedded_request_pass(board):
    for path in board.parent.rglob('*.json'):
        assert validate_file(path)['通过'], validate_file(path)
    assert read_json(SCHEMA_DIR / '状态.schema.json')['properties']['角色变更申请']['items'] == (
        read_json(SCHEMA_DIR / '角色变更申请.schema.json')['items'])


@pytest.mark.parametrize('name,field', [
    ('动作记录', '执行者'), ('动作记录', '对象'), ('归属表', '说话人'),
])
def test_temporary_person_must_be_declared(board, name, field):
    path = change(board, name, lambda d: d[0].update({field: 'E01-P01-T1'}))
    assert validate_file(path)['通过'], validate_file(path)
    change(board, name, lambda d: d[0].update({field: 'E01-P01-T2'}))
    assert_bad(path, 'E01-P01-T2（临时人物）')


@pytest.mark.parametrize('mode', ['missing', 'invalid_json', 'parent_only'])
def test_temporary_person_registry_is_required_in_same_directory(board, mode):
    path = change(board, '归属表', lambda d: d[0].update({'说话人': 'E01-P01-T1'}))
    registry = board / '临时人物.json'
    if mode == 'invalid_json':
        registry.write_text('坏 JSON', encoding='utf-8')
    else:
        if mode == 'parent_only':
            shutil.copy(registry, board.parent)
        registry.unlink()
    assert_bad(path, '无法读取引用文件')


@pytest.mark.parametrize('field,value,message', [
    ('映射', 'C999', '不存在'), ('映射', '', '编号格式'),
    ('片段', 'E01-P99', '不存在'), ('出现镜头', ['E01-K999'], '不存在'),
    ('id', 'E01-P01-T0', '格式不符'), ('id', 'E02-P01-T1', '集号'),
    ('片段', 'E02-P01', '集号'), ('出现镜头', ['E02-K001'], '集号'),
])
def test_temporary_person_invalid_values(board, field, value, message):
    path = change(board, '临时人物', lambda d: d[0].update({field: value}))
    assert_bad(path, message)


@pytest.mark.parametrize('mapping', [None, 'C001'])
def test_temporary_mapping_may_be_unresolved_or_existing_role(board, mapping):
    path = change(board, '临时人物', lambda d: d[0].update({'映射': mapping}))
    assert validate_file(path)['通过'], validate_file(path)


def test_temporary_id_matches_declared_segment(board):
    segments = read_json(board / '片段表.json')
    second = dict(segments['片段'][0], id='E01-P02')
    segments['片段'].append(second)
    save(board / '片段表.json', segments)
    path = change(board, '临时人物', lambda d: d[0].update({'片段': 'E01-P02'}))
    assert_bad(path, '编号与所属片段不一致')


def test_temporary_person_duplicate_ids(board):
    path = change(board, '临时人物', lambda d: d.append(dict(d[0])))
    assert_bad(path, '重复')


@pytest.mark.parametrize('name', ['角色变更申请', '状态'])
@pytest.mark.parametrize('field,value,message', [
    ('涉及角色', ['C999'], '不存在'), ('涉及角色', ['林舟'], '格式不符'),
    ('id', 'E01-R1', '格式不符'), ('id', 'E02-R01', '集号'),
    ('类型', '新增', '取值不在允许范围'), ('状态', '完成', '取值不在允许范围'),
    ('提出集', 2, '提出集'), ('提出集', 0, '不得小于'),
    ('涉及集', [0], '不得小于'), ('涉及集', ['2'], '类型错误'),
    ('依据', '画面证据', '类型错误'), ('审核结论', 1, '符合任一'),
])
def test_role_changes_invalid_in_standalone_and_state(board, name, field, value, message):
    def edit(data):
        records = data['角色变更申请'] if name == '状态' else data
        records[0][field] = value
    assert_bad(change(board, name, edit), message)


@pytest.mark.parametrize('name', ['角色变更申请', '状态'])
def test_role_change_missing_fields_and_duplicate_ids(board, name):
    original = read_json(board / f'{name}.json')
    records = original['角色变更申请'] if name == '状态' else original
    for field in records[0]:
        bad = copy.deepcopy(original)
        (bad['角色变更申请'] if name == '状态' else bad)[0].pop(field)
        assert_bad(save(board / f'{name}.json', bad), '必填字段')
    records.append(copy.deepcopy(records[0]))
    assert_bad(save(board / f'{name}.json', original), '重复')


@pytest.mark.parametrize('request_type', ['合并', '拆分', '改名'])
@pytest.mark.parametrize('status', ['待审核', '同意', '驳回', '待确认'])
def test_role_change_allowed_enums(board, request_type, status):
    def edit(data):
        data['角色变更申请'][0].update({'类型': request_type, '状态': status, '审核结论': '画面依据已核对'})
    path = change(board, '状态', edit)
    assert validate_file(path)['通过'], validate_file(path)


def test_role_change_infers_episode_without_directory(board, tmp_path):
    requests = read_json(board / '角色变更申请.json')
    target = tmp_path / '独立结果/角色变更申请.json'
    save(target.parent / '角色卡.json', read_json(board.parent / '角色卡.json'))
    save(target, requests)
    assert validate_file(target)['通过'], validate_file(target)
    requests[0]['提出集'] = 2
    assert_bad(save(target, requests), '提出集')


@pytest.mark.parametrize('edit,message', [
    (lambda d: d['镜头'][0].update({'起': '00:00.1'}), '首镜头'),
    (lambda d: d.update({'时长': '00:20.1'}), '末镜头'),
    (lambda d: d['镜头'][1].update({'起': '00:05.1'}), '首尾相接'),
    (lambda d: d['镜头'][1].update({'起': '00:04.9'}), '首尾相接'),
    (lambda d: d['镜头'][0]['帧'].reverse(), '时间升序'),
    (lambda d: d['镜头'][0]['帧'][0].update({'文件': '../K001_01.jpg'}), '格式不符'),
    (lambda d: d['镜头'][0]['帧'][0].update({'文件': 'K001_1.jpg'}), '格式不符'),
    (lambda d: d['镜头'][0]['帧'][0].update({'文件': 'K002_01.jpg'}), '与镜头编号不一致'),
    (lambda d: d['镜头'][0]['帧'][0].update({'时间': '00:09.0'}), '超出起止范围'),
    (lambda d: d.update({'镜头': []}), '不能为空'),
])
def test_shot_timeline_and_frame_rules(board, edit, message):
    assert validate_file(board / '镜头表.json')['通过']
    assert_bad(change(board, '镜头表', edit), message)


@pytest.fixture
def split_segments(board):
    original = read_json(board / '片段表.json')['片段'][0]
    first = dict(original, 镜头=original['镜头'][:2], 止='00:10.0', 衔接后='E01-K003')
    second = dict(original, id='E01-P02', 镜头=original['镜头'][2:], 起='00:10.0', 衔接前='E01-K002')
    path = save(board / '片段表.json', {'集': 1, '片段': [first, second]})
    assert validate_file(path)['通过'], validate_file(path)
    return board


@pytest.mark.parametrize('edit,message', [
    (lambda d: d['片段'][0]['镜头'].pop(), '恰好覆盖'),
    (lambda d: d['片段'][1]['镜头'].insert(0, 'E01-K002'), '恰好覆盖'),
    (lambda d: d['片段'][0]['镜头'].reverse(), '恰好覆盖'),
    (lambda d: d['片段'].reverse(), '恰好覆盖'),
    (lambda d: d.update({'片段': []}), '恰好覆盖'),
    (lambda d: d['片段'][0].update({'镜头': []}), '数组条目至少'),
    (lambda d: d['片段'][0].update({'起': '00:00.1'}), '首镜头起点'),
    (lambda d: d['片段'][1].update({'止': '00:19.9'}), '末镜头终点'),
    (lambda d: d['片段'][0].update({'衔接前': 'E01-K001'}), '衔接前必须'),
    (lambda d: d['片段'][1].update({'衔接后': 'E01-K004'}), '衔接后必须'),
    (lambda d: d['片段'][1].update({'衔接前': None}), '衔接前必须'),
    (lambda d: d['片段'][0].update({'衔接后': None}), '衔接后必须'),
])
def test_segment_exact_coverage_times_and_connections(split_segments, edit, message):
    assert_bad(change(split_segments, '片段表', edit), message)


@pytest.mark.parametrize('bad_shots', [None, [], {'镜头': [None]}, {'镜头': None}, {'镜头': 1}])
def test_segments_need_readable_structurally_valid_shot_table(board, bad_shots):
    shots = board / '镜头表.json'
    shots.unlink()
    assert_bad(board / '片段表.json', '无法读取引用文件')
    save(shots, bad_shots)
    assert_bad(board / '片段表.json', '结构不符合')


@pytest.mark.parametrize('edit,message', [
    (lambda d: d.update({'帧率': 0}), None),
    (lambda d: d.update({'帧率': -1}), None),
    (lambda d: d.update({'起': '00:13.0'}), '起晚于止'),
    (lambda d: d.update({'起': '0:11.0'}), '格式不符'),
    (lambda d: d['帧'].reverse(), '时间升序'),
    (lambda d: d['帧'][0].update({'文件': '../t00-11.0.jpg'}), '格式不符'),
    (lambda d: d['帧'][0].update({'文件': 't00-11.1.jpg'}), '与时间不一致'),
    (lambda d: d['帧'][0].update({'时间': '00:10.9'}), '超出起止范围'),
    (lambda d: d['帧'][1].update({'时间': '00:12.1'}), '超出起止范围'),
])
def test_dense_frames_rules(board, edit, message):
    path = board / '素材/frames_dense/00-11.0-00-12.0/帧.json'
    assert validate_file(path)['通过']
    data = read_json(path)
    edit(data)
    assert_bad(save(path, data), message)


@pytest.mark.parametrize('kind', ['加密帧索引', 'dense_frames', '帧'])
def test_dense_frames_type_aliases(board, kind):
    path = board / '素材/frames_dense/00-11.0-00-12.0/帧.json'
    assert validate_file(path, kind)['通过']


@pytest.mark.parametrize('status', ['进行中', '验收中', '返修中', '验收通过', '待人工裁定'])
def test_series_state_matches_episode_status_enum(board, status):
    path = save(board.parent / '全剧状态.json', {'状态': status, '集': [{'集': 1, '状态': status}]})
    assert validate_file(path)['通过']
    for field in ['root', 'episode']:
        data = read_json(path)
        (data if field == 'root' else data['集'][0])['状态'] = '完成'
        assert_bad(save(path, data), '取值不在允许范围')
        save(path, {'状态': status, '集': [{'集': 1, '状态': status}]})


def test_series_state_required_fields_types_and_duplicates(board):
    path = board.parent / '全剧状态.json'
    original = read_json(path)
    for bad in [{}, {'状态': '进行中'}, {'状态': '进行中', '集': [1]},
                {'状态': '进行中', '集': [{'集': 0, '状态': '进行中'}]}]:
        assert_bad(save(path, bad))
    original['集'].append(dict(original['集'][0]))
    assert_bad(save(path, original), '重复')


def test_initialized_board_includes_new_files_and_preserves_existing(tmp_path):
    initialize_board('新剧', 2, tmp_path)
    root = tmp_path / '新剧'
    for ep in (1, 2):
        folder = root / f'E{ep:02d}'
        assert read_json(folder / '片段表.json') == {'集': ep, '片段': []}
        assert read_json(folder / '临时人物.json') == []
    for path in root.rglob('*.json'):
        assert validate_file(path)['通过'], validate_file(path)
    path = root / 'E01/临时人物.json'
    path.write_text('保留原内容', encoding='utf-8')
    assert initialize_board('新剧', 2, tmp_path)['新建'] == []
    assert path.read_text(encoding='utf-8') == '保留原内容'


def test_dense_frame_cli_infers_type_and_returns_failure_code(board):
    path = board / '素材/frames_dense/00-11.0-00-12.0/帧.json'
    command = [sys.executable, 'tools/validate.py', str(path)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)['通过']
    data = read_json(path)
    data['帧率'] = 0
    save(path, data)
    process = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert process.returncode == 1
    assert not json.loads(process.stdout)['通过']
    assert process.stderr == ''
