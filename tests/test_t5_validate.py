import json
from pathlib import Path
import shutil

from jsonschema import Draft202012Validator
import pytest

from s2s.validate import SCHEMA_DIR, read_json, time_seconds, validate_file

FIXTURE = Path(__file__).parent / 'fixtures/t5/完整黑板'


@pytest.fixture
def board(tmp_path):
    root = tmp_path / '黑板'
    shutil.copytree(FIXTURE, root)
    return root / 'E01'


def change(board, name, edit):
    file = board / f'{name}.json'
    data = read_json(file)
    edit(data)
    file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return file


def issue():
    return {'id': 'E01-Q001', '来源': '验收', '位置': {'场景': '场景1-1', '行': 4, '镜头': 'E01-K001', '时间': '00:01.0'},
        '原文': '△ 他走到桌旁。', '类型': '人物指代与角色卡', '级别': '规范', '相关人物': ['C001'],
        '依据': 'K001_01.jpg', '修正要求': '写出姓名', '处理职责': '成稿', '状态': '待修正', '复核结果': None}


def test_all_schemas_are_valid_and_complete_fixture_passes(board):
    for schema in SCHEMA_DIR.glob('*.schema.json'):
        Draft202012Validator.check_schema(read_json(schema))
    for file in board.parent.rglob('*.json'):
        assert validate_file(file)['通过'], validate_file(file)
    for name in ('待核对', '问题清单'):
        file = change(board, name, lambda data: data.append(issue()))
        assert validate_file(file)['通过'], validate_file(file)
    file = change(board, '状态', lambda data: data['未决问题'].append('E01-Q001'))
    assert validate_file(file)['通过']


@pytest.mark.parametrize('name,field,value', [
    ('声音记录', '听辨', '猜测'), ('归属表', '形态', '旁白'), ('归属表', '确定度', '肯定'),
    ('对照表', '类型', '正文'), ('字幕记录', '类型', '字幕'), ('动作记录', '确定度', '猜测'), ('动作记录', '关键类别', 6),
])
def test_invalid_enums(board, name, field, value):
    file = change(board, name, lambda data: data[0].update({field: value}))
    assert not validate_file(file)['通过']
    assert any(field in e['位置'] for e in validate_file(file)['错误'])


@pytest.mark.parametrize('field,value', [('来源', '程序'), ('类型', '格式'), ('级别', '严重'), ('状态', '完成')])
def test_invalid_issue_enums(board, field, value):
    bad = issue()
    bad[field] = value
    file = change(board, '问题清单', lambda data: data.append(bad))
    assert not validate_file(file)['通过']


@pytest.mark.parametrize('name,field,value', [('状态', '状态', '完成'), ('状态', '返修轮数', 4),
    ('状态', '集', '1'), ('状态', '返修轮数', -1)])
def test_invalid_state(board, name, field, value):
    file = change(board, name, lambda data: data.update({field: value}))
    assert not validate_file(file)['通过']


@pytest.mark.parametrize('field,value', [('名称类型', '昵称'), ('状态', '确定')])
def test_invalid_role_enums(board, field, value):
    file = board.parent / '角色卡.json'
    data = read_json(file)
    data[0][field] = value
    file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    assert not validate_file(file)['通过']


@pytest.mark.parametrize('edit', [lambda d: d[0].pop('文本'), lambda d: d[0].update({'文本': 1}),
    lambda d: d[0].update({'额外': 1}), lambda d: d[0].update({'id': 'S001'}),
    lambda d: d[0].update({'id': 'E01-S1'}), lambda d: d[0].update({'声线': 'V1'})])
def test_required_types_extra_fields_and_id_format(board, edit):
    file = change(board, '声音记录', edit)
    assert not validate_file(file)['通过']


@pytest.mark.parametrize('name,edit', [
    ('声音记录', lambda d: d.append(dict(d[0]))),
    ('镜头表', lambda d: d['镜头'].append(dict(d['镜头'][0]))),
    ('片段表', lambda d: d['片段'].append(dict(d['片段'][0]))),
    ('归属表', lambda d: d.append(dict(d[0]))),
    ('对照表', lambda d: d.append(dict(d[0]))),
])
def test_duplicates(board, name, edit):
    file = change(board, name, edit)
    assert any('重复' in e['问题'] for e in validate_file(file)['错误'])


@pytest.mark.parametrize('edit', [lambda d: d[0].update({'id': 'E02-S001'}),
    lambda d: d[0].update({'声线': 'E02-V1'})])
def test_episode_ids_match_directory(board, edit):
    file = change(board, '声音记录', edit)
    assert any('集号' in e['问题'] for e in validate_file(file)['错误'])


def test_episode_top_level_and_scene_episode(board):
    file = change(board, '镜头表', lambda d: d.update({'集': 2}))
    assert any('与目录' in e['问题'] for e in validate_file(file)['错误'])
    file = change(board, '对照表', lambda d: d[0].update({'场景': '场景2-1'}))
    assert any('场景集号' in e['问题'] for e in validate_file(file)['错误'])


@pytest.mark.parametrize('value', ['0:01.0', '00:60.0', '00:01', '00:01.00', '-1:00.0', '１２:０１.０'])
def test_time_format(board, value):
    file = change(board, '声音记录', lambda d: d[0].update({'起': value}))
    assert not validate_file(file)['通过']
    with pytest.raises(ValueError):
        time_seconds(value)


def test_time_conversion_order_and_nested_frame(board):
    assert time_seconds('01:23.4') == 83.4
    file = change(board, '声音记录', lambda d: d[0].update({'起': '00:09.0'}))
    assert any(e['问题'] == '起晚于止' for e in validate_file(file)['错误'])
    file = change(board, '镜头表', lambda d: d['镜头'][0]['帧'][0].update({'时间': 'bad'}))
    assert not validate_file(file)['通过']


@pytest.mark.parametrize('name,edit', [
    ('动作记录', lambda d: d[0].update({'镜头': 'E01-K999'})),
    ('动作记录', lambda d: d[0].update({'执行者': 'C999'})),
    ('动作记录', lambda d: d[0].update({'对象': 'C999'})),
    ('归属表', lambda d: d[0].update({'句': 'E01-S999'})),
    ('归属表', lambda d: d[0].update({'说话人': 'C999'})),
    ('对照表', lambda d: d[0]['动作'].append('E01-A9999')),
    ('对照表', lambda d: d[0]['镜头'].append('E01-K999')),
    ('片段表', lambda d: d['片段'][0].update({'衔接后': 'E01-K999'})),
    ('片段表', lambda d: d['片段'][0]['镜头'].append('E01-K999')),
    ('状态', lambda d: d['未决问题'].append('E01-Q999')),
    ('状态', lambda d: d['本集新增人物'].append('C999')),
])
def test_unknown_references(board, name, edit):
    file = change(board, name, edit)
    assert any('不存在' in e['问题'] for e in validate_file(file)['错误'])


def test_role_relation_and_cross_episode_reference(board):
    card = board.parent / '角色卡.json'
    data = read_json(card)
    data[0]['关系'] = [{'对象': 'C002', '关系': '同事', '依据': '测试'}]
    card.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    assert validate_file(card)['通过']
    data[0]['关系'][0]['对象'] = 'C999'
    card.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    assert not validate_file(card)['通过']
    data[0]['关系'] = []
    data[0]['首次出场'] = 'E02-K001'
    card.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    other = board.parent / 'E02'
    other.mkdir()
    shots = read_json(board / '镜头表.json')
    shots['集'] = 2
    shots['镜头'][0]['id'] = 'E02-K001'
    (other / '镜头表.json').write_text(json.dumps(shots), encoding='utf-8')
    assert validate_file(card)['通过'], validate_file(card)


def test_temporary_person_checks_episode_and_segment(board):
    file = change(board, '动作记录', lambda d: d[0].update({'执行者': 'E01-P01-T1'}))
    assert validate_file(file)['通过']
    file = change(board, '动作记录', lambda d: d[0].update({'执行者': 'E01-P99-T1'}))
    assert not validate_file(file)['通过']


def test_subtitle_extra_fields_and_action_category(board):
    assert validate_file(board / '字幕记录.json')['通过']
    file = change(board, '字幕记录', lambda d: d[0].update({'置信度': 1.1}))
    assert not validate_file(file)['通过']
    file = change(board, '动作记录', lambda d: d[0].update({'关键': False, '关键类别': None}))
    assert validate_file(file)['通过']
    file = change(board, '动作记录', lambda d: d[0].update({'关键类别': 1}))
    assert not validate_file(file)['通过']
    file = change(board, '动作记录', lambda d: d[0].update({'关键': True, '关键类别': None}))
    assert not validate_file(file)['通过']


def test_infer_type_explicit_type_and_invalid_json(board):
    custom = board / '返回结果.json'
    shutil.copy(board / '声音记录.json', custom)
    assert not validate_file(custom)['通过']
    assert validate_file(custom, '声音记录')['通过']
    assert validate_file(custom, 'sound')['通过']
    assert not validate_file(custom, '../声音记录')['通过']
    custom.write_text('不是JSON', encoding='utf-8')
    assert not validate_file(custom, '声音记录')['通过']
    assert not validate_file(board / '不存在.json', '声音记录')['通过']


def test_missing_or_invalid_companion_is_reported(board):
    card = board.parent / '角色卡.json'
    card.write_text('不是JSON', encoding='utf-8')
    assert any('无法读取引用文件' in e['问题'] for e in validate_file(board / '动作记录.json')['错误'])
    (board / '镜头表.json').unlink()
    assert not validate_file(board / '动作记录.json')['通过']


def test_role_voice_reference_and_nested_episode(board):
    card = board.parent / '角色卡.json'
    data = read_json(card)
    data[0]['声线'][0]['标签'] = 'E01-V999'
    card.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    assert any('引用编号不存在' in e['问题'] for e in validate_file(card)['错误'])
    data[0]['声线'][0]['标签'] = 'E02-V1'
    card.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    assert any('编号集号' in e['问题'] for e in validate_file(card)['错误'])


def test_box_requires_normalized_coordinates(board):
    file = change(board, '字幕记录', lambda d: d[0]['框'].update({'x': -0.1}))
    assert not validate_file(file)['通过']
    file = change(board, '字幕记录', lambda d: d[0]['框'].update({'x': 0.2, 'width': 0.7}))
    assert validate_file(file)['通过']


def test_free_text_that_looks_like_id_is_not_a_reference(board):
    file = change(board, '声音记录', lambda d: d[0].update({'文本': 'E02-S001'}))
    assert validate_file(file)['通过']
    file = change(board, '动作记录', lambda d: d[0].update({'动作': 'E02-A9999', '说明': 'E02-K999'}))
    assert validate_file(file)['通过']


def test_episode_inference_ignores_free_text_and_field_order(tmp_path):
    file = tmp_path / '返回.json'
    record = read_json(FIXTURE / 'E01/声音记录.json')[0]
    record = {'文本': 'E99-S001', **{k: v for k, v in record.items() if k != '文本'}}
    file.write_text(json.dumps([record], ensure_ascii=False), encoding='utf-8')
    assert validate_file(file, '声音记录')['通过']


def test_scene_reference_must_exist_in_script(board):
    file = change(board, '对照表', lambda d: d[0].update({'场景': '场景1-99'}))
    assert any('引用场景不存在' in e['问题'] for e in validate_file(file)['错误'])
