import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

from jsonschema import Draft202012Validator
import pytest

from s2s.validate import SCHEMA_DIR, read_json, validate_file

FIXTURES = Path(__file__).parent / 'fixtures/t5'
FILES = {'口型记录': '口型记录.json', '声纹聚类': '声纹聚类.json',
         '声纹候选': '声纹候选.json', '人声段': '素材/人声段.json',
         '裁定迁移': '素材/回听/裁定迁移.json'}
ALIASES = {'口型记录': 'lipsync', '声纹聚类': 'diarization',
           '声纹候选': 'voiceprint_candidates', '人声段': 'speech_regions',
           '裁定迁移': 'decision_migrations'}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return path


@pytest.fixture
def board(tmp_path):
    root = tmp_path / '黑板'
    shutil.copytree(FIXTURES / '完整黑板', root)
    episode = root / 'E01'
    for kind, name in FILES.items():
        save(episode / name, read_json(FIXTURES / f'T5c/{kind}.json'))
    frame = episode / '素材/frames/K002_01.jpg'
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b'frame')
    return episode


def change(board, kind, path, value):
    target = board / FILES[kind]
    data = read_json(target)
    child = data
    for key in path[:-1]:
        child = child[key]
    child[path[-1]] = value
    return save(target, data)


def assert_bad(file, message=None, kind=None):
    result = validate_file(file, kind)
    assert not result['通过'], result
    if message:
        assert any(message in e['问题'] for e in result['错误']), result
    return result


@pytest.mark.parametrize('kind', FILES)
def test_schema_and_all_required_top_fields(kind):
    schema = read_json(SCHEMA_DIR / f'{kind}.schema.json')
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    data = read_json(FIXTURES / f'T5c/{kind}.json')
    assert validator.is_valid(data)
    target_schema = schema['items'] if isinstance(data, list) else schema
    for field in target_schema['required']:
        bad = copy.deepcopy(data)
        target = bad[0] if isinstance(bad, list) else bad
        del target[field]
        assert not validator.is_valid(bad), field
        target[field] = {}
        assert not validator.is_valid(bad), field
    assert not validator.is_valid({} if isinstance(data, list) else [])


@pytest.mark.parametrize('kind', FILES)
def test_auto_type_explicit_type_alias_and_cli(board, kind):
    file = board / FILES[kind]
    for name in (None, kind, ALIASES[kind]):
        assert validate_file(file, name)['通过'], validate_file(file, name)
    alternate = file.with_name('返回结果.json')
    shutil.copy(file, alternate)
    assert validate_file(alternate, kind)['通过'], validate_file(alternate, kind)
    for target, args in ((file, []), (alternate, ['--type', kind])):
        result = subprocess.run([sys.executable, 'tools/validate.py', str(target), *args],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr + result.stdout
        assert json.loads(result.stdout)['通过']
        assert result.stderr == ''
    save(file, {})
    result = subprocess.run([sys.executable, 'tools/validate.py', str(file)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert not json.loads(result.stdout)['通过']
    assert result.stderr == ''


LIP_ROW = ('句', 0)
TRACK = (*LIP_ROW, '轨迹', 0)


@pytest.mark.parametrize('path,value,message', [
    ((*LIP_ROW, '句'), 'E01-S999', '引用编号不存在'),
    ((*LIP_ROW, '句'), 'S001', '格式不符'),
    ((*LIP_ROW, '文本'), '改写对白', '文本与同目录声音记录不一致'),
    ((*LIP_ROW, '起'), '00:05.1', '起与同目录声音记录不一致'),
    ((*LIP_ROW, '止'), '00:07.1', '止与同目录声音记录不一致'),
    ((*LIP_ROW, '起'), '00:05.033', '格式不符'),
    ((*LIP_ROW, '镜头'), ['E01-K999'], '引用编号不存在'),
    ((*LIP_ROW, '结论'), '画外', '取值不在允许范围'),
    ((*TRACK, '状态'), '画内', '取值不在允许范围'),
    ((*TRACK, 'id'), 'E01-K999-F1', '引用编号不存在'),
    ((*TRACK, 'id'), 'E01-K001-F1', '未列在本句镜头'),
    ((*TRACK, 'id'), 'E01-K002-F0', '格式不符'),
    ((*TRACK, '依据帧'), ['素材/frames/missing.jpg'], '依据帧文件不存在'),
    ((*TRACK, '框', 'x'), -.01, '数值不得小于'),
    ((*TRACK, '说话分数'), 1.1, '数值不得大于'),
    ((*TRACK, '曲线', 0, '时间'), '00:60.033', '格式不符'),
    ((*TRACK, '曲线', 0, '时间'), '00:05.03', '格式不符'),
    ((*TRACK, '曲线', 0, '有效'), 1, '字段类型错误'),
    ((*TRACK, '模型曲线', 0, '时间'), '00:60.040', '格式不符'),
    ((*TRACK, '模型曲线', 0, '说话概率'), 1.1, '数值不得大于'),
    ((*LIP_ROW, '复核'), {'状态': '已复核'}, '取值不符合'),
    (('集',), 2, '与目录'),
    ((*LIP_ROW, '声线'), 'E02-V1', '编号集号'),
])
def test_lipsync_rejects_invalid_values(board, path, value, message):
    assert_bad(change(board, '口型记录', path, value), message)


@pytest.mark.parametrize('path,value', [
    ((*LIP_ROW, '结论'), '未开口'), ((*LIP_ROW, '结论'), '看不到'), ((*LIP_ROW, '结论'), '存疑'),
    ((*TRACK, '状态'), '未开口'), ((*TRACK, '状态'), '看不到'), ((*TRACK, '状态'), '存疑'),
    ((*TRACK, '轨迹组'), None), ((*TRACK, '轨迹组'), '侧脸候选甲'),
    ((*TRACK, '曲线', 0, '时间'), '00:05.0'),
    ((*TRACK, '曲线', 0, '张合比'), None),
    ((*TRACK, '嘴部区域'), None), ((*TRACK, '嘴部区域'), [.1, .1, .2, .2]),
    ((*TRACK, '模型分数'), None),
    (('版本',), 'fused'),
    ((*LIP_ROW, '复核'), {'状态': '未调用', '原因': '有效帧不足'}),
    ((*LIP_ROW, '复核'), {'状态': '失败', '原因': '识图失败'}),
    ((*LIP_ROW, '复核'), {'状态': '已复核', '依据帧': ['素材/frames/K002_01.jpg'],
                        '结果': '看不到', '画面结论': '看不到', '与代码一致': False}),
])
def test_lipsync_allowed_extensions_and_states(board, path, value):
    file = change(board, '口型记录', path, value)
    assert validate_file(file)['通过'], validate_file(file)


def test_lipsync_pixel_and_model_extensions(board):
    data = read_json(board / FILES['口型记录'])
    track = data['句'][0]['轨迹'][0]
    track.update({'检测来源': ['YuNet嘴部像素'], '规则分数': .3, '模型分数': .5})
    track['曲线'][0].update({'张合比': None, '嘴部像素变化': .02})
    track['评分依据'].update({'测量方式': '像素变化', '规则状态': '存疑', '模型有效帧': 1, '评分冲突': False})
    data.update({'复用模型输出': '模型输出.json', '复用侧脸记录': 'side.json',
                 '复核统计': {'耗时秒': 1}, '统计': {'复用前流程耗时秒': 2, '本次增量耗时秒': 1}})
    file = save(board / FILES['口型记录'], data)
    assert validate_file(file)['通过'], validate_file(file)


@pytest.mark.parametrize('mode', ['absolute', 'project_relative', 'review'])
def test_lipsync_evidence_path_resolution(board, mode, monkeypatch):
    frame = board / '素材/frames/K002_01.jpg'
    value = str(frame)
    if mode == 'project_relative':
        monkeypatch.chdir(board.parent)
        value = 'E01/素材/frames/K002_01.jpg'
    path = (*TRACK, '依据帧')
    if mode == 'review':
        file = change(board, '口型记录', (*LIP_ROW, '复核'),
                      {'状态': '已复核', '依据帧': [value], '结果': '存疑'})
    else:
        file = change(board, '口型记录', path, [value])
    assert validate_file(file)['通过'], validate_file(file)
    frame.unlink()
    assert_bad(file, '依据帧文件不存在')


@pytest.mark.parametrize('path', [('句',), (*LIP_ROW, '轨迹')])
def test_lipsync_duplicate_sentence_or_track(board, path):
    file = board / FILES['口型记录']
    data = read_json(file)
    items = data
    for key in path:
        items = items[key]
    items.append(copy.deepcopy(items[0]))
    assert_bad(save(file, data), '重复')


@pytest.mark.parametrize('bad', [None, [], {'句': [None]}, {'句': [{'句': []}]},
                               {'句': [{'句': 'E01-S001', '轨迹': [None]}]}])
def test_lipsync_bad_local_structure_reports_errors_without_crashing(board, bad):
    assert_bad(save(board / FILES['口型记录'], bad))


def test_lipsync_schema_error_does_not_hide_evidence_or_snapshot_errors(board):
    file = change(board, '口型记录', (*TRACK, '框', 'x'), -.1)
    data = read_json(file)
    data['句'][0]['文本'] = '过期文本'
    data['句'][0]['轨迹'][0]['依据帧'] = ['缺失.jpg']
    result = assert_bad(save(file, data))
    for message in ('数值不得小于', '声音记录不一致', '依据帧文件不存在'):
        assert any(message in e['问题'] for e in result['错误'])


@pytest.mark.parametrize('mode', ['missing', 'invalid', 'parent_only'])
def test_lipsync_needs_same_directory_sound(board, mode):
    sound = board / '声音记录.json'
    if mode == 'invalid':
        sound.write_text('坏 JSON', encoding='utf-8')
    else:
        if mode == 'parent_only':
            shutil.copy(sound, board.parent)
        sound.unlink()
    assert_bad(board / FILES['口型记录'], '无法读取引用文件')


@pytest.mark.parametrize('path,value,message', [
    (('句列表', 0, '句'), 'E01-S999', '引用编号不存在'),
    (('句列表', 0, '声线'), 'V1', '取值不符合'),
    (('句列表', 0, '声线'), 'E02-V1', '编号集号'),
    (('句列表', 0, '声线'), 'E01-V2', '未在类别中声明'),
    (('类别', 0, '句数'), 2, '类别句数与句列表不一致'),
    (('类别', 0, '聚类句数'), 0, '聚类句数与句列表不一致'),
    (('类别', 0, '聚类句数'), 2, '不得超过类别句数'),
    (('类别', 0, '代表句', 0, '句'), 'E01-S002', '代表句不属于'),
    (('类别', 0, '代表句', 0, '句'), 'E01-S999', '引用编号不存在'),
    (('矩阵声线顺序',), ['E01-V2'], '全部类别声线'),
    (('类间相似度矩阵',), [[1, .2]], '维度'),
    (('句列表', 0, '合并依据'), ['剧情'], '取值不在允许范围'),
    (('句列表', 0, '置信度'), 1.1, '取值不符合'),
    (('句列表', 0, '相似度'), -1.1, '取值不符合'),
    (('约束', 'cannot-link'), [['E01-S001', 'E01-S999']], '引用编号不存在'),
    (('约束', '冲突'), [{'类型': '口型输入不匹配', '句': ['E01-S999']}], '引用编号不存在'),
])
def test_diarization_rules(board, path, value, message):
    assert_bad(change(board, '声纹聚类', path, value), message)


@pytest.mark.parametrize('field', ['句列表', '类别'])
def test_diarization_duplicate_ids(board, field):
    file = board / FILES['声纹聚类']
    data = read_json(file)
    data[field].append(copy.deepcopy(data[field][0]))
    assert_bad(save(file, data), '重复')


def test_diarization_at_most_three_representatives(board):
    file = board / FILES['声纹聚类']
    data = read_json(file)
    row = data['类别'][0]['代表句'][0]
    data['类别'][0]['代表句'] = [row] * 3
    assert validate_file(save(file, data))['通过']
    data['类别'][0]['代表句'].append(row)
    assert_bad(save(file, data), '数组条目最多')


def test_diarization_pending_sentence_preserves_sound_voice(board):
    file = board / FILES['声纹聚类']
    data = read_json(file)
    data['句列表'][0].update({'声线': '待定', '待定': True, '置信度': None, '相似度': None, '合并依据': []})
    data.update({'类别': [], '矩阵声线顺序': [], '类间相似度矩阵': []})
    assert validate_file(save(file, data))['通过'], validate_file(file)
    assert read_json(board / '声音记录.json')[0]['声线'] == 'E01-V1'
    data['句列表'][0]['待定'] = False
    assert_bad(save(file, data), '格式不符')


@pytest.mark.parametrize('path,value,message', [
    (('仅供参考',), False, 'JSON Schema'),
    (('声线', 0, '候选', 0, '角色'), 'C999', '引用编号不存在'),
    (('声线', 0, '候选', 0, '角色'), '林舟', '格式不符'),
    (('声线', 0, '候选', 0, '相似度'), 1.1, '数值不得大于'),
    (('声线', 0, '声线'), 'V1', '格式不符'),
    (('声线', 0, '声线'), 'E02-V1', '编号集号'),
    (('集',), 'E02', '与目录'),
])
def test_voiceprint_candidates_rules(board, path, value, message):
    assert_bad(change(board, '声纹候选', path, value), message)


def test_candidates_missing_registry_only_checks_role_format(board):
    (board.parent / '角色卡.json').unlink()
    file = change(board, '声纹候选', ('声线', 0, '候选', 0, '角色'), 'C999')
    assert validate_file(file)['通过'], validate_file(file)
    assert_bad(change(board, '声纹候选', ('声线', 0, '候选', 0, '角色'), 'C9'), '格式不符')


def test_candidates_uses_series_card_even_if_episode_has_local_card(board):
    save(board / '角色卡.json', [])
    file = board / FILES['声纹候选']
    assert validate_file(file)['通过'], validate_file(file)
    assert_bad(change(board, '声纹候选', ('声线', 0, '候选', 0, '角色'), 'C999'), '引用编号不存在')


def test_candidates_present_but_unreadable_registry_is_error(board):
    (board.parent / '角色卡.json').write_text('坏 JSON', encoding='utf-8')
    assert_bad(board / FILES['声纹候选'], '无法读取引用文件')


def test_candidates_empty_and_three_candidates_and_duplicates(board):
    file = change(board, '声纹候选', ('声线', 0, '候选'), [])
    assert validate_file(file)['通过']
    candidates = [{'角色': f'C00{i}', '相似度': .5} for i in (1, 2, 3)]
    (board.parent / '角色卡.json').unlink()
    assert validate_file(change(board, '声纹候选', ('声线', 0, '候选'), candidates))['通过']
    assert_bad(change(board, '声纹候选', ('声线', 0, '候选'), candidates + [candidates[0]]), '数组条目最多')
    data = read_json(file)
    data['声线'][0]['候选'] = []
    data['声线'].append(copy.deepcopy(data['声线'][0]))
    assert_bad(save(file, data), '重复')


def test_candidates_standalone_episode_comes_from_string_field(tmp_path):
    file = save(tmp_path / '声纹候选.json', read_json(FIXTURES / 'T5c/声纹候选.json'))
    assert validate_file(file)['通过']
    data = read_json(file)
    data['声线'][0]['声线'] = 'E02-V1'
    assert_bad(save(file, data), '编号集号')


@pytest.mark.parametrize('regions,message', [
    ([{'起': '00:00.1', '止': .5}], '字段类型错误'),
    ([{'起': True, '止': .5}], '字段类型错误'),
    ([{'起': -.1, '止': .5}], '数值不得小于'),
    ([{'起': .6, '止': .5}], '起晚于止'),
    ([{'起': 2, '止': 3}, {'起': 0, '止': 1}], '未按起点排序'),
    ([{'起': 0, '止': 2}, {'起': 1, '止': 3}], '人声段重叠'),
])
def test_speech_regions_rules(board, regions, message):
    assert_bad(save(board / FILES['人声段'], regions), message)


@pytest.mark.parametrize('regions', [[], [{'起': 0, '止': 1}],
                                     [{'起': 0, '止': 1}, {'起': 1, '止': 2}],
                                     [{'起': .06, '止': .49}, {'起': 1, '止': 2}]])
def test_speech_regions_allow_numeric_seconds_sorted_and_touching(board, regions):
    file = save(board / FILES['人声段'], regions)
    assert validate_file(file)['通过'], validate_file(file)


@pytest.mark.parametrize('path,value,message', [
    ((0, '新句号'), 'E01-S999', '引用编号不存在'),
    ((0, '新句号'), 'E02-S001', '编号集号'),
    ((0, '新句号'), 'S001', '取值不符合'),
    ((0, '旧句号'), ['S999'], '格式不符'),
    ((0, '状态'), '完成', '取值不在允许范围'),
    ((0, '原裁定'), {}, '字段类型错误'),
])
def test_decision_migration_rules(board, path, value, message):
    assert_bad(change(board, '裁定迁移', path, value), message)


@pytest.mark.parametrize('status', ['已沿用', '待确认', '未匹配'])
def test_decision_migration_all_statuses_and_historical_ids(board, status):
    file = change(board, '裁定迁移', (0, '状态'), status)
    assert validate_file(file)['通过'], validate_file(file)
    assert read_json(file)[0]['旧句号'] == ['E01-S999']


def test_decision_migration_uses_episode_sound_not_subdirectory(board):
    file = board / FILES['裁定迁移']
    save(file.parent / '声音记录.json', [])
    assert validate_file(file)['通过'], validate_file(file)
    (board / '声音记录.json').unlink()
    assert_bad(file, '无法读取引用文件')


def test_decision_migration_duplicate_new_ids_but_multiple_unmatched_allowed(board):
    file = board / FILES['裁定迁移']
    data = read_json(file)
    data.append(dict(data[1], 旧句号=['E01-S997']))
    assert validate_file(save(file, data))['通过']
    data.append(copy.deepcopy(data[0]))
    assert_bad(save(file, data), '重复')


@pytest.mark.parametrize('kind,path,definition', [
    ('口型记录', LIP_ROW, '句'), ('口型记录', TRACK, '轨迹'),
    ('口型记录', (*TRACK, '框'), '框'),
    ('口型记录', (*TRACK, '评分依据'), '评分依据'),
    ('口型记录', (*TRACK, '曲线', 0), '曲线项'),
    ('口型记录', (*TRACK, '模型曲线', 0), '模型曲线项'),
    ('声纹聚类', ('句列表', 0), '句'), ('声纹聚类', ('类别', 0), '类别'),
])
def test_nested_required_fields_have_positive_and_negative_examples(kind, path, definition):
    schema = read_json(SCHEMA_DIR / f'{kind}.schema.json')
    validator = Draft202012Validator(schema)
    data = read_json(FIXTURES / f'T5c/{kind}.json')
    assert validator.is_valid(data)
    for field in schema['$defs'][definition]['required']:
        bad = copy.deepcopy(data)
        record = bad
        for key in path:
            record = record[key]
        del record[field]
        assert not validator.is_valid(bad), (definition, field)
