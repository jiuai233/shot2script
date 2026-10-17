import copy
import json
from pathlib import Path
import shutil

import pytest
from s2s.evaluate import aggregate, evaluate, main, validate_annotation

FIXTURES = Path(__file__).parent / 'fixtures' / 't7'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


@pytest.fixture
def board(tmp_path):
    shutil.copytree(FIXTURES, tmp_path / 'data')
    base = tmp_path / 'data'
    return base / '标注.json', base / '完美剧' / 'E01'


def test_perfect_board(board):
    annotation, episode = board
    result = evaluate(annotation, episode)
    assert result['指标']['台词逐字一致率']['比率'] == 1
    assert not result['待复核']
    for key, value in result['指标'].items():
        assert not value['错误']
        if key != '台词逐字一致率':
            assert value['比率'] == 0


def inject_errors(episode):
    refs = load(episode / '对照表.json')
    lines = (episode / '正式稿.md').read_text().splitlines()
    remove_lines = {r['行'] for r in refs if r['句'] in [['E01-S005'], ['E01-S006']] or r['动作'] == ['E01-A0001']}
    old_to_new, new_lines = {}, []
    for i, line in enumerate(lines, 1):
        if i in remove_lines:
            continue
        old_to_new[i] = len(new_lines) + 1
        if line.startswith('林舟：今天'):
            line = line.replace('林舟：', '沈宁：', 1)
        if '林舟举起文件' in line:
            line = line.replace('林舟举起文件', '沈宁举起文件')
        new_lines.append(line)
    refs = [{**r, '行': old_to_new[r['行']]} for r in refs if r['行'] not in remove_lines]
    actions = [r for r in load(episode / '动作记录.json') if r['id'] != 'E01-A0001']
    extra = {**actions[0], 'id': 'E01-A0099', '起': '00:25.0', '止': '00:25.5',
             '执行者': 'C001', '动作': '摔碎玻璃杯', '对象': '玻璃杯', '物件': '玻璃杯'}
    actions.append(extra)
    new_lines.append('△ 林舟摔碎玻璃杯。')
    refs.append({'行': len(new_lines), '类型': '△', '场景': '场景1-1', '句': [], '镜头': ['E01-K099'], '动作': [extra['id']]})
    save(episode / '动作记录.json', actions)
    save(episode / '对照表.json', refs)
    (episode / '正式稿.md').write_text('\n'.join(new_lines) + '\n', encoding='utf-8')


def test_known_error_counts(board):
    annotation, episode = board
    inject_errors(episode)
    result = evaluate(annotation, episode)
    metrics = result['指标']
    assert (metrics['台词逐字一致率']['分子'], metrics['台词逐字一致率']['分母']) == (4, 6)
    assert len(metrics['台词逐字一致率']['错误']) == 2
    assert (metrics['说话人错归率']['分子'], metrics['说话人错归率']['分母']) == (1, 4)
    assert metrics['关键动作漏写率']['分子'] == 1
    assert metrics['无据动作率']['分子'] == 1
    assert (metrics['人物错配率']['分子'], metrics['人物错配率']['分母']) == (1, 3)
    for value in metrics.values():
        for error in value['错误']:
            assert {'位置', '标注', '正文'} <= error.keys()
    assert metrics['无据动作率']['错误'][0]['需要人工复核'] is True


def test_body_not_asr_or_attribution_is_scored(board):
    annotation, episode = board
    path = episode / '正式稿.md'
    path.write_text(path.read_text().replace('今天必须说清楚。', '今天必须讲清楚。'))
    result = evaluate(annotation, episode)
    assert result['指标']['台词逐字一致率']['分子'] == 5
    assert result['指标']['说话人错归率']['分子'] == 0


def test_normalization(board):
    annotation, episode = board
    data = load(annotation)
    data['台词'][0]['文本'] += 'Ａ１！'
    save(annotation, data)
    path = episode / '正式稿.md'
    path.write_text(path.read_text().replace('今天必须说清楚。', '今天 必须说清楚a1！'.replace('a1', 'A1')))
    assert evaluate(annotation, episode)['指标']['台词逐字一致率']['分子'] == 6


def test_merged_dialogue_and_split_dialogue(board):
    annotation, episode = board
    data = load(annotation)
    first = data['台词'].pop(0)
    data['台词'][0] = {**first, '止': data['台词'][0]['止'], '文本': first['文本'] + data['台词'][0]['文本']}
    # 同一说话人的一句拆成两行，统计文字应保持完全一致。
    data['台词'][0]['说话人'] = 'H01'
    save(annotation, data)
    path = episode / '正式稿.md'
    path.write_text(path.read_text().replace('沈宁：原件在这里。', '林舟：原件在这里。'))
    metrics = evaluate(annotation, episode)['指标']
    assert metrics['台词逐字一致率']['分子'] == 5
    assert metrics['说话人错归率']['分子'] == 0


def test_repeated_lines_match_by_time(board):
    annotation, episode = board
    data = load(annotation)
    data['台词'][1]['文本'] = data['台词'][0]['文本']
    save(annotation, data)
    path = episode / '正式稿.md'
    path.write_text(path.read_text().replace('原件在这里。', '今天必须说清楚。'))
    result = evaluate(annotation, episode)
    assert result['指标']['台词逐字一致率']['分子'] == 6
    assert result['指标']['说话人错归率']['分子'] == 0


def test_time_tolerance_is_parameter(board):
    annotation, episode = board
    data = load(annotation)
    data['台词'][0].update({'起': '00:03.1', '止': '00:03.2'})
    save(annotation, data)
    assert evaluate(annotation, episode, time_tolerance=.2)['指标']['台词逐字一致率']['分子'] == 6
    assert evaluate(annotation, episode, time_tolerance=0)['指标']['台词逐字一致率']['分子'] == 5


def test_record_alone_can_cover_key_action(board):
    annotation, episode = board
    path = episode / '正式稿.md'
    path.write_text(path.read_text().replace('△ 林舟右掌拍在桌面上。', ''))
    result = evaluate(annotation, episode)
    assert result['指标']['关键动作漏写率']['分子'] == 0
    assert any(row['类型'] == '动作映射没有正文动作' for row in result['待复核'])


def test_body_alone_can_cover_key_action(board):
    annotation, episode = board
    # 记录文本错误，但正文动作和对应时间仍能独立覆盖。
    actions = load(episode / '动作记录.json')
    actions[0]['动作'] = '推开木门'
    save(episode / '动作记录.json', actions)
    result = evaluate(annotation, episode)
    assert result['指标']['关键动作漏写率']['分子'] == 0


def test_target_mismatch(board):
    annotation, episode = board
    data = load(annotation)
    data['关键动作'][-1]['对象'] = 'H02'
    save(annotation, data)
    metrics = evaluate(annotation, episode)['指标']
    assert metrics['人物错配率']['分子'] == 1


def test_unmapped_body_actions_are_not_silently_ignored(board):
    annotation, episode = board
    path = episode / '正式稿.md'
    path.write_text(path.read_text() + '△ 林舟摔碎玻璃杯。\n林舟（拍桌）：结束了。\n')
    result = evaluate(annotation, episode)
    assert result['指标']['无据动作率']['分子'] == 2
    assert result['待复核']


def test_unfinished_annotation_is_rejected(board):
    annotation, episode = board
    data = load(annotation)
    data['校对完成'] = False
    save(annotation, data)
    with pytest.raises(ValueError, match='尚未人工校对'):
        evaluate(annotation, episode)


@pytest.mark.parametrize('mutation', [
    lambda data: data['台词'][0].update({'起': '错误'}),
    lambda data: data['台词'][0].update({'止': '00:00.0'}),
    lambda data: data['台词'][0].update({'说话人': 'H99'}),
    lambda data: data['台词'][0].update({'说话人': None}),
    lambda data: data['关键动作'][0].update({'类别': 6}),
    lambda data: data['人物'][0].update({'出现': ['00:04.0-00:03.0']}),
    lambda data: data['台词'].append(copy.deepcopy(data['台词'][0])),
    lambda data: data['台词'][0].update({'待核对': '听不清'}),
])
def test_annotation_schema_and_references(mutation):
    data = load(FIXTURES / '标注.json')
    mutation(data)
    with pytest.raises(ValueError):
        validate_annotation(data)


def test_empty_denominators(board):
    annotation, episode = board
    data = load(annotation)
    data['台词'] = []; data['关键动作'] = []; data['人物'] = []
    save(annotation, data)
    for name in ('声音记录', '动作记录', '归属表', '对照表'):
        save(episode / f'{name}.json', [])
    (episode / '正式稿.md').write_text('第1集\n')
    assert all(x['比率'] is None for x in evaluate(annotation, episode)['指标'].values())


def test_missing_mapping_is_not_treated_as_speaker_error(board):
    annotation, episode = board
    save(annotation.with_name('人物映射.json'), {})
    with pytest.raises(ValueError, match='人物映射'):
        evaluate(annotation, episode)


def test_aggregate_and_cli(board, capsys):
    annotation, episode = board
    assert main([str(annotation), str(episode)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['合计']['台词逐字一致率']['分子'] == 6
    doubled = aggregate([output['各集'][0], output['各集'][0]])
    assert doubled['台词逐字一致率']['分母'] == 12
    assert main([str(annotation), str(episode), '--time-tolerance', '-1']) == 1
    assert '错误' in json.loads(capsys.readouterr().out)


def test_duplicate_references_rejected(board):
    annotation, episode = board
    refs = load(episode / '对照表.json')
    refs.append(refs[0])
    save(episode / '对照表.json', refs)
    with pytest.raises(ValueError, match='重复'):
        evaluate(annotation, episode)


def test_two_people_actions_in_one_body_line(board):
    annotation, episode = board
    refs = load(episode / '对照表.json')
    action_refs = [row for row in refs if row['动作']]
    first, second = action_refs[:2]
    path = episode / '正式稿.md'
    lines = path.read_text().splitlines()
    lines[first['行'] - 1] += lines[second['行'] - 1].removeprefix('△')
    lines[second['行'] - 1] = ''
    first['动作'] += second['动作']
    refs.remove(second)
    save(episode / '对照表.json', refs)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    metrics = evaluate(annotation, episode)['指标']
    assert metrics['人物错配率']['分子'] == 0
    assert metrics['人物错配率']['分母'] == 4
    assert metrics['无据动作率']['分子'] == 0


def test_object_changed_in_body_is_not_hidden_by_record(board):
    annotation, episode = board
    path = episode / '正式稿.md'
    path.write_text(path.read_text().replace('沈宁把信封递给林舟', '沈宁把信封递给沈宁'))
    assert evaluate(annotation, episode)['指标']['人物错配率']['分子'] == 1


def test_multiepisode_cli(board, capsys):
    annotation, episode = board
    second = annotation.parent / 'second'
    shutil.copytree(annotation.parent, second, ignore=shutil.ignore_patterns('second'))
    data = load(second / '标注.json')
    data['集'] = 2
    save(second / '标注.json', data)
    assert main([str(annotation), str(episode), '--追加', str(second / '标注.json'), str(second / '完美剧' / 'E01')]) == 0
    output = json.loads(capsys.readouterr().out)
    assert [row['集'] for row in output['各集']] == [1, 2]
    assert output['合计']['台词逐字一致率']['分子'] == 12
    assert output['合计']['人物错配率']['分母'] == 8


def test_two_sentences_in_one_body_line_keep_sentence_denominator(board):
    annotation, episode = board
    refs = load(episode / '对照表.json')
    first, second = [row for row in refs if row['句']][:2]
    path = episode / '正式稿.md'
    lines = path.read_text().splitlines()
    lines[first['行'] - 1] += '原件在这里。'
    lines[second['行'] - 1] = ''
    first['句'] += second['句']
    refs.remove(second)
    save(episode / '对照表.json', refs)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    metrics = evaluate(annotation, episode)['指标']
    assert metrics['台词逐字一致率']['分子'] == 6
    assert metrics['说话人错归率']['分母'] == 6
    assert metrics['说话人错归率']['分子'] == 1
