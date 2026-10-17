import json
from pathlib import Path
import shutil

import pytest

from s2s.coverage import check_coverage, normalize_text

FIXTURE = Path(__file__).parent / 'fixtures/t5/完整黑板'


@pytest.fixture
def board(tmp_path):
    root = tmp_path / '黑板'
    shutil.copytree(FIXTURE, root)
    return root / 'E01'


def change_json(board, name, edit):
    file = board / f'{name}.json'
    data = json.loads(file.read_text(encoding='utf-8'))
    edit(data)
    file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


def problem_types(board):
    return {e['类型'] for e in check_coverage(board)['错误']}


def test_complete_board_and_punctuation_normalization(board):
    assert check_coverage(board)['通过']
    script = board / '工作稿.md'
    script.write_text(script.read_text().replace('今天必须说清楚。', '今天 必须，说清楚！'), encoding='utf-8')
    assert check_coverage(board)['通过']
    assert normalize_text('ＡＢＣ１２３，“你 好”！') == normalize_text('ABC123你 好')
    assert normalize_text('必须') != normalize_text('需要')
    assert normalize_text('Ａ＋Ｂ') == normalize_text('A+B')


def test_missing_sentence(board):
    change_json(board, '对照表', lambda data: data[3]['句'].clear())
    assert '漏句' in problem_types(board)
    error = next(e for e in check_coverage(board)['错误'] if e['类型'] == '漏句')
    assert error['位置']['句'] == 'E01-S001'


def test_repeated_sentence(board):
    change_json(board, '对照表', lambda data: data[3]['句'].append('E01-S001'))
    assert '重复句' in problem_types(board)


def test_sentence_order_uses_time_not_input_order(board):
    change_json(board, '声音记录', lambda data: data.reverse())
    change_json(board, '对照表', lambda data: data.reverse())
    assert check_coverage(board)['通过']
    def swap(data):
        rows = sorted([r for r in data if r['类型'] == '对白'], key=lambda r: r['行'])
        rows[0]['句'], rows[1]['句'] = rows[1]['句'], rows[0]['句']
    change_json(board, '对照表', swap)
    assert '句序错乱' in problem_types(board)


def test_key_action_missing_and_nonkey_ignored(board):
    change_json(board, '对照表', lambda data: data[2]['动作'].clear())
    assert '关键动作未覆盖' in problem_types(board)
    change_json(board, '动作记录', lambda data: data[0].update({'关键': False, '关键类别': None}))
    assert check_coverage(board)['通过']


def test_dialogue_modified_character(board):
    script = board / '工作稿.md'
    script.write_text(script.read_text().replace('今天必须说清楚。', '今天需要说清楚。'), encoding='utf-8')
    assert '台词不一致' in problem_types(board)
    error = next(e for e in check_coverage(board)['错误'] if e['类型'] == '台词不一致')
    assert error['位置']['行'] == 5


def test_concatenated_sentences_on_one_dialogue_line(board):
    change_json(board, '对照表', lambda data: (data[3]['句'].append('E01-S002'), data[5]['句'].clear(), data[5].update({'类型': '△'})))
    file = board / '工作稿.md'
    lines = file.read_text().splitlines()
    lines[4] += '合同的最后一页还没有签字。'
    lines[6] = '△ 沈宁坐在桌旁。'
    file.write_text('\n'.join(lines), encoding='utf-8')
    assert check_coverage(board)['通过']


@pytest.mark.parametrize('mode,expected', [
    ('unknown_sentence', '未知句'), ('unknown_action', '未知动作'), ('unmapped', '对白未映射'),
    ('duplicate_row', '重复行映射'), ('out_of_range', '行号越界'), ('wrong_type', '对白行类型'),
    ('non_dialogue', '非对白句映射'), ('duplicate_source', '编号重复'),
])
def test_reference_and_mapping_errors(board, mode, expected):
    def edit(data):
        if mode == 'unknown_sentence':
            data[3]['句'] = ['E01-S999']
        elif mode == 'unknown_action':
            data[3]['动作'].append('E01-A9999')
        elif mode == 'unmapped':
            data.pop(3)
        elif mode == 'duplicate_row':
            data.append(dict(data[3]))
        elif mode == 'out_of_range':
            data[3]['行'] = 99
        elif mode == 'wrong_type':
            data[3]['行'] = 4
        elif mode == 'non_dialogue':
            data[3]['类型'] = '△'
    if mode == 'duplicate_source':
        change_json(board, '声音记录', lambda data: data.append(dict(data[0])))
    else:
        change_json(board, '对照表', edit)
    assert expected in problem_types(board)


@pytest.mark.parametrize('file,content', [('声音记录.json', '{}'), ('动作记录.json', '[{}]'),
    ('对照表.json', '不是JSON'), ('声音记录.json', '[{"id": "E01-S001", "文本": "字", "起": "bad", "止": "00:01.0"}]')])
def test_invalid_input_is_json_error(board, file, content):
    (board / file).write_text(content, encoding='utf-8')
    assert '输入文件' in problem_types(board)


def test_missing_input_or_script(board):
    (board / '声音记录.json').unlink()
    assert '输入文件' in problem_types(board)
    assert not check_coverage(board, '没有稿.md')['通过']


def test_script_option(board):
    alternate = board / '另稿.md'
    shutil.copy(board / '工作稿.md', alternate)
    assert check_coverage(board, '另稿.md')['通过']
    assert check_coverage(board, alternate)['通过']


def test_invalid_utf8_script_is_json_error(board):
    (board / '工作稿.md').write_bytes(b'\xff')
    assert '输入文件' in problem_types(board)
