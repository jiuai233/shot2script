"""动作排布由确认记录约束，不按字数、关键词或剧情猜测。"""

import copy
import json
from pathlib import Path
import shutil

import pytest

from s2s.quality import check_script
from test_production_script import accept_review

FIXTURE = Path(__file__).parent / 'fixtures/t5/完整黑板'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


@pytest.fixture
def board(tmp_path):
    root = tmp_path / '剧名'
    shutil.copytree(FIXTURE, root)
    episode = root / 'E01'
    actions = json.loads((episode / '动作记录.json').read_text())
    for action, sentence in zip(actions, (None, 'E01-S001', None, 'E01-S003')):
        action['排布'] = {'位置': '对白括号' if sentence else '独立△', '句': sentence,
                          '依据': '人物和时间已核对，独立项须完整展开，括号项直接伴随本句。'}
    save(episode / '动作记录.json', actions)
    accept_review(episode)
    return episode


def change(board, name, edit):
    path = board / (name + '.json')
    value = json.loads(path.read_text())
    edit(value)
    save(path, value)


def rules(board, formal=False):
    return {e['检查项'] for e in check_script(board, formal=formal)['错误']}


def split_short_action(board):
    path = board / '工作稿.md'
    text = path.read_text().replace('林舟（右掌拍在桌面上，抬高声音）：', '△ 林舟右掌拍在桌面上。\n林舟（抬高声音）：')
    path.write_text(text)
    mapping = json.loads((board / '对照表.json').read_text())
    for row in mapping:
        if row['行'] > 5:
            row['行'] += 1
        elif row['行'] == 5:
            row['行'] = 6
            row['动作'] = []
    mapping.insert(3, {'行': 5, '类型': '△', '场景': '场景1-1', '句': [],
                       '镜头': ['E01-K002'], '动作': ['E01-A0002']})
    save(board / '对照表.json', mapping)


def test_confirmed_placement_fixture_passes(board):
    assert check_script(board, formal=True)['通过'], check_script(board, formal=True)


def test_short_action_must_follow_reviewed_parenthesis_placement(board):
    split_short_action(board)
    assert '动作排布不一致' in rules(board)


def test_independent_listener_action_cannot_be_merged(board):
    change(board, '动作记录', lambda rows: rows[1]['排布'].update(位置='独立△', 句=None))
    assert '动作排布不一致' in rules(board)


def test_parenthesis_action_must_match_the_exact_target_sentence(board):
    change(board, '动作记录', lambda rows: rows[1]['排布'].update(句='E01-S003'))
    assert '动作关联句不一致' in rules(board)


def test_parenthesis_cannot_merge_action_that_finished_before_speech(board):
    change(board, '动作记录', lambda rows: rows[1].update(起='00:04.0', 止='00:04.5', 镜头='E01-K001',
                                                        依据帧=['K001_01.jpg']))
    change(board, '对照表', lambda rows: rows[3]['镜头'].append('E01-K001'))
    assert '动作对白时序' in rules(board)


def test_parenthesis_action_cannot_extend_after_speech(board):
    change(board, '动作记录', lambda rows: rows[1].update(止='00:08.0'))
    assert '动作对白时序' in rules(board)


def test_missing_placement_does_not_auto_guess_a_short_action(board):
    change(board, '动作记录', lambda rows: rows[1].pop('排布'))
    assert check_script(board)['通过']
    assert '动作排布未确认' in rules(board, formal=True)


def test_same_event_with_different_ids_is_not_written_twice(board):
    def duplicate(rows):
        row = copy.deepcopy(rows[1])
        row['id'] = 'E01-A0005'
        rows.append(row)
    change(board, '动作记录', duplicate)
    change(board, '对照表', lambda rows: rows[3]['动作'].append('E01-A0005'))
    assert '重复动作事件' in rules(board)


def test_action_id_in_mapping_without_action_content_is_not_coverage(board):
    path = board / '工作稿.md'
    path.write_text(path.read_text().replace('右掌拍在桌面上，抬高声音', '抬高声音'))
    assert '动作呈现待核实' in rules(board, formal=True)


def test_speaker_action_cannot_include_another_actor(board):
    change(board, '动作记录', lambda rows: rows[1].update(执行者='C002'))
    assert '括号动作错归属' in rules(board)


def test_real_repeated_actions_at_different_times_are_not_duplicates(board):
    def repeat(rows):
        row = copy.deepcopy(rows[1])
        row.update(id='E01-A0005', 起='00:18.0', 止='00:18.5', 镜头='E01-K004', 执行者='C002',
                   依据帧=['K004_01.jpg'], 排布={'位置': '对白括号', '句': 'E01-S003', '依据': '另一时刻再次拍桌'})
        rows.append(row)
    change(board, '动作记录', repeat)
    change(board, '对照表', lambda rows: rows[-1]['动作'].append('E01-A0005'))
    path = board / '工作稿.md'
    path.write_text(path.read_text().replace('沈宁（接过信封，低头查看）', '沈宁（接过信封，低头查看，右掌拍在桌面上）'))
    assert '重复动作事件' not in rules(board)


def test_reviewed_action_already_started_and_continues_during_speech_is_allowed(board):
    change(board, '声音记录', lambda rows: rows[0].update(起='00:05.2'))
    assert '动作对白时序' not in rules(board)


def test_reviewed_short_rewording_is_allowed_but_not_silent_rewriting(board):
    path = board / '工作稿.md'
    path.write_text(path.read_text().replace('右掌拍在桌面上', '右手拍桌'))
    assert '动作呈现待核实' in rules(board)
    change(board, '动作记录', lambda rows: rows[1]['排布'].update(表述='右手拍桌'))
    assert '动作呈现待核实' not in rules(board)


@pytest.mark.parametrize('field,value', [('位置', '随便放'), ('句', None), ('依据', ' '), ('表述', '')])
def test_placement_requires_valid_position_sentence_and_evidence(board, field, value):
    change(board, '动作记录', lambda rows: rows[1]['排布'].update({field: value}))
    assert '动作记录' in rules(board)
