import json
from pathlib import Path

import pytest

from s2s.lint_script import lint_file, lint_text, parse_dialogue

FIXTURES = Path(__file__).parent / 'fixtures/t5'
ROLES = [{'名称': '林舟'}, {'名称': '沈宁'}, {'名称': '穿红衣服的小男孩'}]
EXAMPLE = (FIXTURES / 'PRD示例.md').read_text(encoding='utf-8')


def rules(text, **kwargs):
    return {e['检查项'] for e in lint_text(text, **kwargs)['错误']}


def test_prd_example_is_exact_fixture_and_passes():
    prd = Path('docs/视频逐镜头画声转写PRD.md').read_text(encoding='utf-8')
    section = prd.split('### 2.3 表达示例', 1)[1].split('### 2.4', 1)[0]
    extracted = '\n'.join(line[2:].rstrip() for line in section.splitlines() if line.startswith('> ')) + '\n'
    assert EXAMPLE == extracted
    assert lint_text(EXAMPLE)['通过']
    assert lint_text(EXAMPLE, ROLES, formal=True)['通过']


def test_replace_lin_with_pronoun_reports_exact_line():
    mutated = EXAMPLE.replace('△ 林舟走到', '△ 他走到')
    result = lint_text(mutated, ROLES)
    assert result['错误'] == [{'行': 4, '原文': mutated.splitlines()[3],
                             '检查项': '人物代词', '命中词': '他'}]


@pytest.mark.parametrize('pronoun', ['他', '她', '他们', '她们'])
@pytest.mark.parametrize('template', ['△ {}走到会议桌旁。', '林舟（看向{}）：他对方。'])
def test_pronouns_are_detected_only_in_description(pronoun, template):
    assert '人物代词' in rules(template.format(pronoun), roles=ROLES)
    assert '人物代词' not in rules(template.format('沈宁'), roles=ROLES)


def test_human_and_nonhuman_tamen():
    assert '人物代词' in rules('△ 林舟和沈宁站在门口，它们转身。', roles=ROLES)
    assert '人物代词' not in rules('△ 两只猫趴在门口，它们抬头。', roles=ROLES)


@pytest.mark.parametrize('name', ['男人', '女人', '男子', '女子', '小男孩', '小女孩', '老人', '老头', '老太太', '年轻人'])
def test_generic_names(name):
    assert '人物泛称' in rules(f'△ {name}递出信封。', roles=ROLES)
    assert '人物泛称' not in rules('△ 穿红衣服的小男孩递出信封。', roles=ROLES)


@pytest.mark.parametrize('word', ['对方', '前者', '后者', '另一人', '另一个人', '其中一人'])
def test_vague_references(word):
    assert '含糊指代' in rules(f'林舟（把信封递给{word}）：他。', roles=ROLES)
    assert '含糊指代' not in rules('林舟（把信封递给沈宁）：对方他。', roles=ROLES)


@pytest.mark.parametrize('shape', ['画外', '心声', '电话', '视频', '广播'])
def test_offscreen_actions_and_voice_only(shape):
    assert '画外形态动作' in rules(f'林舟（{shape}，抬头）：今天说清楚。', roles=ROLES)
    assert '画外形态动作' not in rules(f'林舟（{shape}，声音发颤）：他拍桌，对方站起。', roles=ROLES)
    assert '画外形态动作' not in rules('林舟（右掌拍在桌面上）：今天说清楚。', roles=ROLES)


def test_dialogue_words_subtitle_and_fixed_names_are_exempt():
    assert lint_text('林舟：他和对方去找那个男人。', ROLES)['通过']
    assert lint_text('△ 画面字幕：他与对方，男人们的会议。', ROLES)['通过']
    assert lint_text('△ 林舟拿起吉他，与其他演员站在门口。', ROLES)['通过']
    assert lint_text('△ 林舟拿起《老人与海》和《他与她》。', ROLES)['通过']
    assert lint_text('△ 她的名字出现在门上。', [{'名称': '她的名字'}])['通过']


@pytest.mark.parametrize('bad,good', [
    ('场景1-1:会议室，日，内', '场景1-1：会议室，日，内'),
    ('场景1-1：会议室，日，室内', '场景1-1：会议室，日，外'),
    ('场景1-1：会议室，内', '场景1-1：会议室，日，内（闪回）'),
])
def test_scene_format(bad, good):
    assert '场景行格式' in rules(f'{bad}\n人物：林舟')
    assert lint_text(f'{good}\n人物：林舟')['通过']


@pytest.mark.parametrize('serial', [2, 3])
def test_scene_numbers_start_at_one_and_stay_continuous(serial):
    assert '场景编号连续' in rules(f'第1集\n场景1-{serial}：会议室，日，内\n人物：林舟')
    assert '场景编号连续' in rules('场景1-1：会议室，日，内\n人物：林舟\n场景1-3：门口，日，外\n人物：林舟')
    assert lint_text('第1集\n场景1-1：会议室，日，内\n人物：林舟\n场景1-2：门口，日，外\n人物：林舟')['通过']


def test_scene_episode_and_reset_between_episodes():
    assert '场景集号' in rules('第1集\n场景2-1：会议室，日，内\n人物：林舟')
    assert lint_text('第1集\n场景1-1：会议室，日，内\n人物：林舟\n第2集\n场景2-1：会议室，日，内\n人物：林舟')['通过']


@pytest.mark.parametrize('following', ['', '\n人物：林舟', '林舟：台词。'])
def test_immediate_cast_line_required(following):
    assert '场景后人物行' in rules('场景1-1：会议室，日，内\n' + following)
    assert lint_text('场景1-1：会议室，日，内\n人物：林舟')['通过']


@pytest.mark.parametrize('line', ['人物：林舟、路人', '路人：今天说清楚。'])
def test_role_names_require_registered_name(line):
    assert '角色卡名称' in rules(line, roles=ROLES)
    assert '角色卡名称' not in rules(line.replace('路人', '沈宁'), roles=ROLES)
    assert '角色卡名称' not in rules(line)


@pytest.mark.parametrize('token', ['【待核对用字】', '【说话人待核对】', '00:12.4', 'E01-S001', 'E01-K001', 'E01-A0001', 'E01-Q001', 'E01-P01'])
def test_formal_residue(token):
    assert '正式稿残留' in rules(f'林舟：{token}', formal=True)
    assert '正式稿残留' not in rules(f'林舟：{token}', formal=False)
    assert '正式稿残留' not in rules('林舟：今天必须说清楚。', formal=True)


def test_custom_word_file_and_errors(tmp_path):
    script = tmp_path / '稿.md'
    script.write_text('林舟（电话，眨眼）：台词。\n△ 访客站在门口。', encoding='utf-8')
    word_file = tmp_path / '词表.json'
    word_file.write_text(json.dumps({'泛称': ['访客'], '动作词': ['眨眼']}, ensure_ascii=False), encoding='utf-8')
    result = lint_file(script, word_file=word_file)
    assert {e['检查项'] for e in result['错误']} == {'画外形态动作', '人物泛称'}
    word_file.write_text('{}')
    assert lint_file(script, word_file=word_file)['错误'][0]['检查项'] == '输入文件'
    assert not lint_file(tmp_path / '不存在.md')['通过']


def test_parenthesis_and_dialogue_are_separated():
    assert parse_dialogue('林舟（画外，声音发颤）：他向对方说：好。') == ('林舟', '画外，声音发颤', '他向对方说：好。')
    assert parse_dialogue('林舟(画外):好。') == ('林舟', '画外', '好。')
    assert parse_dialogue('△ 林舟拍桌。') is None
    assert '行类型' in rules('任务已完成')
