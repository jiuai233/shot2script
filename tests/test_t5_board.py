import json
from pathlib import Path

import pytest

from s2s.board import initialize_board
from s2s.validate import validate_file


def test_initialize_directories_and_valid_empty_files(tmp_path):
    result = initialize_board('测试剧', 2, tmp_path)
    assert result['通过']
    board = tmp_path / '测试剧'
    assert (board / '合并').is_dir()
    assert (board / '角色变更记录.md').read_text() == ''
    for ep in (1, 2):
        folder = board / f'E{ep:02d}'
        assert (folder / '素材/frames').is_dir()
        assert {p.name for p in folder.iterdir()} == {
            '素材', '镜头表.json', '片段表.json', '临时人物.json', '声音记录.json', '字幕记录.json', '动作记录.json',
            '归属表.json', '待核对.json', '工作稿.md', '对照表.json', '问题清单.json',
            '验收报告.md', '正式稿.md', '状态.json'}
        assert json.loads((folder / '状态.json').read_text())['状态'] == '进行中'
    for file in board.rglob('*.json'):
        assert validate_file(file)['通过'], validate_file(file)


def test_existing_files_are_not_overwritten(tmp_path):
    first = initialize_board('测试剧', 1, tmp_path)
    role = tmp_path / '测试剧/角色卡.json'
    role.write_text('保留原内容', encoding='utf-8')
    second = initialize_board('测试剧', 1, tmp_path)
    assert role.read_text() == '保留原内容'
    assert second['新建'] == []
    assert set(first['新建']) == set(second['保留'])


@pytest.mark.parametrize('name,episodes', [('', 1), ('.', 1), ('..', 1), ('../逃逸', 1),
    ('a/b', 1), ('a\\b', 1), ('测试剧', 0), ('测试剧', -1), ('测试剧', True)])
def test_invalid_arguments(tmp_path, name, episodes):
    with pytest.raises(ValueError):
        initialize_board(name, episodes, tmp_path)
    assert not list(tmp_path.iterdir())
