import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

FIXTURE = Path(__file__).parent / 'fixtures/t5/完整黑板'


def run_tool(tool, *args):
    return subprocess.run([sys.executable, str(Path('tools') / f'{tool}.py'), *map(str, args)],
                          text=True, capture_output=True, timeout=10)


@pytest.mark.parametrize('tool,args', [
    ('validate', ['tests/fixtures/t5/完整黑板/E01/声音记录.json']),
    ('coverage', ['tests/fixtures/t5/完整黑板/E01']),
    ('lint_script', ['tests/fixtures/t5/PRD示例.md', '--角色卡', 'tests/fixtures/t5/完整黑板/角色卡.json', '--正式稿']),
])
def test_success_cli_json_and_exit_code(tool, args):
    process = run_tool(tool, *args)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)['通过']
    assert process.stderr == ''


@pytest.mark.parametrize('tool', ['validate', 'coverage', 'lint_script'])
def test_failure_cli_json_and_exit_code(tmp_path, tool):
    board = tmp_path / '黑板'
    shutil.copytree(FIXTURE, board)
    if tool == 'validate':
        target = board / 'E01/声音记录.json'
        target.write_text('[]坏JSON', encoding='utf-8')
    elif tool == 'coverage':
        target = board / 'E01'
        (target / '工作稿.md').write_text('林舟：改字。', encoding='utf-8')
    else:
        target = board / 'E01/工作稿.md'
        target.write_text('△ 他走到桌旁。', encoding='utf-8')
    process = run_tool(tool, target)
    assert process.returncode == 1
    result = json.loads(process.stdout)
    assert not result['通过'] and result['错误']
    assert process.stderr == ''


def test_board_cli_uses_parser_and_json_without_touching_shared_board(tmp_path):
    code = 'import runpy, sys; import s2s.board; from pathlib import Path; s2s.board.PROJECT_ROOT=Path(sys.argv[1]); sys.argv=["tools/board_init.py", "测试剧", "2"]; runpy.run_path("tools/board_init.py", run_name="__main__")'
    process = subprocess.run([sys.executable, '-c', code, str(tmp_path)], text=True, capture_output=True, timeout=10)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result['通过']
    assert Path(result['目录']) == tmp_path / '剧目/测试剧'
    assert (tmp_path / '剧目/测试剧/E02/状态.json').exists()
    assert process.stderr == ''
