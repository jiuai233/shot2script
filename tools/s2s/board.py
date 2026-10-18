"""建立黑板空结构，保留已存在的文件。"""

import argparse
import json
from pathlib import Path

from s2s.config import PROJECT_ROOT


EPISODE_LISTS = ('声音记录', '字幕记录', '动作记录', '临时人物', '归属表', '待核对', '对照表', '问题清单')


def initial_state(episode: int) -> dict:
    return {'集': episode, '状态': '进行中', '返修轮数': 0, '未决问题': [],
            '角色变更申请': [], '本集新增人物': [], '结尾衔接': '',
            '步骤': {str(i): False for i in range(1, 10)}}


def initialize_board(name: str, episodes: int, root: str | Path | None = None) -> dict:
    if not name.strip() or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('剧名须为单个目录名')
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError('集数须为正整数')
    base = Path(root) if root is not None else PROJECT_ROOT / '剧目'
    board = base / name
    board.mkdir(parents=True, exist_ok=True)
    (board / '合并').mkdir(exist_ok=True)
    files = {board / '角色卡.json': [], board / '角色变更记录.md': '',
             board / '全剧状态.json': {'状态': '进行中', '集': [
                 {'集': ep, '状态': '进行中'} for ep in range(1, episodes + 1)]}}
    for ep in range(1, episodes + 1):
        folder = board / f'E{ep:02d}'
        (folder / '素材' / 'frames').mkdir(parents=True, exist_ok=True)
        for kind in EPISODE_LISTS:
            files[folder / f'{kind}.json'] = []
        files[folder / '镜头表.json'] = {'集': ep, '视频': '', '时长': '00:00.0', '镜头': []}
        files[folder / '片段表.json'] = {'集': ep, '片段': []}
        files[folder / '本集人物.json'] = {'版本': 1, '输入版本': None, '人物': []}
        files[folder / '状态.json'] = initial_state(ep)
        for name in ('工作稿.md', '验收报告.md', '正式稿.md'):
            files[folder / name] = ''
    created, kept = [], []
    for path, data in files.items():
        try:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(data if isinstance(data, str) else
                             json.dumps(data, ensure_ascii=False, indent=2) + '\n')
            created.append(str(path))
        except FileExistsError:
            kept.append(str(path))
    return {'目录': str(board), '通过': True, '新建': created, '保留': kept}


def main() -> int:
    parser = argparse.ArgumentParser(description='初始化剧目黑板，不覆盖已有文件')
    parser.add_argument('剧名')
    parser.add_argument('集数', type=int)
    args = parser.parse_args()
    try:
        result = initialize_board(args.剧名, args.集数)
    except (ValueError, OSError) as exc:
        result = {'通过': False, '错误': str(exc)}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['通过'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
