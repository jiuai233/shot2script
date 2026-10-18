"""用法：.venv/bin/python tools/rolecard_materials.py 本集快照 新输出.json。"""

import argparse
import json
from pathlib import Path

from s2s.dispatch.materials import dumps
from s2s.dispatch.rolecard_materials import build_rolecard_materials


def main(argv=None):
    parser = argparse.ArgumentParser(description='只读生成候选代表帧清单与容量统计；不调用模型、不预处理、不改黑板')
    parser.add_argument('episode', type=Path, metavar='本集快照')
    parser.add_argument('output', type=Path, metavar='新输出.json')
    parser.add_argument('--max-images', type=int, help='显式图片上限，默认复用 vision.max_images_per_call')
    args = parser.parse_args(argv)
    try:
        episode = args.episode.resolve()
        output = args.output.resolve()
        if output.is_relative_to(episode):
            raise ValueError('输出必须位于本集快照之外')
        if args.output.exists() or args.output.is_symlink():
            raise ValueError('输出已存在，拒绝覆盖：' + str(args.output))
        data = build_rolecard_materials(episode, max_images=args.max_images)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            stream.write(dumps(data) + '\n')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'候选材料核验失败：{exc}\n')
    print(json.dumps({'输出': str(output), '统计': data['统计']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
