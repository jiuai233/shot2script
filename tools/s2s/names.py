"""角色显示名的确定性检查；描述名中的泛称不等于非法名称。"""

import json
from pathlib import Path


GENERIC_NAMES = set(json.loads((Path(__file__).with_name('schemas') / 'lint_words.json').read_text(encoding='utf-8'))['泛称'])
VAGUE_NAMES = {'他', '她', '他们', '她们', '它们', '对方', '前者', '后者', '另一人', '另一个人', '其中一人'}


def invalid_name(name: str) -> bool:
    return not name.strip() or name != name.strip() or name in GENERIC_NAMES | VAGUE_NAMES


def name_errors(roles: list[dict]) -> list[tuple[int, str]]:
    errors, seen = [], set()
    for i, role in enumerate(roles):
        name = role['名称']
        if invalid_name(name):
            errors.append((i, '名称为空、含首尾空白或属于非法泛称/指代：' + name))
        if name in seen:
            errors.append((i, '不同角色不能共用显示名称：' + name))
        seen.add(name)
    return errors
