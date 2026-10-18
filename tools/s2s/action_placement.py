"""成稿同次返回的结构化动作排布，只改排布字段，不改事件事实。"""

from copy import deepcopy


def apply_placements(actions, draft, *, required=False):
    rows = draft.get('动作排布')
    rendered = {ident for row in draft['对照表'] for ident in row['动作']}
    if rows is None:
        if required and rendered:
            raise ValueError('固定成稿必须为每个呈现动作返回独立排布决定')
        return deepcopy(actions)
    identifiers = [row['动作'] for row in rows]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != rendered:
        raise ValueError('动作排布必须恰好覆盖每个正文动作一次')
    decisions = {row['动作']: row['排布'] for row in rows}
    if not rendered <= {row['id'] for row in actions}:
        raise ValueError('动作排布引用未知事件')
    return [{**deepcopy(row), **({'排布': deepcopy(decisions[row['id']])} if row['id'] in decisions else {})}
            for row in actions]
