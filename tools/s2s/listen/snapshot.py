"""回听摘要只绑定句号、文本及起止；完整快照供列出其他字段变化。"""

from copy import deepcopy

FIELDS = ("id", "文本", "起", "止")


def summary(sounds):
    return [{k: r[k] for k in FIELDS} for r in sounds]


def field_changes(before, after):
    result = []
    for old, new in zip(before, after):
        changed = {k: {"原值": deepcopy(old.get(k)), "现值": deepcopy(new.get(k))}
                   for k in old.keys() | new.keys() if k not in FIELDS and old.get(k) != new.get(k)}
        if changed:
            result.append({"句号": new["id"], "字段": changed})
    return result
