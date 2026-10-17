"""先在副本上校验，再备份并写入黑板。"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from ..validate import validate_file


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_board(episode, sounds, pending, operation):
    episode = Path(episode)
    episode.mkdir(parents=True, exist_ok=True)
    values = {"声音记录.json": sounds, "待核对.json": pending}
    with tempfile.TemporaryDirectory(prefix=".listen-", dir=episode) as directory:
        stage = Path(directory)
        for source in episode.iterdir():
            if source.is_file() and source.name not in values:
                (stage / source.name).symlink_to(source.resolve())
        roles = episode.parent / "角色卡.json"
        if not (stage / "角色卡.json").exists() and roles.is_file():
            (stage / "角色卡.json").symlink_to(roles.resolve())
        for name, value in values.items():
            write(stage / name, value)
        checks = [validate_file(stage / name) for name in values]
        if any(not check["通过"] for check in checks):
            raise ValueError("黑板校验失败：" + json.dumps(checks, ensure_ascii=False))
        backup = episode / "素材" / "回听" / "备份" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + operation)
        existing = [episode / name for name in values if (episode / name).exists()]
        if existing:
            backup.mkdir(parents=True)
            for source in existing:
                # 读取并备份实际目标，不覆盖上一轮备份。
                read(source)
                shutil.copy2(source, backup / source.name)
        for name in values:
            (stage / name).replace(episode / name)
    checks = [validate_file(episode / name) for name in values]
    if any(not check["通过"] for check in checks):
        raise ValueError("写回后校验失败：" + json.dumps(checks, ensure_ascii=False))
    return {"备份": str(backup) if existing else None, "校验": checks}
