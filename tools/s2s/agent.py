"""按角色继承后的配置调用模型。"""

from pathlib import Path

from . import config, runners


def run(
    role: str,
    prompt: str,
    attachments: list[str | Path] | None = None,
    images: list[str | Path] | None = None,
    **override,
) -> str:
    settings = {**config.get_agent(role), **override}
    config.check_model(settings.get("model"))
    limit = settings.pop("max_images_per_call", None)
    if limit is not None and len(images or []) > limit:
        raise ValueError(f"角色 {role} 的图片数量 {len(images)} 超过 max_images_per_call={limit}")
    return runners.run(prompt, images=images, files=attachments, **settings)
