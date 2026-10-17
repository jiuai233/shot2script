"""按识图配置调用视觉模型，不自动拆分图片。"""

from pathlib import Path

from . import config, runners


def ask(images: list[str | Path], prompt: str, **override) -> str:
    settings = {**config.get_vision(), **override}
    limit = settings.pop("max_images_per_call", 12)
    if len(images) > limit:
        raise ValueError(f"图片数量 {len(images)} 超过 max_images_per_call={limit}，请明确划分调用范围")
    return runners.run(prompt, images=images, **settings)
