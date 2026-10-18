"""按识图配置调用视觉模型，不自动拆分图片。"""

from pathlib import Path

from . import config, runners

SYSTEM = "你是短剧画面识别助手，只根据画面回答。"


def ask(images: list[str | Path], prompt: str, **override) -> str:
    settings = {"system": SYSTEM, **config.get_vision(), **override}
    config.check_model(settings.get("model"))
    limit = settings.pop("max_images_per_call", 12)
    if len(images) > limit:
        raise ValueError(f"图片数量 {len(images)} 超过 max_images_per_call={limit}，请明确划分调用范围")
    return runners.run(prompt, images=images, **settings)
