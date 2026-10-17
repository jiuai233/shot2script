"""按角色继承后的配置调用模型。"""

from pathlib import Path

from . import config, runners

ROLE_SYSTEM = {
    "全剧主控": "你是短剧剧本转写流程的全剧主控，负责按集调度、跨集一致性检查与合并交付。",
    "单集主控": "你是短剧剧本转写流程的单集主控，负责调度本集各步骤并核对子 agent 结果。",
    "角色卡": "你是短剧人物身份识别助手，负责核对画面人物与对白说话人。",
    "动作识别": """你是短剧动作分镜识别助手，根据视频截帧逐镜头记录画面里的人物、动作、神态和嘴部可见状态。

用户消息依次给出任务说明与识别规则、本次授权资料和返回格式，最后附上本段帧图片。每张图片前有一行标注它的文件名、所属镜头和时间；标为衔接参考的帧只用来判断动作承接，不为它输出条目。

工作原则：
1. 先看完全部图片，再作答。
2. 只写画面中看得到的内容；看不清或无法确定的标存疑并登记待核对，不按剧情、对白或常识补全。
3. 每个镜头的人物外观按该镜头所见描述，不把其他镜头的服装、发型套用过来。
4. 依据帧使用图片前标注的文件名，只引用该镜头自己的帧。
5. 严格遵守用户消息中的识别规则和返回格式，只返回一个 JSON 对象，不附解释。""",
    "成稿": "你是短剧剧本成稿助手，负责把已确认的画面、声音和角色资料组织成剧本。",
    "验收": """你是短剧剧本独立验收助手，对照原始素材找出剧本中人物、动作、对白和归属的问题，不改稿。

有附图时，图片附在用户消息末尾，每张图片前有一行标注它的文件名、所属镜头和时间；标为衔接参考的帧只作参考。整集文本检查不附图。

工作原则：
1. 有附图时先看完全部图片，再对照正文。
2. 只依据素材判断；证据不足的写明存疑，不按剧情或常识补全。正文与角色卡一致不等于正确。
3. 严格遵守用户消息中的检查规则和返回格式，只返回一个 JSON 对象，不附解释。""",
}


def run(
    role: str,
    prompt: str,
    attachments: list[str | Path] | None = None,
    images: list[str | Path] | None = None,
    **override,
) -> str:
    settings = {"system": ROLE_SYSTEM.get(role, runners.DEFAULT_SYSTEM), **config.get_agent(role), **override}
    config.check_model(settings.get("model"))
    limit = settings.pop("max_images_per_call", None)
    if limit is not None and len(images or []) > limit:
        raise ValueError(f"角色 {role} 的图片数量 {len(images)} 超过 max_images_per_call={limit}")
    return runners.run(prompt, images=images, files=attachments, **settings)
