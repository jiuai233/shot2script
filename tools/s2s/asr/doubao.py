"""豆包音频理解；时间戳和声线标签均为模型估计，需回听复核。"""

import base64
import json
import re

from .base import Sentence
from .common import ChunkedBackend
from .postprocess import seconds, sentence_times

PROMPT = """逐句听写音频中的全部人声，逐字照录，包括称呼、嗯、啊、重复词和插话，不润色，不根据剧情补写。
只返回 JSON 数组，每项字段：start（纯秒数）、end（纯秒数）、text（台词原文）、speaker（按首次出现编号 spk0、spk1……）、clarity（清楚/重叠/听不清）。
时间从本段音频 0 秒开始，start/end 对应实际发声起止，禁止时钟格式；一句一项，不漏短句。
仅凭声线区分说话人，同一声线前后一致；不能确认声线时 speaker 填 null，不猜人物姓名、画内或画外。
重叠说话分别记录并标重叠；听不清的字写（听不清）并标听不清；纯音乐、音效和静音不编造台词。
没有人声返回 []，不要解释，不要 Markdown。"""


def build_payload(path, settings):
    return {"model": settings["model"], "messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(path.read_bytes()).decode("ascii"), "format": "wav"}},
        {"type": "text", "text": PROMPT},
    ]}], "temperature": 0, "max_tokens": 16000}


class DoubaoBackend(ChunkedBackend):
    provider = "doubao"
    # 沿用已验证调用的 25 MB 音频文件上限；600 秒 PCM16 WAV 小于 20 MB。
    max_seconds = 600.0
    max_bytes = 25_000_000

    def send(self, client, path, key):
        return client.post(self.settings["api_base"].rstrip("/") + "/chat/completions",
                           headers={"Authorization": f"Bearer {key}"}, json=build_payload(path, self.settings))

    def parse_response(self, data, offset, duration, namespace):
        if not isinstance(data, dict):
            raise ValueError("响应须为对象")
        choice = data["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("响应缺少 message")
        if choice.get("finish_reason") == "length":
            raise ValueError("响应被 token 上限截断")
        content = choice["message"]["content"]
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not isinstance(content, str):
            raise ValueError("响应 content 须为文本")
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        rows = json.loads(content)
        if not isinstance(rows, list):
            raise ValueError("听写结果必须为数组")
        result = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError("句条目缺少文本")
            text = row["text"].strip()
            if not text:
                continue
            a, b = seconds(row["start"]), seconds(row["end"])
            if b <= a or a >= duration or b > duration + 0.15:
                raise ValueError("模型时间戳越界")
            speaker = row["speaker"]
            if speaker is not None and (isinstance(speaker, bool) or not isinstance(speaker, (str, int))):
                raise ValueError("声线标签无效")
            label = str(speaker) if speaker is not None else f"unknown-{index}"
            clarity = row.get("clarity", "听不清")
            if clarity not in ("清楚", "重叠", "听不清"):
                raise ValueError("听辨标签无效")
            if "听不清" in text and clarity != "重叠":
                clarity = "听不清"
            begin, stop = sentence_times(offset + a, offset + min(b, duration))
            result.append(Sentence(begin, stop, text, f"{namespace}:{label}", clarity))
        return sorted(result, key=lambda s: seconds(s.start))
