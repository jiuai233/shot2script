#!/usr/bin/env python3
"""单集音画隔离实验：语音与视频并行、融合写稿、文稿检查；MiMo 负责转写与读视频，DeepSeek 负责写稿与检查。"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Lock
import time

import httpx
from dotenv import dotenv_values
from jsonschema import validate, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from s2s import audio
from s2s.asr.postprocess import seconds
from s2s.asr.single import segment
from s2s.vad import detect, waveform

ENDPOINT = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
ASR_MODEL = "mimo-v2.5-asr"
PRO_MODEL = "mimo-v2.6-pro"
DEEPSEEK_MODEL = "deepseek-flash"
# 模型 → 端点、密钥变量、请求参数。
MODELS = {
    ASR_MODEL: (ENDPOINT, "MIMO_API_KEY", {"max_completion_tokens": 4096}),
    PRO_MODEL: (ENDPOINT, "MIMO_API_KEY", {"max_completion_tokens": 16384}),
    DEEPSEEK_MODEL: (DEEPSEEK_ENDPOINT, "DEEPSEEK_API_KEY",
                   {"max_tokens": 32768, "thinking": {"type": "enabled"}, "reasoning_effort": "high"}),
}
LOCAL_PYTHON = ROOT / "envs/local-asr/bin/python"
ALIGNER = ROOT / "models/asr/qwen3-aligner-0.6b"
STREAM_IDLE_SECONDS = 60
STREAM_TOTAL_SECONDS = 600
TZ = timezone(timedelta(hours=8))
STRING = {"type": "string"}
NULLABLE = {"type": ["string", "null"]}
STRINGS = {"type": "array", "items": STRING}
BOOL = {"type": "boolean"}
NUMBER = {"type": ["number", "string"]}


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties)}


def array(item):
    return {"type": "array", "items": item}


ISSUE = obj({"位置": {"type": "object"}, "依据": STRING, "修正要求": STRING})
FUSION_SCHEMA = obj({
    "人物": array(obj({"id": STRING, "名称": STRING, "外观": STRING, "别称": STRINGS,
                       "名称类型": {"enum": ["姓名", "关系称谓", "外观描述"]},
                       "命名依据": {"type": "string", "minLength": 1}, "视觉人物": STRINGS,
                       "依据": STRING, "待核对": BOOL})),
    "工作稿": {"type": "string", "minLength": 1},
    "逐句归属": array(obj({"句": STRING, "人物": NULLABLE, "形态": NULLABLE,
                         "起": NUMBER, "止": NUMBER, "文本": STRING,
                         "依据": STRING, "待核对": BOOL})),
    "采用动作": array(obj({"动作": STRING, "人物": NULLABLE, "对象人物": STRINGS,
                         "起": NUMBER, "止": NUMBER, "表述": STRING, "依据": STRING})),
    "未决问题": array(obj({"id": STRING, **ISSUE["properties"]})),
})
REVIEW_SCHEMA = obj({"问题": array(obj({**ISSUE["properties"], "原文": STRING,
                                          "级别": {"enum": ["阻断", "规范"]}}))})
PENDING_MARK = re.compile(r"\s*【待核对[^】]*】")
BRACKET_LIMIT = 12
LINE_TAGS = re.compile(r"^((?:〔[^〕]+〕)+)\s*")
VISION_SCHEMA = obj({name: array({"type": "object"}) for name in
                     ("人物", "动作", "对白归属候选", "待核对")})


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now():
    return datetime.now(TZ).isoformat(timespec="milliseconds")


def parse_response(data):
    choice = data["choices"][0]
    reason = choice.get("finish_reason")
    if reason != "stop":
        raise ValueError(f"模型请求未正常完成：finish_reason={reason}")
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型未返回完整文本")
    return content


def parse_json(content, schema):
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    value = json.loads(content)
    try:
        validate(value, schema)
    except ValidationError as exc:
        raise ValueError(f"结果结构损坏：{list(exc.absolute_path)}：{exc.message}") from None
    return value


def fusion_inputs(episode, total, speech, vision, confirmed_names=()):
    rows = [{**{k: source[k] for k in ("id", "起", "止", "文本")},
             "异常": source.get("核对", {}).get("异常", [])} for source in speech["句列表"]]
    inputs = {"集号": episode, "时长秒": total, "视觉": vision,
              "语音": {"句列表": rows, "疑似漏段": speech.get("疑似漏段", []),
                       "待核对": speech["待核对"]}}
    if confirmed_names:
        inputs["已确认人物命名"] = confirmed_names
    return inputs


def parallel_branches(speech, vision):
    # 只在此建立两路并发；任一路失败均不进入融合。
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(speech), pool.submit(vision)
        return a.result(), b.result()


def issue(reason, **location):
    return {"位置": location, "类型": "全文完整性与一致性", "级别": "规范",
            "依据": reason, "修正要求": "按原素材和原句记录核对，保留完整台词及未决事实。"}


def blocker(reason, **location):
    """台词被改字、漏句、乱序或凭空多出对白，读者会被误导，判为阻断。"""
    return {**issue(reason, **location), "级别": "阻断"}


def valid_range(row, total):
    try:
        a, b = seconds(row["起"]), seconds(row["止"])
        return math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= total + .05
    except (ValueError, TypeError, KeyError):
        return False


def locate_lines(fusion):
    """去掉行首编号标记，按标记位置填写逐句归属和采用动作的正文行号。"""
    problems, places, lines = [], {}, []
    for number, line in enumerate(fusion["工作稿"].splitlines(), 1):
        match = LINE_TAGS.match(line)
        if match:
            for tag in re.findall(r"〔([^〕]+)〕", match[1]):
                if tag in places:
                    problems.append(issue("同一编号标记出现在多行", 编号=tag, 行=[places[tag], number]))
                places.setdefault(tag, number)
            line = line[match.end():]
        lines.append(line)
    known = {r["句"] for r in fusion["逐句归属"]} | {a["动作"] for a in fusion["采用动作"]}
    problems.extend(issue("正文标记引用未登记的句或采用动作", 编号=tag, 行=number)
                    for tag, number in places.items() if tag not in known)
    # 无标记的行号记为 0，由固定检查报告为无效行引用。
    return {**fusion, "工作稿": "\n".join(lines),
            "逐句归属": [{**r, "行": places.get(r["句"], 0)} for r in fusion["逐句归属"]],
            "采用动作": [{**a, "行": places.get(a["动作"], 0)} for a in fusion["采用动作"]]}, problems


def script_parts(line):
    """分离允许统一名称的文字与必须逐字保留的对白、字卡。"""
    stripped = line.lstrip()
    if stripped.startswith("△ 画面字幕："):
        return "", line
    if stripped.startswith(("△", "人物：", "场景", "第")):
        return line, ""
    head, colon, dialogue = line.partition("：")
    return head, colon + dialogue


def render_script(fusion):
    names = {p["id"]: p["名称"] for p in fusion["人物"]}
    lines = []
    for line in fusion["工作稿"].splitlines():
        # 未决问题只进清单，正文不保留待核对标记。
        line = PENDING_MARK.sub("", line)
        head, untouched = script_parts(line)
        head = re.sub(r"\{\{(P\d+)\}\}", lambda m: names.get(m[1], m[0]), head)
        lines.append(head + untouched)
    return {**fusion, "工作稿": "\n".join(lines)}


def name_checks(fusion):
    problems, owners = [], {}
    people = fusion["人物"]
    names = {p["名称"] for p in people}
    aliases = {alias for p in people for alias in p["别称"] if alias and alias not in names}
    name_pattern = "|".join(re.escape(n) for n in sorted(names | aliases, key=len, reverse=True) if n)
    for person in people:
        for visual in person["视觉人物"]:
            if visual in owners:
                problems.append({**issue("同一视觉人物重复映射到多个统一人物", 视觉人物=visual),
                                 "类型": "人物指代与角色卡"})
            owners[visual] = person["id"]
    for number, line in enumerate(fusion["工作稿"].splitlines(), 1):
        head, _ = script_parts(line)
        if not head.strip():
            continue
        reasons = []
        if re.search(r"\{\{P\d+\}\}", head):
            reasons.append("正文人物引用不存在")
        if head.strip().startswith("人物："):
            listed = re.split(r"[、，,]", head.strip().removeprefix("人物："))
            if any(n.strip() not in names for n in listed):
                reasons.append("人物行未使用统一显示名")
        if name_pattern and any(m[0] in aliases for m in re.finditer(name_pattern, head)):
            reasons.append("非对白出现别称或旧外观名，未使用统一显示名")
        if reasons:
            problems.append({**issue("；".join(reasons), 行=number), "类型": "人物指代与角色卡"})
    return problems


def fixed_checks(fusion, speech, vision, total):
    problems = name_checks(fusion)
    original = {r["id"]: r for r in speech["句列表"]}
    rows = fusion["逐句归属"]
    ids = [r["句"] for r in rows]
    if ids != list(original):
        problems.append(blocker("原句未按序覆盖一次（遗漏、重复、未知句或乱序）", 原句=list(original), 实际句=ids))
    people = {p["id"]: p for p in fusion["人物"]}
    if len(people) != len(fusion["人物"]) or len({p["名称"] for p in people.values()}) != len(people):
        problems.append(issue("人物编号或显示名重复"))
    lines = fusion["工作稿"].splitlines()
    known_actions = {a.get("id") for a in vision["动作"]}
    known_visual = {p.get("id") for p in vision["人物"]}
    for person in people.values():
        if any(v not in known_visual for v in person["视觉人物"]):
            problems.append(issue("人物映射引用不存在的视觉人物", 人物=person["id"]))
    assigned_lines = []
    for row in rows + fusion["采用动作"]:
        location = {"行": row["行"]}
        if "句" in row:
            location["句"] = row["句"]
        if row["人物"] is not None and row["人物"] not in people:
            problems.append(issue("人物引用不存在，未知必须用 JSON null", **location))
        if not valid_range(row, total):
            problems.append(issue("时间范围无效或超出整集边界", **location))
        index = row["行"] - 1
        if not 0 <= index < len(lines):
            problems.append((blocker if "句" in row else issue)("正文缺少编号标记或行引用无效", **location))
            continue
        if "句" in row:
            assigned_lines.append(row["行"])
            source = original.get(row["句"])
            if source:
                if row["文本"] != source["文本"]:
                    problems.append(issue("对照台词与原文不一致", **location))
                if (row["起"], row["止"]) != (source["起"], source["止"]):
                    problems.append(issue("原句对齐时间被改动", **location))
                # 冒号后仅允许原句和显式待核对标记，防止对照表正确而正文改字。
                dialogue = lines[index].partition("：")[2]
                dialogue = re.sub(r"【待核对[^】]*】", "", dialogue).strip()
                if dialogue != source["文本"]:
                    problems.append(blocker("正文台词与原句不一致", **location))
            if row["形态"] not in (None, "画内", "画外", "心声", "电话", "视频", "广播"):
                problems.append(issue("声音形态无效，未知必须用 JSON null", **location))
            if row["人物"] is None or row["形态"] is None or row["待核对"]:
                problems.append(issue("对白身份、形态或归属仍待核对", **location))
            label = lines[index].partition("：")[0]
            speaker = label.split("（", 1)[0].strip()
            if row["人物"] in people and speaker != people[row["人物"]]["名称"]:
                problems.append({**issue("正文说话人未使用归属表统一显示名", **location),
                                 "类型": "人物指代与角色卡"})
            bracket = re.search(r"（(.*)）", label)
            if bracket and len(bracket[1]) > BRACKET_LIMIT:
                problems.append({**issue(f"说话人括号超过 {BRACKET_LIMIT} 字，应改写 △", **location),
                                 "类型": "动作与对白关联"})
        else:
            if row["动作"] not in known_actions:
                problems.append(issue("采用动作引用不存在", **location))
            if any(p not in people for p in row["对象人物"]):
                problems.append(issue("动作对象人物引用不存在", **location))
    if len(assigned_lines) != len(set(assigned_lines)):
        problems.append(blocker("多个原句引用同一正文行"))
    if assigned_lines != sorted(assigned_lines):
        problems.append(blocker("正文对白行顺序与原句顺序不一致"))
    for n, line in enumerate(lines, 1):
        if "：" in line and not line.startswith(("场景", "人物：", "△", "第")) and n not in assigned_lines:
            problems.append(blocker("正文存在未关联原句的对白行", 行=n))
    return problems


def final_review(review, fusion, speech, vision, checks):
    """只有阻断问题判不通过；上游待核对和规范问题列入清单，不阻断。"""
    issues = [{"来源": "固定检查", **p} for p in checks] + [{"来源": "文稿检查", **p} for p in review["问题"]]
    for label, values in (("融合未决", fusion["未决问题"]), ("视频待核对", vision["待核对"]),
                          ("语音待核对", speech["待核对"])):
        issues.extend({"来源": label, **p, "级别": "规范"} for p in values)
    issues.extend({"来源": "人物映射", **issue("人物映射待核对", 人物=p["id"])} for p in fusion["人物"] if p["待核对"])
    blocking = [p for p in issues if p.get("级别") == "阻断"]
    listed = [p for p in issues if p.get("级别") != "阻断"]
    conclusion = "不通过" if blocking else "通过（附问题清单）" if listed else "通过"
    return {"结论": conclusion, "阻断问题": blocking, "问题清单": listed}


class Run:
    def __init__(self, output, video, keys):
        self.output, self.video, self.keys = output, video, keys
        self.artifacts = output / "阶段产物"
        self.artifacts.mkdir()
        self.lock = Lock()
        self.record = {"输入": str(video), "开始时间": now(), "流程状态": "运行中", "内容结论": None,
                       "端点": {"MiMo": ENDPOINT, "DeepSeek": DEEPSEEK_ENDPOINT}, "远程请求数": 0, "阶段": {}, "调用": {},
                       "视频参数": {"fps": 2, "media_resolution": "default", "实测采样帧数": None,
                                  "实测缩放分辨率": None}, "套餐实际扣额": None}
        self.persist()

    def hide(self, text):
        for key in self.keys.values():
            text = text.replace(key, "[REDACTED]")
        return text

    def persist(self):
        save(self.output / "运行记录.json", self.record)

    @contextmanager
    def stage(self, name):
        started = time.monotonic()
        row = {"开始时间": now(), "状态": "运行中"}
        with self.lock:
            self.record["阶段"][name] = row
            self.persist()
        print(f"[{now()}] {name} 开始", flush=True)
        try:
            yield
            row["状态"] = "完成"
        except Exception as exc:
            row.update(状态="失败", 错误=self.hide(str(exc)))
            raise
        finally:
            with self.lock:
                row.update(结束时间=now(), 耗时秒=round(time.monotonic() - started, 3))
                self.persist()
            print(f"[{now()}] {name} {row['状态']}", flush=True)

    def request(self, name, model, content, **options):
        endpoint, key_name, settings = MODELS[model]
        headers = {"Authorization": "Bearer " + self.keys[key_name]}
        payload = {"model": model, "messages": [{"role": "user", "content": content}], **settings, **options}
        row = {"模型": model, "端点": endpoint, "响应模型": None, "参数": {**settings, **options},
               "开始时间": now(), "HTTP状态": None, "finish_reason": None, "usage": None,
               "重试次数": 0}
        with self.lock:
            self.record["调用"][name] = row
            self.record["远程请求数"] += 1
            self.persist()
        started = time.monotonic()
        try:
            if model == ASR_MODEL:
                with httpx.Client(timeout=httpx.Timeout(180, connect=15), follow_redirects=False) as client:
                    response = client.post(endpoint, headers=headers, json=payload)
                row["HTTP状态"] = response.status_code
                (self.artifacts / f"{name}-原始响应.txt").write_text(self.hide(response.text), encoding="utf-8")
                if response.is_error:
                    raise ValueError(f"HTTP {response.status_code}：{response.text[:1000]}")
                data = response.json()
            else:
                data = self.stream(name, endpoint, headers, payload, row, started)
            save(self.artifacts / f"{name}-响应.json", data)
            row.update(响应模型=data.get("model"), usage=data.get("usage"),
                       finish_reason=(data.get("choices") or [{}])[0].get("finish_reason"))
            content = parse_response(data)
            (self.artifacts / f"{name}-输出.txt").write_text(content, encoding="utf-8")
            return content
        except Exception as exc:
            row["错误"] = self.hide(str(exc))
            raise
        finally:
            with self.lock:
                row.update(结束时间=now(), 耗时秒=round(time.monotonic() - started, 3))
                save(self.artifacts / f"{name}-调用记录.json", row)
                self.persist()

    def stream(self, name, endpoint, headers, payload, row, started):
        """流式接收，按空闲时间判断中断；原始分片与推理过程边收边写入阶段产物。"""
        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        row["参数"].update(stream=True)
        reasoning, content, finish, usage, model = [], [], None, None, None
        timeout = httpx.Timeout(connect=15, read=STREAM_IDLE_SECONDS, write=60, pool=15)
        raw = (self.artifacts / f"{name}-原始响应.txt").open("w", encoding="utf-8")
        try:
            with httpx.Client(timeout=timeout, follow_redirects=False) as client, \
                    client.stream("POST", endpoint, headers=headers, json=payload) as response:
                row["HTTP状态"] = response.status_code
                if response.is_error:
                    text = response.read().decode("utf-8", "replace")
                    raw.write(text)
                    raise ValueError(f"HTTP {response.status_code}：{text[:1000]}")
                for line in response.iter_lines():
                    raw.write(line + "\n")
                    if time.monotonic() - started > STREAM_TOTAL_SECONDS:
                        raise ValueError(f"总等待超过 {STREAM_TOTAL_SECONDS} 秒，已停止，未重试")
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    model, usage = chunk.get("model") or model, chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        reasoning.append(delta.get("reasoning_content") or "")
                        content.append(delta.get("content") or "")
                        finish = choice.get("finish_reason") or finish
        except httpx.ReadTimeout:
            raise ValueError(f"{STREAM_IDLE_SECONDS} 秒未收到新数据，已停止，未重试") from None
        finally:
            raw.close()
            (self.artifacts / f"{name}-推理.txt").write_text("".join(reasoning), encoding="utf-8")
        return {"model": model, "usage": usage, "choices": [{"finish_reason": finish, "message": {
            "content": "".join(content), "reasoning_content": "".join(reasoning)}}]}

    def ask(self, name, inputs, schema, video=None, model=PRO_MODEL):
        prompt = (ROOT / f"prompts/mvp/{name}.md").read_text(encoding="utf-8")
        text = prompt + "\n\n输入：\n" + json.dumps(inputs, ensure_ascii=False)
        text += "\n\n返回 Schema：\n" + json.dumps(schema, ensure_ascii=False)
        (self.artifacts / f"{name}-输入.txt").write_text(text, encoding="utf-8")
        content = [] if video is None else [video]
        content.append({"type": "text", "text": text})
        with self.stage(name):
            result = parse_json(self.request(name, model, content), schema)
            save(self.artifacts / f"{name}-结果.json", result)
            return result

    def speech(self, wav, episode):
        with self.stage("语音支路"):
            with self.stage("ASR"):
                encoded = base64.b64encode(wav.read_bytes()).decode()
                text = self.request("ASR", ASR_MODEL, [{"type": "input_audio", "input_audio": {
                    "data": "data:audio/wav;base64," + encoded}}], asr_options={"language": "zh"})
                text_path = self.artifacts / "ASR-转写.txt"
                text_path.write_text(text, encoding="utf-8")
            with self.stage("指定文本对齐"):
                env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                env["PYTHONPATH"] = str(ROOT / "tools") + os.pathsep + env.get("PYTHONPATH", "")
                output = self.artifacts / "对齐.json"
                try:
                    process = subprocess.run([str(LOCAL_PYTHON), "-m", "s2s.asr.align_text", str(wav),
                                              str(text_path), str(output), "--model", str(ALIGNER)],
                                             env=env, capture_output=True, text=True, timeout=180)
                except subprocess.TimeoutExpired:
                    raise ValueError("指定文本对齐超过 180 秒，已停止，未重试") from None
                (self.artifacts / "对齐日志.txt").write_text(process.stderr, encoding="utf-8")
                if process.returncode:
                    raise ValueError("指定文本对齐失败，见阶段产物/对齐日志.txt")
                alignment = json.loads(output.read_text(encoding="utf-8"))
            with self.stage("VAD与切句"):
                voices = detect(wav)
                save(self.artifacts / "VAD.json", voices)
                sounds, missing = segment(alignment["words"], voices, waveform(wav), episode,
                                          min_pause=.3, max_sentence=6)
                # 切句附带的占位声线不是识别结果，不进入后续输入。
                for sound in sounds:
                    sound.pop("声线", None)
                save(self.artifacts / "切句.json", {"句列表": sounds, "疑似漏段": missing})
            pending = [issue("；".join(sound["核对"]["异常"]), 句=sound["id"], 时间=sound["起"])
                       for sound in sounds if sound["核对"]["异常"]]
            pending.extend(issue("有人声但没有识别词，疑似漏段", 时间=m) for m in missing)
            result = {"ASR全文": text, "词时间": alignment["words"], "句列表": sounds,
                      "疑似漏段": missing, "待核对": pending}
            save(self.artifacts / "语音结果.json", result)
            return result

    def execute(self):
        started = time.monotonic()
        try:
            with self.stage("固定准备"):
                total = audio.duration(self.video)
                self.record["时长秒"] = total
                wav = audio.extract_audio(self.video, self.artifacts / "原始混音.wav")
                video = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," +
                         base64.b64encode(self.video.read_bytes()).decode()},
                         "fps": 2, "media_resolution": "default"}
                episode = self.video.stem if re.fullmatch(r"E\d{2,}", self.video.stem) else "E01"
            speech, vision = parallel_branches(
                lambda: self.speech(wav, episode),
                lambda: self.ask("视频理解", {"集": episode, "时长秒": total}, VISION_SCHEMA, video))
            inputs = fusion_inputs(int(episode[1:]), total, speech, vision)
            fusion, placement = locate_lines(self.ask("融合写稿", inputs, FUSION_SCHEMA, model=DEEPSEEK_MODEL))
            fusion = render_script(fusion)
            (self.output / "工作稿.md").write_text(fusion["工作稿"], encoding="utf-8")
            save(self.output / "人物归属.json", {k: v for k, v in fusion.items() if k != "工作稿"})
            with self.stage("固定检查"):
                checks = placement + fixed_checks(fusion, speech, vision, total)
                save(self.artifacts / "固定检查.json", checks)
            lines = fusion["工作稿"].splitlines()
            review = self.ask("文稿检查", {"集": episode,
                "带行号工作稿": "\n".join(f"{i}: {line}" for i, line in enumerate(lines, 1)),
                "台词": [{k: r[k] for k in ("id", "起", "止", "文本")} for r in speech["句列表"]],
                "人物": [{k: p[k] for k in ("名称", "别称")} for p in fusion["人物"]]
            }, REVIEW_SCHEMA, model=DEEPSEEK_MODEL)
            result = final_review(review, fusion, speech, vision, checks)
            save(self.output / "核对结果.json", result)
            self.record.update(流程状态="完成", 内容结论=result["结论"])
            return result
        except Exception as exc:
            self.record.update(流程状态="中断", 错误=self.hide(str(exc)))
            raise
        finally:
            self.record.update(结束时间=now(), 耗时秒=round(time.monotonic() - started, 3))
            self.persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--输出", "--output", dest="output", type=Path, required=True)
    args = parser.parse_args()
    env = dotenv_values(ROOT / ".env")
    keys = {name: (env.get(name) or "").strip() for name in ("MIMO_API_KEY", "DEEPSEEK_API_KEY")}
    missing = [name for name, key in keys.items() if not key]
    if missing:
        parser.error(f"项目 .env 缺少 {'、'.join(missing)}，未发请求")
    if not args.video.is_file():
        parser.error(f"视频不存在：{args.video}")
    if 4 * ((args.video.stat().st_size + 2) // 3) >= 50 * 1024 * 1024:
        parser.error("视频 Base64 超出本入口整集 50 MB 上限")
    if not LOCAL_PYTHON.is_file() or not (ALIGNER / "model.safetensors").is_file():
        parser.error("缺少本地对齐环境或权重，未发请求")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"输出目录已存在，不覆盖：{args.output}")
    run = Run(args.output.resolve(), args.video.resolve(), keys)
    try:
        result = run.execute()
    except Exception as exc:
        print(f"流程中断：{run.hide(str(exc))}；产物保存在 {args.output}", file=sys.stderr)
        return 1
    print(f"流程完成／{result['结论']}：{args.output / '工作稿.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
