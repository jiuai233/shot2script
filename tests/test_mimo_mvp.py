"""MiMo 隔离入口的汇合、完整性、解析与真实调用边界；不发远程请求。"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
import time
import wave

import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mimo_mvp", ROOT / "tools/run_mimo_mvp.py")
mvp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mvp)


@pytest.fixture
def sample():
    speech = {"ASR全文": "你好。", "词时间": [{"text": "你好。", "start": .5, "end": 1.5}],
              "句列表": [{"id": "E01-S001", "起": "00:00.5", "止": "00:01.5", "文本": "你好。"}], "待核对": []}
    vision = {"人物": [{"id": "T001"}], "动作": [], "对白归属候选": [], "待核对": []}
    fusion = {"人物": [{"id": "P001", "名称": "穿白衬衫的短发女子", "视觉人物": ["T001"],
                       "外观": "白衬衫、短发", "别称": [], "名称类型": "外观描述",
                       "命名依据": "可见衣着和发型", "依据": "衣着和原声连续", "待核对": False}],
              "逐句归属": [{"句": "E01-S001", "人物": "P001", "形态": "画内", "起": "00:00.5",
                          "止": "00:01.5", "文本": "你好。", "行": 4, "依据": "原声与嘴部变化", "待核对": False}],
              "采用动作": [], "工作稿": "第1集\n场景1-1：走廊，日，内\n人物：穿白衬衫的短发女子\n穿白衬衫的短发女子：你好。",
              "未决问题": []}
    review = {"问题": []}
    return speech, vision, fusion, review


def test_parallel_branches_wait_for_both():
    barrier = Barrier(2)
    completed = Event()

    def speech():
        barrier.wait(timeout=2)
        return "speech"

    def vision():
        barrier.wait(timeout=2)
        time.sleep(.02)
        completed.set()
        return "vision"

    assert mvp.parallel_branches(speech, vision) == ("speech", "vision")
    assert completed.is_set()


def test_failed_branch_does_not_return_joined_data():
    def fail():
        raise ValueError("ASR 失败")
    with pytest.raises(ValueError, match="ASR 失败"):
        mvp.parallel_branches(fail, lambda: {})


def test_valid_draft_and_json_null(sample):
    speech, vision, fusion, _ = sample
    assert mvp.fixed_checks(fusion, speech, vision, 2) == []
    fusion["逐句归属"][0]["人物"] = None
    mvp.validate(fusion, mvp.FUSION_SCHEMA)
    reasons = [p["依据"] for p in mvp.fixed_checks(fusion, speech, vision, 2)]
    assert not any("人物引用不存在" in r for r in reasons)
    assert any("仍待核对" in r for r in reasons)


@pytest.mark.parametrize("change", ["missing", "duplicate", "text", "body", "time", "null", "shape_null", "line", "extra"])
def test_fixed_checks_keep_content_differences(sample, change):
    speech, vision, fusion, _ = sample
    row = fusion["逐句归属"][0]
    if change == "missing":
        fusion["逐句归属"] = []
    elif change == "duplicate":
        fusion["逐句归属"].append(deepcopy(row))
    elif change == "text":
        row["文本"] = "您好。"
    elif change == "body":
        fusion["工作稿"] = fusion["工作稿"].replace("你好。", "您好。")
    elif change == "time":
        row["止"] = 3
    elif change == "null":
        row["人物"] = "null"
    elif change == "shape_null":
        row["形态"] = "null"
    elif change == "line":
        row["行"] = 99
    elif change == "extra":
        fusion["工作稿"] += "\n穿白衬衫的短发女子：再见。"
    assert mvp.fixed_checks(fusion, speech, vision, 2)


def test_bad_structure_stops_but_content_differences_parse(sample):
    _, _, fusion, _ = sample
    fusion["逐句归属"][0]["人物"] = "null"
    assert mvp.parse_json(json.dumps(fusion), mvp.FUSION_SCHEMA) == fusion
    del fusion["工作稿"]
    with pytest.raises(ValueError, match="结构损坏"):
        mvp.parse_json(json.dumps(fusion), mvp.FUSION_SCHEMA)
    with pytest.raises(ValueError):
        mvp.parse_json("{未完成", mvp.FUSION_SCHEMA)


@pytest.mark.parametrize("reason,content", [("content_filter", "The request was rejected because it was considered high risk"),
                                            ("length", "{未完成")])
def test_http_200_nonstop_is_not_success(reason, content):
    with pytest.raises(ValueError, match=reason):
        mvp.parse_response({"choices": [{"finish_reason": reason, "message": {"content": content}}]})


def test_only_blocking_issues_fail_and_upstream_issues_are_listed(sample):
    speech, vision, fusion, review = sample
    assert mvp.final_review(review, fusion, speech, vision, []) == {"结论": "通过", "阻断问题": [], "问题清单": []}
    # 上游自报“阻断”也只列入清单。
    fusion["未决问题"] = [{"id": "Q001", "位置": {"行": 4}, "级别": "阻断", "依据": "电话声音待核对", "修正要求": "回听"}]
    review["问题"] = [{"位置": {"行": 4}, "原文": "穿白衬衫的短发女子：你好。", "级别": "规范",
                     "依据": "括号用法", "修正要求": "调整"}]
    result = mvp.final_review(review, fusion, speech, vision, [mvp.issue("人物行未使用统一显示名", 行=3)])
    assert result["结论"] == "通过（附问题清单）" and result["阻断问题"] == []
    assert [p["来源"] for p in result["问题清单"]] == ["固定检查", "文稿检查", "融合未决"]
    review["问题"][0]["级别"] = "阻断"
    assert mvp.final_review(review, fusion, speech, vision, [])["结论"] == "不通过"
    review["问题"] = []
    result = mvp.final_review(review, fusion, speech, vision, [mvp.blocker("正文台词与原句不一致", 行=4)])
    assert result["结论"] == "不通过" and result["阻断问题"][0]["来源"] == "固定检查"


def test_dialogue_integrity_problems_block_but_pending_attribution_does_not(sample):
    speech, vision, fusion, _ = sample
    fusion["逐句归属"][0]["待核对"] = True
    assert {p["级别"] for p in mvp.fixed_checks(fusion, speech, vision, 2)} == {"规范"}
    fusion["工作稿"] = fusion["工作稿"].replace("你好。", "您好。")
    assert "阻断" in {p["级别"] for p in mvp.fixed_checks(fusion, speech, vision, 2)}


KEYS = {"MIMO_API_KEY": "test-mimo-key", "DEEPSEEK_API_KEY": "test-deepseek-key"}


def make_run(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    video = tmp_path / "E01.mp4"
    video.write_bytes(b"sample-video")
    return mvp.Run(output, video, KEYS)


def reply(payload, data):
    """按请求方式返回：流式请求拆成推理与正文分片。"""
    if not payload.get("stream"):
        return httpx.Response(200, json=data)
    choice = data["choices"][0]
    chunks = [{"model": data["model"], "choices": [{"delta": {"reasoning_content": "先想一想。"}}]},
              {"model": data["model"], "choices": [{"delta": {"content": choice["message"]["content"]},
                                                     "finish_reason": choice["finish_reason"]}]},
              {"model": data["model"], "choices": [], "usage": data.get("usage")}]
    body = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def install_http(monkeypatch, handler):
    client = httpx.Client
    monkeypatch.setattr(mvp.httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handler), **kw))


def test_rejected_request_saved_without_retry(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    requests = []

    def handler(request):
        requests.append(request)
        return reply(json.loads(request.content), {"model": mvp.PRO_MODEL, "choices": [{"finish_reason": "content_filter",
            "message": {"content": "The request was rejected because it was considered high risk"}}]})

    install_http(monkeypatch, handler)
    with pytest.raises(ValueError, match="content_filter"):
        run.request("视频理解", mvp.PRO_MODEL, [])
    assert len(requests) == 1
    assert (run.artifacts / "视频理解-原始响应.txt").is_file()
    assert run.record["调用"]["视频理解"]["HTTP状态"] == 200
    assert run.record["调用"]["视频理解"]["usage"] is None
    assert run.record["调用"]["视频理解"]["参数"]["stream"] is True
    assert run.record["调用"]["视频理解"]["finish_reason"] == "content_filter"


@pytest.mark.parametrize("corrupt_draft", [False, True])
def test_full_chain_four_calls_text_check(tmp_path, monkeypatch, sample, corrupt_draft):
    speech, vision, fusion, review = sample
    run = make_run(tmp_path)
    requests = []
    monkeypatch.setattr(mvp.audio, "duration", lambda _: 2.)

    def extract(source, target):
        with wave.open(str(target), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(np.full(32000, 1000, dtype="<i2").tobytes())
        return target

    monkeypatch.setattr(mvp.audio, "extract_audio", extract)
    monkeypatch.setattr(mvp, "detect", lambda _: [{"起": .4, "止": 1.6}])

    def align(command, **kwargs):
        assert command[2] == "s2s.asr.align_text"
        mvp.save(command[5], {"text": "你好。", "words": speech["词时间"]})
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(mvp.subprocess, "run", align)
    monkeypatch.setattr(mvp, "read_subtitles", lambda video, directory: [
        {"起": "00:00.5", "止": "00:01.5", "文字": "你好", "类型": "台词字幕", "置信度": 1.0}])

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        endpoint, key_name, settings = mvp.MODELS[payload["model"]]
        assert str(request.url) == endpoint
        assert request.headers["Authorization"] == "Bearer " + KEYS[key_name]
        assert {k: payload[k] for k in settings} == settings
        if payload["model"] != mvp.DEEPSEEK_MODEL:
            assert "thinking" not in payload and "reasoning_effort" not in payload
        content = payload["messages"][0]["content"]
        assert payload.get("stream", False) == (payload["model"] != mvp.ASR_MODEL)
        if payload["model"] == mvp.ASR_MODEL:
            assert payload["asr_options"] == {"language": "zh"}
            return reply(payload, {"model": payload["model"], "usage": {"total_tokens": 12},
                "choices": [{"finish_reason": "stop", "message": {"content": "你好。"}}]})
        prompt = content[-1]["text"]
        if prompt.startswith("# 整集视频理解"):
            assert content[0]["fps"] == 2 and payload["model"] == mvp.PRO_MODEL
            assert "语音结果" not in prompt
            result = json.dumps(vision, ensure_ascii=False)
        elif prompt.startswith("# 单集音画融合写稿"):
            assert payload["model"] == mvp.DEEPSEEK_MODEL and len(content) == 1
            assert run.record["阶段"]["语音支路"]["状态"] == "完成"
            assert run.record["阶段"]["视频理解"]["状态"] == "完成"
            inputs = json.loads(prompt.split("\n\n输入：\n", 1)[1].rsplit("\n\n返回 Schema：\n", 1)[0])
            assert "成稿内容规则" not in inputs
            assert "词时间" not in inputs["语音"] and "ASR全文" not in inputs["语音"]
            assert "核对" not in inputs["语音"]["句列表"][0]
            assert "声纹" not in prompt and "声线" not in prompt
            actual = json.loads((run.artifacts / "语音结果.json").read_text())["句列表"][0]
            fusion["逐句归属"][0].update({k: actual[k] for k in ("起", "止", "文本")})
            if corrupt_draft:
                fusion["工作稿"] = fusion["工作稿"].replace("你好。", "您好。")
            assert '"行"' not in prompt.split("\n\n返回 Schema：\n", 1)[1]
            script = fusion["工作稿"].replace(fusion["人物"][0]["名称"], "{{P001}}")
            template = {**fusion, "工作稿": script.replace("{{P001}}：", "〔E01-S001〕{{P001}}："),
                        "逐句归属": [{k: v for k, v in r.items() if k != "行"} for r in fusion["逐句归属"]]}
            result = json.dumps(template, ensure_ascii=False)
        else:
            assert prompt.startswith("# 单集文稿检查") and payload["model"] == mvp.DEEPSEEK_MODEL
            assert len(content) == 1 and "词时间" not in prompt
            assert "衣着和原声连续" not in prompt
            assert "声纹" not in prompt and "声线" not in prompt
            assert "T001" not in prompt and "E01-V1" not in prompt
            result = json.dumps(review, ensure_ascii=False)
        return reply(payload, {"model": payload["model"], "usage": {"total_tokens": 12},
            "choices": [{"finish_reason": "stop", "message": {"content": result}}]})

    install_http(monkeypatch, handler)
    result = run.execute()
    assert len(requests) == 4
    assert run.record["流程状态"] == "完成"
    assert result["结论"] == ("不通过" if corrupt_draft else "通过")
    assert run.record["内容结论"] == result["结论"]
    assert run.record["阶段"]["字幕校字"]["状态"] == "完成"
    assert {"工作稿.md", "人物归属.json", "核对结果.json", "运行记录.json", "阶段产物"} <= {p.name for p in run.output.iterdir()}
    record = (run.output / "运行记录.json").read_text()
    assert all(key not in record for key in KEYS.values())
    assert sorted(r["model"] for r in requests[:2]) == sorted([mvp.ASR_MODEL, mvp.PRO_MODEL])
    assert [r["model"] for r in requests[2:]] == [mvp.DEEPSEEK_MODEL, mvp.DEEPSEEK_MODEL]
    assert "{{P001}}" not in (run.output / "工作稿.md").read_text()
    assert "{{P001}}" in (run.artifacts / "融合写稿-结果.json").read_text()
    assert "〔" not in (run.output / "工作稿.md").read_text()
    assert json.loads((run.output / "人物归属.json").read_text())["逐句归属"][0]["行"] == 4
    assert not (ROOT / "config/models.json").samefile(run.output / "运行记录.json")


def test_body_order_cannot_differ_from_sentence_order(sample):
    speech, vision, fusion, _ = sample
    second = {**speech["句列表"][0], "id": "E01-S002", "起": "00:01.5", "止": "00:01.9", "文本": "再见。"}
    speech["句列表"].append(second)
    fusion["逐句归属"][0]["行"] = 5
    fusion["逐句归属"].append({**fusion["逐句归属"][0], "句": second["id"], "起": second["起"],
                               "止": second["止"], "文本": second["文本"], "行": 4})
    fusion["工作稿"] = fusion["工作稿"].replace("女子：你好。", "女子：再见。\n穿白衬衫的短发女子：你好。")
    assert any("顺序" in p["依据"] for p in mvp.fixed_checks(fusion, speech, vision, 2))


def test_branch_rejection_interrupts_before_fusion(tmp_path, monkeypatch, sample):
    run = make_run(tmp_path)
    monkeypatch.setattr(mvp.audio, "duration", lambda _: 2.)
    monkeypatch.setattr(mvp.audio, "extract_audio", lambda source, target: target)
    monkeypatch.setattr(run, "speech", lambda *_: sample[0])
    calls = []

    def ask(name, *args):
        calls.append(name)
        raise ValueError("finish_reason=content_filter")

    monkeypatch.setattr(run, "ask", ask)
    with pytest.raises(ValueError, match="content_filter"):
        run.execute()
    assert calls == ["视频理解"]
    assert run.record["流程状态"] == "中断" and run.record["内容结论"] is None
    assert not (run.output / "工作稿.md").exists()


@pytest.fixture
def grandma(sample):
    speech, vision, fusion, review = sample
    person = fusion["人物"][0]
    fusion["工作稿"] = fusion["工作稿"].replace(person["名称"], "奶奶")
    person.update(名称="奶奶", 名称类型="关系称谓", 外观="灰白卷发、红色碎花衬衫",
                  别称=["桂枝", "妈", "穿碎花衬衫的白发女子"], 命名依据="已确认的祖孙关系与称呼回应")
    return speech, vision, fusion, review


def test_render_names_from_one_person_table_without_changing_dialogue(grandma):
    _, _, fusion, _ = grandma
    fusion["工作稿"] = ("第1集\n场景1-1：店内，日，内\n人物：{{P001}}\n"
                       "△ {{P001}}接起电话。\n{{P001}}（看着{{P001}}的手机）：妈，桂枝说{{P001}}。")
    rendered = mvp.render_script(fusion)
    assert rendered["工作稿"] == ("第1集\n场景1-1：店内，日，内\n人物：奶奶\n"
                               "△ 奶奶接起电话。\n奶奶（看着奶奶的手机）：妈，桂枝说{{P001}}。")
    assert "{{P001}}" in fusion["工作稿"]


@pytest.mark.parametrize("target", ["人物行", "动作执行者", "动作对象", "括号", "说话人", "说话人前缀"])
def test_alias_drift_is_reported_anywhere_outside_dialogue(grandma, target):
    speech, vision, fusion, _ = grandma
    if target == "人物行":
        fusion["工作稿"] = fusion["工作稿"].replace("人物：奶奶", "人物：穿碎花衬衫的白发女子")
    elif target == "动作执行者":
        fusion["工作稿"] += "\n△ 穿碎花衬衫的白发女子接起电话。"
    elif target == "动作对象":
        fusion["工作稿"] += "\n△ 奶奶把手机递给桂枝。"
    elif target == "括号":
        fusion["工作稿"] = fusion["工作稿"].replace("奶奶：", "奶奶（看着桂枝的手机）：")
    elif target == "说话人":
        fusion["工作稿"] = fusion["工作稿"].replace("奶奶：", "桂枝：")
    else:
        fusion["工作稿"] = fusion["工作稿"].replace("奶奶：", "奶奶的妈妈：")
    problems = mvp.fixed_checks(fusion, speech, vision, 2)
    assert any(p["类型"] == "人物指代与角色卡" for p in problems)


def test_dialogue_and_title_cards_keep_original_aliases(grandma):
    speech, vision, fusion, _ = grandma
    text = "妈，桂枝，穿碎花衬衫的白发女子来了。"
    speech["句列表"][0]["文本"] = text
    fusion["逐句归属"][0]["文本"] = text
    fusion["工作稿"] = fusion["工作稿"].replace("你好。", text)
    fusion["工作稿"] += "\n△ 画面字幕：桂枝"
    assert mvp.fixed_checks(fusion, speech, vision, 2) == []


def test_longer_registered_names_are_not_mistaken_for_aliases(grandma):
    speech, vision, fusion, _ = grandma
    fusion["人物"].append({**fusion["人物"][0], "id": "P002", "名称": "妈妈", "视觉人物": [], "别称": []})
    fusion["工作稿"] += "\n△ 妈妈把手机递给奶奶。"
    assert mvp.fixed_checks(fusion, speech, vision, 2) == []


def test_alias_containing_the_registered_name_is_still_drift(grandma):
    speech, vision, fusion, _ = grandma
    fusion["人物"][0]["别称"].append("白发奶奶")
    fusion["工作稿"] += "\n△ 白发奶奶接起电话。"
    assert any(p["类型"] == "人物指代与角色卡" for p in mvp.fixed_checks(fusion, speech, vision, 2))


def test_confirmed_relation_name_is_compatible_with_existing_script_checks():
    from s2s.lint_script import lint_text

    text = "第1集\n场景1-1：走廊，日，内\n人物：奶奶\n△ 奶奶接起电话。\n奶奶：妈，桂枝来了。"
    roles = [{"id": "C001", "名称": "奶奶", "名称类型": "描述名", "别名称呼": ["妈", "桂枝"]}]
    assert lint_text(text, roles)["通过"]


def test_same_visual_person_cannot_be_split_into_two_identities(grandma):
    speech, vision, fusion, _ = grandma
    fusion["人物"].append({**fusion["人物"][0], "id": "P002", "名称": "白发女子"})
    problems = mvp.fixed_checks(fusion, speech, vision, 2)
    assert any("重复映射" in p["依据"] for p in problems)


def test_fusion_inputs_keep_sentences_and_issues(sample):
    speech, vision, _, _ = sample
    speech["疑似漏段"] = [{"起": 1.8, "止": 1.9}]
    speech["句列表"][0]["核对"] = {"状态": "待核对", "原始文本": "你好。", "异常": ["截断"], "优先级": 1}
    speech["待核对"] = [{"位置": {"句": "E01-S001"}, "依据": "截断", "修正要求": "回听"}]
    before = deepcopy(speech)
    names = [{"视觉人物": "T001", "名称": "奶奶", "来源": "人工确认"}]
    inputs = mvp.fusion_inputs(1, 2, speech, vision, names)
    compact = inputs["语音"]
    assert set(compact) == {"句列表", "疑似漏段", "待核对"}
    assert compact["句列表"] == [{k: before["句列表"][0][k] for k in ("id", "起", "止", "文本")} | {"异常": ["截断"]}]
    assert compact["疑似漏段"] == before["疑似漏段"]
    assert compact["待核对"] == before["待核对"]
    assert inputs["视觉"] == vision and inputs["已确认人物命名"] == names
    assert speech == before


def test_fusion_prompt_contains_one_consistent_set_of_script_rules():
    prompt = (ROOT / "prompts/mvp/融合写稿.md").read_text()
    for rule in ("逐字使用一次", "合成一行", "{{P001}}", "场景", "人物：", "△", "画外", "电话音", "对着电话",
                 "关系称谓", "待核对", "字卡", "画面人物"):
        assert rule in prompt
    assert "声纹" not in prompt and "声线" not in prompt
    for obsolete in ("遵守附带的成稿内容规则", "可以合成一行", "交回主控", "主控确认的 `排布`"):
        assert obsolete not in prompt


def test_unknown_template_person_is_preserved_and_reported(grandma):
    speech, vision, fusion, _ = grandma
    fusion["工作稿"] += "\n△ {{P999}}走进店里。"
    fusion = mvp.render_script(fusion)
    assert "{{P999}}" in fusion["工作稿"]
    assert any("人物引用" in p["依据"] for p in mvp.fixed_checks(fusion, speech, vision, 2))


def tagged(fusion):
    fusion["逐句归属"] = [{k: v for k, v in r.items() if k != "行"} for r in fusion["逐句归属"]]
    fusion["采用动作"] = [{"动作": "A001", "人物": "P001", "对象人物": [], "起": .5, "止": 1.5,
                         "表述": "推门", "依据": "画面可见"}]
    fusion["工作稿"] = ("第1集\n场景1-1：走廊，日，内\n人物：穿白衬衫的短发女子\n"
                       "〔E01-S001〕〔A001〕穿白衬衫的短发女子（推门）：你好。")
    return fusion


def test_line_tags_are_stripped_and_become_line_numbers(sample):
    speech, vision, fusion, _ = sample
    vision["动作"] = [{"id": "A001"}]
    located, problems = mvp.locate_lines(tagged(fusion))
    assert problems == []
    assert located["工作稿"].splitlines()[3] == "穿白衬衫的短发女子（推门）：你好。"
    assert located["逐句归属"][0]["行"] == 4 and located["采用动作"][0]["行"] == 4
    assert mvp.fixed_checks(located, speech, vision, 2) == []


@pytest.mark.parametrize("change", ["unknown", "duplicate", "missing"])
def test_line_tag_errors_are_reported(sample, change):
    speech, vision, fusion, _ = sample
    vision["动作"] = [{"id": "A001"}]
    fusion = tagged(fusion)
    if change == "unknown":
        fusion["工作稿"] += "\n〔A999〕△ 穿白衬衫的短发女子关门。"
    elif change == "duplicate":
        fusion["工作稿"] += "\n〔A001〕△ 穿白衬衫的短发女子关门。"
    else:
        fusion["工作稿"] = fusion["工作稿"].replace("〔E01-S001〕", "")
    located, problems = mvp.locate_lines(fusion)
    assert "〔" not in located["工作稿"]
    assert problems + mvp.fixed_checks(located, speech, vision, 2)


def test_stream_saves_reasoning_and_rejects_length(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    usage = {"completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 3}}
    install_http(monkeypatch, lambda request: reply(json.loads(request.content), {"model": mvp.DEEPSEEK_MODEL,
        "usage": usage, "choices": [{"finish_reason": "length", "message": {"content": ""}}]}))
    with pytest.raises(ValueError, match="length"):
        run.request("融合写稿", mvp.DEEPSEEK_MODEL, [])
    row = run.record["调用"]["融合写稿"]
    assert row["finish_reason"] == "length" and row["usage"] == usage and row["响应模型"] == mvp.DEEPSEEK_MODEL
    assert row["端点"] == mvp.DEEPSEEK_ENDPOINT and row["参数"]["max_tokens"] == 32768
    assert (run.artifacts / "融合写稿-推理.txt").read_text() == "先想一想。"
    assert "[DONE]" in (run.artifacts / "融合写稿-原始响应.txt").read_text()


def test_stream_idle_timeout_stops_without_retry(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("idle", request=request)

    install_http(monkeypatch, handler)
    with pytest.raises(ValueError, match="未收到新数据"):
        run.request("融合写稿", mvp.PRO_MODEL, [])
    assert len(requests) == 1


def test_pending_marks_are_removed_from_script(grandma):
    _, _, fusion, _ = grandma
    fusion["工作稿"] += "\n奶奶：接到了啊。【待核对 Q002】【待核对 Q004】"
    assert mvp.render_script(fusion)["工作稿"].splitlines()[-1] == "奶奶：接到了啊。"


def test_long_speaker_bracket_is_listed(sample):
    speech, vision, fusion, _ = sample
    short = fusion["工作稿"].replace("女子：", "女子（拍桌，提高音量）：")
    assert mvp.fixed_checks({**fusion, "工作稿": short}, speech, vision, 2) == []
    long = fusion["工作稿"].replace("女子：", "女子（从口袋取出手机，手指滑动屏幕查找联系人）：")
    problems = mvp.fixed_checks({**fusion, "工作稿": long}, speech, vision, 2)
    assert [p["级别"] for p in problems] == ["规范"] and "括号" in problems[0]["依据"]


def two_sentences(sample, second_person="P001"):
    speech, vision, fusion, _ = sample
    speech["句列表"].append({"id": "E01-S002", "起": "00:01.5", "止": "00:02.0", "文本": "吃了吗？"})
    fusion["人物"].append({**fusion["人物"][0], "id": "P002", "名称": "穿黑外套的老人", "视觉人物": ["T002"]})
    vision["人物"].append({"id": "T002"})
    fusion["逐句归属"].append({**fusion["逐句归属"][0], "句": "E01-S002", "人物": second_person,
                             "起": "00:01.5", "止": "00:02.0", "文本": "吃了吗？"})
    fusion["工作稿"] = fusion["工作稿"].replace("你好。", "你好。吃了吗？")
    return speech, vision, fusion


def test_consecutive_sentences_of_one_speaker_merge_into_one_line(sample):
    speech, vision, fusion = two_sentences(sample)
    assert mvp.fixed_checks(fusion, speech, vision, 3) == []


def test_merged_line_must_keep_one_speaker_and_exact_text(sample):
    speech, vision, fusion = two_sentences(deepcopy(sample), "P002")
    assert "阻断" in {p["级别"] for p in mvp.fixed_checks(fusion, speech, vision, 3)}
    speech, vision, fusion = two_sentences(sample)
    fusion["工作稿"] = fusion["工作稿"].replace("你好。吃了吗？", "你好。")
    assert any(p["依据"] == "正文台词与原句不一致" for p in mvp.fixed_checks(fusion, speech, vision, 3))


def subtitle(start, end, text, confidence=1.0, kind="台词字幕"):
    return {"起": start, "止": end, "文字": text, "类型": kind, "置信度": confidence}


def test_subtitles_fix_homophones_only():
    sounds = [{"id": "E01-S001", "起": "00:00.0", "文本": "桂枝，都四点四十多了，你还坐这儿啊？"},
              {"id": "E01-S002", "起": "00:03.0", "文本": "他喝着东西呢。"}]
    records = [subtitle("00:00.0", "00:02.0", "桂芝都4:40多了"), subtitle("00:02.0", "00:03.0", "你还坐这啊"),
               subtitle("00:03.0", "00:04.0", "她喝着东西呢"), subtitle("00:03.0", "00:03.5", "她喝着乐西呢", .3),
               subtitle("00:00.0", "00:09.0", "棋牌室", kind="地点字幕")]
    fixes = mvp.subtitle_fixes(sounds, records)
    assert [s["文本"] for s in sounds] == ["桂芝，都四点四十多了，你还坐这儿啊？", "她喝着东西呢。"]
    assert fixes == [{"句": "E01-S001", "时间": "00:00.0", "转写": "桂枝，都四点四十多了，你还坐这儿啊？",
                      "字幕": "桂芝，都四点四十多了，你还坐这儿啊？"},
                     {"句": "E01-S002", "时间": "00:03.0", "转写": "他喝着东西呢。", "字幕": "她喝着东西呢。"}]


def test_long_subtitle_difference_keeps_transcript():
    sounds = [{"id": "E01-S001", "起": "00:00.0", "文本": "你今天回不回来吃饭？"}]
    assert mvp.subtitle_fixes(sounds, [subtitle("00:00.0", "00:02.0", "你明晚回不回家吃饭")]) == []
    assert sounds[0]["文本"] == "你今天回不回来吃饭？"


def test_subtitle_fixes_are_listed_without_blocking(sample):
    speech, vision, fusion, review = sample
    speech["字幕校字"] = [{"句": "E01-S001", "时间": "00:00.5", "转写": "你号。", "字幕": "你好。"}]
    result = mvp.final_review(review, fusion, speech, vision, [])
    assert result["结论"] == "通过（附问题清单）"
    assert result["问题清单"][0]["来源"] == "字幕校字" and "你号。 → 你好。" in result["问题清单"][0]["依据"]
