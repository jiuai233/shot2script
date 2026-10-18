"""统一的 pi / opencode / DeepSeek 直连非交互调用接口。"""

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from .config import check_model, get_environment, get_runner_config


# 显式系统提示词覆盖 pi 自动读取的 ~/.pi/agent/SYSTEM.md。
DEFAULT_SYSTEM = "你是短剧逐镜头转写流程中的助手，严格按用户消息中的要求输出。"


MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
TRANSIENT = {429, 500, 502, 503, 504}


class RunnerError(RuntimeError):
    """运行器启动、执行或输出失败。"""


def _safe(value: str | bytes | None, env: dict[str, str]) -> str:
    text = value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
    for name, secret in env.items():
        if len(secret) >= 4 and any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PROXY")):
            text = text.replace(secret, "[已隐藏]")
    return text


def run(
    prompt: str,
    images: list[str | Path] | None = None,
    files: list[str | Path] | None = None,
    *,
    model: str,
    thinking: str,
    tools: bool = False,
    system: str = DEFAULT_SYSTEM,
    labels: list[str] | None = None,
    events: bool = False,
    runner: str = "pi",
    timeout: float = 120,
    bin: str | None = None,
    request_budget=None,
) -> str:
    if runner not in {"pi", "opencode", "deepseek"}:
        raise RunnerError(f"未知运行器：{runner}；模型={model}；退出码=未启动；stderr(末20行)=无")
    env = get_environment()

    def failure(reason, code="未启动", stderr=""):
        tail = "\n".join(_safe(stderr, env).splitlines()[-20:]) or "无"
        return RunnerError(_safe(f"运行器={runner}；模型={model}；退出码={code}；{reason}\nstderr(末20行)：\n{tail}", env))

    attachments = []
    for filename in [*(images or []), *(files or [])]:
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            raise failure(f"附件不存在：{path}")
        attachments.append(str(path))
    if not model or not thinking or not system or timeout <= 0:
        raise failure("model、thinking、system 必须非空，timeout 必须大于 0")
    check_model(model)
    if request_budget is not None and runner != 'deepseek':
        raise failure('固定预算只支持可逐HTTP请求记账的DeepSeek直连；不能把Pi子进程当一次模型请求')
    if runner == "deepseek":
        if files or tools:
            raise failure("deepseek 直连只接受图片，不开放工具和文件附件")
        if labels is not None and len(labels) != len(attachments):
            raise failure("图片标注数量须与图片一致")
        return _deepseek(prompt, attachments, labels, model=model, thinking=thinking, system=system,
                         timeout=timeout, events=events, env=env, failure=failure, request_budget=request_budget)
    if labels:
        raise failure("图片标注只用于 deepseek 直连")
    executable = bin or get_runner_config(runner).get("bin", runner)
    if runner == "pi":
        command = [executable, "-p", "--no-session", "--no-context-files", "--model", model, "--thinking", thinking,
                   "--system-prompt", system]
        if not tools:
            command.append("--no-tools")
        # -- 防止以短横线开头的提示词被当作命令行选项。
        command.extend(["--", *(f"@{path}" for path in attachments), prompt])
    else:
        command = [executable, "run", "--format", "json", "--model", model]
        if thinking != "off":
            command.extend(["--variant", thinking])
        for path in attachments:
            command.extend(["--file", path])
        command.extend(["--", prompt])
        if not tools:
            try:
                inline = json.loads(env.get("OPENCODE_CONFIG_CONTENT") or "{}")
                # 全局 deny 使用标量，避免遗留的逐工具 allow 覆盖通配符。
                inline["permission"] = "deny"
                env["OPENCODE_CONFIG_CONTENT"] = json.dumps(inline)
            except (ValueError, TypeError) as exc:
                raise failure("OPENCODE_CONFIG_CONTENT 必须是 JSON 对象") from exc
    try:
        # 不继承上层标准输入：pi 在非终端输入上会等待读完，上层管道不关闭时会一直挂起。
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise failure(f"调用超时（{timeout} 秒）", "超时", exc.stderr) from exc
    except OSError as exc:
        raise failure(f"无法启动：{exc}") from exc
    if result.returncode:
        raise failure("调用失败", result.returncode, result.stderr)
    output = result.stdout.strip()
    if runner == "opencode" and output:
        parts = []
        try:
            for line in output.splitlines():
                event = json.loads(line)
                if event.get("type") == "error":
                    raise failure(f"接口错误：{json.dumps(event.get('error', {}), ensure_ascii=False)}", result.returncode, result.stderr)
                if event.get("type") == "text":
                    parts.append(event["part"]["text"])
            output = "".join(parts).strip()
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise failure("无法解析 opencode JSON 事件", result.returncode, result.stderr) from exc
    if not output:
        raise failure("空输出", result.returncode, result.stderr)
    return output


def _deepseek(prompt, images, labels, *, model, thinking, system, timeout, events, env, failure, request_budget=None):
    """一次请求送入全部图片；每张图前可插一行标注。events=True 时返回 pi 格式的 message_end 事件。"""
    settings = get_runner_config("deepseek")
    key_name = settings.get("api_key_env", "DEEPSEEK_API_KEY")
    if not env.get(key_name):
        raise failure(f"缺少 {key_name}")
    content = [{"type": "text", "text": prompt}]
    for index, path in enumerate(images):
        mime = MIME.get(Path(path).suffix.lower())
        if mime is None:
            raise failure(f"不支持的图片格式：{path}")
        if labels:
            content.append({"type": "text", "text": labels[index]})
        data = base64.b64encode(Path(path).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    body = {"model": model.split("/", 1)[1],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "thinking": {"type": "disabled" if thinking == "off" else "enabled"}}
    if thinking != "off":
        body["reasoning_effort"] = thinking
    request = urllib.request.Request(
        settings["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {env[key_name]}", "Content-Type": "application/json"})
    for attempt in range(3):
        ticket = request_budget.begin({'模型': model, '图片数': len(images), 'HTTP尝试': attempt + 1,
                                       '重试原因': '瞬态HTTP或网络错误' if attempt else None,
                                       '请求SHA256': hashlib.sha256(request.data).hexdigest()}) if request_budget else None
        response_usage, request_error = None, None
        try:
            remaining = min(timeout, request_budget.remaining_seconds()) if request_budget else timeout
            with urllib.request.urlopen(request, timeout=remaining) as reply:
                result = json.loads(reply.read())
            response_usage = result.get('usage')
            break
        except urllib.error.HTTPError as exc:
            request_error = 'HTTP ' + str(exc.code)
            if exc.code not in TRANSIENT or attempt == 2:
                raise failure("接口错误", exc.code, exc.read()) from None
        except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
            request_error = type(exc).__name__
            # 超时不重试，避免一次派发等待数倍时长。
            if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError):
                raise failure(f"调用超时（{timeout} 秒）", "超时") from None
            if attempt == 2:
                raise failure(f"网络错误：{exc}") from None
        except ValueError as exc:
            request_error = '响应JSON无效'
            raise failure("无法解析接口响应") from exc
        finally:
            if request_budget is not None:
                request_budget.finish(ticket, usage=response_usage, error=request_error)
        time.sleep(2 * 2 ** attempt)
    try:
        choice = result["choices"][0]
        text = (choice["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise failure("接口响应缺少正文") from exc
    if choice.get("finish_reason") == "length":
        raise failure("输出达到长度上限被截断")
    if not text:
        raise failure("空输出")
    if not events:
        return text
    usage = result.get("usage") or {}
    normalized = {}
    for source, target in (('prompt_tokens', 'promptTokens'), ('prompt_cache_hit_tokens', 'cacheRead'),
                           ('completion_tokens', 'output'), ('total_tokens', 'totalTokens')):
        if isinstance(usage.get(source), (int, float)):
            normalized[target] = usage[source]
    if 'promptTokens' in normalized and 'cacheRead' in normalized:
        normalized['input'] = normalized.pop('promptTokens') - normalized['cacheRead']
    reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
    if isinstance(reasoning, (int, float)):
        normalized['reasoning'] = reasoning
    message = {"role": "assistant", "content": [{"type": "text", "text": text}], "stopReason": "stop",
               "usage": normalized}
    return json.dumps({"type": "message_end", "message": message}, ensure_ascii=False)
