"""统一的 pi / opencode 非交互调用接口。"""

import json
from pathlib import Path
import subprocess

from .config import check_model, get_environment, get_runner_config


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
    runner: str = "pi",
    timeout: float = 120,
    bin: str | None = None,
) -> str:
    if runner not in {"pi", "opencode"}:
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
    if not model or not thinking or timeout <= 0:
        raise failure("model、thinking 必须非空，timeout 必须大于 0")
    check_model(model)
    executable = bin or get_runner_config(runner).get("bin", runner)
    if runner == "pi":
        command = [executable, "-p", "--no-session", "--no-context-files", "--model", model, "--thinking", thinking]
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
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
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
