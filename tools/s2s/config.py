"""读取模型配置；每次读取都应用当前环境变量，不缓存角色继承结果。"""

from copy import deepcopy
import json
import os
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "models.json"
PRODUCTION_MODEL = "opencode-go/deepseek-v4.1-flash"


class ConfigError(ValueError):
    """配置文件或引用无效。"""


def get_environment() -> dict[str, str]:
    """进程环境优先于项目 .env；返回值可能含密钥，不应写入日志。"""
    local = {k: v for k, v in dotenv_values(PROJECT_ROOT / ".env").items() if v is not None}
    return {**local, **os.environ}


def load_config() -> dict:
    env = get_environment()
    path = Path(env.get("S2S_CONFIG") or DEFAULT_CONFIG).expanduser()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("配置顶层须为对象")
        for section in ("asr", "vision", "agents", "runners"):
            if not isinstance(data.get(section), dict):
                raise ValueError(f"缺少对象配置：{section}")
        overrides = (
            ("S2S_ASR_PROVIDER", data["asr"], "provider"),
            ("S2S_VISION_MODEL", data["vision"], "model"),
            ("S2S_VISION_RUNNER", data["vision"], "runner"),
            ("S2S_AGENT_MODEL", data["agents"].setdefault("default", {}), "model"),
            ("S2S_AGENT_RUNNER", data["agents"]["default"], "runner"),
        )
        for name, target, field in overrides:
            if env.get(name):
                target[field] = env[name]
        for settings in (data["vision"], *data["agents"].values()):
            if "model" in settings:
                check_model(settings["model"])
        return data
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigError(f"无法读取模型配置 {path}：{exc}") from exc


def get_asr_backend(name: str | None = None) -> dict:
    asr = load_config()["asr"]
    provider = name or asr.get("provider")
    try:
        return {**deepcopy(asr["backends"][provider]), "provider": provider}
    except KeyError as exc:
        raise ConfigError(f"未配置 ASR 后端：{provider}") from exc


def check_model(model: str) -> None:
    """生产接入固定；配置文件与单次覆盖不能扩展白名单。"""
    if model != PRODUCTION_MODEL:
        raise ConfigError(f"生产模型只允许 {PRODUCTION_MODEL}，不允许：{model}")


def get_vision() -> dict:
    return deepcopy(load_config()["vision"])


def get_agent(role: str) -> dict:
    data = load_config()
    agents = data["agents"]
    if role not in agents:
        raise ConfigError(f"未配置角色：{role}")
    own = deepcopy(agents[role])
    parent = own.pop("use", None)
    if parent is None:
        return own
    if parent == "vision":
        base = data["vision"]
    elif parent == "default":
        base = agents["default"]
        if base.get("use"):
            raise ConfigError("agents.default 不允许继承其他配置")
    else:
        raise ConfigError(f"角色 {role} 的 use 无效：{parent}，仅支持 vision 或 default")
    return {**deepcopy(base), **own}


def get_runner_config(name: str) -> dict:
    try:
        return deepcopy(load_config()["runners"][name])
    except KeyError as exc:
        raise ConfigError(f"未配置运行器：{name}") from exc
