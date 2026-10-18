#!/usr/bin/env python3
"""Compare Pi (read/attach) with direct OpenCode Go; no live calls without --live."""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import httpx

DEFAULT_URL = "https://opencode.ai/zen/go/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"
PI_MODEL = f"opencode-go/{MODEL}"
ALLOWED_VARIANTS = ("pi-read", "pi-attach", "direct", "direct-tuned")
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
MAX_REQUEST_SIZE = 48 * 1024 * 1024


@dataclass
class Trial:
    variant: str
    elapsed: float
    text: str
    usage: dict[str, Any]
    tool_calls: int = 0
    read_calls: int = 0


def fingerprint(prompt: str, images: list[Path]) -> str:
    h = hashlib.sha256(prompt.encode("utf-8"))
    for path in images:
        h.update(path.read_bytes())
    return h.hexdigest()[:20]


def image_block(path: Path) -> dict[str, Any]:
    kind = MIME.get(path.suffix.lower())
    if not kind or not path.is_file():
        raise ValueError(f"不是受支持的图片文件：{path}")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{kind};base64,{b64}", "detail": "high"}}


def make_payload(prompt: str, images: list[Path], *, model: str = MODEL, reasoning: str = "high", max_tokens: int = 8192) -> dict[str, Any]:
    if model != MODEL:
        raise ValueError("仅允许同一个 DeepSeek 模型，防止混淆客户端与模型效果")
    if reasoning not in ("off", "low", "high", "max") or max_tokens < 1:
        raise ValueError("不支持的思考级别或 max_tokens")
    content: str | list[dict[str, Any]] = prompt
    if images:
        content = [{"type": "text", "text": prompt}, *[image_block(p) for p in images]]
    payload: dict[str, Any] = {
        "model": model, "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "stream": False,
        "thinking": {"type": "disabled" if reasoning == "off" else "enabled"},
    }
    if reasoning != "off":
        payload["reasoning_effort"] = reasoning
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_REQUEST_SIZE:
        raise ValueError("请求体超过 48 MiB，请减少图片")
    return payload


def direct_call(prompt: str, images: list[Path], *, url: str, api_key: str,
                reasoning: str, max_tokens: int, timeout: float,
                transport: httpx.BaseTransport | None = None) -> Trial:
    if not api_key:
        raise ValueError("直连需设置 OPENCODE_API_KEY；不会读取 Pi 登录凭据")
    from urllib.parse import urlparse
    parsed = urlparse(url)
    same_provider = parsed.scheme == "https" and parsed.hostname == "opencode.ai" and parsed.path.endswith("/chat/completions")
    local_mock = parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port is not None
    if not (same_provider or local_mock):
        raise ValueError("端点必须为 opencode.ai 的 HTTPS Chat Completions，或 127.0.0.1 本地测试地址")
    payload = make_payload(prompt, images, reasoning=reasoning, max_tokens=max_tokens)
    started = time.perf_counter()
    with httpx.Client(timeout=timeout, transport=transport) as client:
        response = client.post(url, json=payload, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"
        })
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(f"API HTTP {response.status_code}；不记录服务端错误体，防止秘密泄露")
    data = response.json()
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0].get("message", {}).get("content"), str):
        raise ValueError("API 未返回文本内容")
    return Trial("direct", elapsed, choices[0]["message"]["content"], data.get("usage") or {})


def pi_call(variant: str, prompt: str, images: list[Path], *, model: str, reasoning: str, timeout: float) -> Trial:
    from s2s import runners
    from s2s.dispatch.transport import pi_options, decode_response
    settings = {"runner": "pi", "model": model, "thinking": reasoning}
    started = time.perf_counter()
    if variant == "pi-read":
        enriched = prompt + "\n\n请使用 read 工具按顺序读取下列全部图片并依据图片作答：\n" + "\n".join(str(p) for p in images)
        with pi_options(settings, read_paths=images) as opts:
            raw = runners.run(enriched, **{**settings, **opts, "timeout": timeout})
        out = decode_response(raw)
        tools = sum(1 for line in raw.splitlines()
                    if '"type":"tool_execution_start"' in line or '"type": "tool_execution_start"' in line)
        return Trial(variant, time.perf_counter()-started, out.text, out.usage, tools, len(out.reads))
    if variant == "pi-attach":
        with pi_options(settings) as opts:
            raw = runners.run(prompt, images=images, **{**settings, **opts, "timeout": timeout})
        out = decode_response(raw)
        return Trial(variant, time.perf_counter()-started, out.text, out.usage)
    raise ValueError(f"未知 Pi 变体：{variant}")


def result_checks(answer: str, *, expected_schema: dict | None) -> dict[str, Any]:
    try:
        value = json.loads(answer)
    except (TypeError, ValueError):
        return {"json_valid": False, "schema_valid": None}
    if expected_schema is None:
        return {"json_valid": True, "schema_valid": None}
    import jsonschema
    return {"json_valid": True, "schema_valid": jsonschema.Draft202012Validator(expected_schema).is_valid(value)}


def options(argv=None):
    p = argparse.ArgumentParser(description="同一 DeepSeek 模型：Pi read / Pi attach / 直连 API 对照")
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--optimized-prompt", type=Path)
    p.add_argument("--image", type=Path, action="append", default=[])
    p.add_argument("--variants", default="pi-read,pi-attach,direct")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--thinking", choices=("off","low","high","max"), default="high")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--endpoint", default=DEFAULT_URL)
    p.add_argument("--schema", type=Path)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--output", type=Path, default=Path("tmp/benchmark-dp"))
    p.add_argument("--live", action="store_true", help="显式开启才调用真实模型")
    args = p.parse_args(argv)
    args.variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    if (not args.variant_names or len(set(args.variant_names)) != len(args.variant_names)
            or any(v not in ALLOWED_VARIANTS for v in args.variant_names)):
        p.error("--variants 包含未知或重复变体")
    if "direct-tuned" in args.variant_names and args.optimized_prompt is None:
        p.error("direct-tuned 必须传 --optimized-prompt")
    if "pi-read" in args.variant_names and not args.image:
        p.error("pi-read 必须传入图片，纯文本请用 pi-attach,direct")
    if args.repeat < 1 or args.repeat > 20 or args.timeout <= 0:
        p.error("repeat 需在 1～20 之间，timeout > 0")
    if len(args.image) > 12:
        p.error("最多输入 12 帧以保证 Pi 附件模式和直调可比较")
    return args


def main(argv=None) -> int:
    args = options(argv)
    prompt = args.prompt.read_text(encoding="utf-8")
    tuned = args.optimized_prompt.read_text(encoding="utf-8") if args.optimized_prompt else None
    images = [x.resolve() for x in args.image]
    schema = json.loads(args.schema.read_text(encoding="utf-8")) if args.schema else None
    make_payload(prompt, images, reasoning=args.thinking, max_tokens=args.max_tokens)
    if tuned is not None:
        make_payload(tuned, images, reasoning=args.thinking, max_tokens=args.max_tokens)
    manifest = {
        "model": PI_MODEL, "endpoint": args.endpoint, "thinking": args.thinking,
        "image_count": len(images), "prompt_chars": len(prompt),
        "base_fingerprint": fingerprint(prompt, images),
        "tuned_fingerprint": fingerprint(tuned, images) if tuned is not None else None,
        "variants": args.variant_names, "repeat": args.repeat, "live": args.live,
        "note": "JSON 或 Schema 通过不等于动作、人物或对白真实",
    }
    if not args.live:
        print(json.dumps({"mode": "offline-preflight", **manifest}, ensure_ascii=False, indent=2))
        return 0
    if any(v.startswith("direct") for v in args.variant_names) and not os.environ.get("OPENCODE_API_KEY"):
        raise SystemExit("直连缺少 OPENCODE_API_KEY；未发起任何模型请求")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    rows = []
    for n in range(args.repeat):
        variants = args.variant_names[n % len(args.variant_names):] + args.variant_names[:n % len(args.variant_names)]
        for variant in variants:
            selected = tuned if variant == "direct-tuned" else prompt
            label = f"{variant}-{n+1:02d}"
            try:
                if variant.startswith("pi-"):
                    t = pi_call(variant, selected, images, model=PI_MODEL, reasoning=args.thinking, timeout=args.timeout)
                else:
                    t = direct_call(selected, images, url=args.endpoint,
                                    api_key=os.environ.get("OPENCODE_API_KEY", ""),
                                    reasoning=args.thinking, max_tokens=args.max_tokens, timeout=args.timeout)
                    t.variant = variant
                path = args.output / f"{label}.txt"
                path.write_text(t.text, encoding="utf-8")
                row = {
                    "variant": variant, "repeat": n+1, "success": True, "elapsed_sec": round(t.elapsed, 3),
                    "usage": t.usage, "tool_calls": t.tool_calls, "read_calls": t.read_calls,
                    "chars_out": len(t.text), "output_path": str(path),
                    **result_checks(t.text, expected_schema=schema),
                }
            except (OSError, RuntimeError, ValueError, httpx.HTTPError) as exc:
                key = os.environ.get("OPENCODE_API_KEY", "")
                message = str(exc).replace(key, "[redacted]") if key else str(exc)
                row = {"variant": variant, "repeat": n+1, "success": False, "error": message}
            rows.append(row)
            (args.output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    summary = []
    for variant in args.variant_names:
        ok = [x for x in rows if x["variant"] == variant and x["success"]]
        summary.append({
            "variant": variant, "success": len(ok), "total": args.repeat,
            "median_sec": round(statistics.median(x["elapsed_sec"] for x in ok),3) if ok else None,
            "json_valid": sum(x["json_valid"] for x in ok),
            "schema_valid": sum(x["schema_valid"] is True for x in ok) if schema is not None else None
        })
    print(json.dumps({"summary": summary, "reports": str(args.output)}, ensure_ascii=False, indent=2))
    return 0 if all(x["success"] for x in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
