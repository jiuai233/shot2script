import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { realpathSync } from "node:fs";
import { resolve } from "node:path";

export default function (pi: ExtensionAPI) {
  const allowed = new Set<string>(JSON.parse(process.env.S2S_DISPATCH_READ_PATHS || "[]"));
  pi.on("tool_call", async (event, ctx) => {
    if (event.toolName !== "read") return { block: true, reason: "子 agent 只允许读取授权图片" };
    try {
      const path = realpathSync(resolve(ctx.cwd, String(event.input.path)));
      if (allowed.has(path)) return;
    } catch {}
    return { block: true, reason: "该路径不在本次派发的图片白名单内" };
  });
}
