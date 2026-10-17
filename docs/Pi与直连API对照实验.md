# Pi 与 DeepSeek 直连 API 对照实验

状态：**实验工具与离线校验已落地；没有在 GitHub CI 或当前会话发起真实模型调用，也没有生成性能结论。** 实际推理必须在有 Pi、帧图片和 OpenCode Go 凭据的开发机上使用 `--live` 显式启用。

## 为什么比较

现生产子角色使用 Pi + `opencode-go/deepseek-v4.1-flash`。这是一条 OpenCode Go 接入路径，**不等于** DeepSeek 官方 `api.deepseek.com` 的某个模型。为比较客户端开销，三组都使用同一 OpenCode Go 模型 ID，并尽可能匹配思考挡位、输入帧和提示词：

| 方案 | 图片提供方法 | 工具权限 |
|---|---|---|
| `pi-read` | 通过 Pi 受控 `read` 按白名单读取 | 只开放 `read`，禁止 bash/edit/write、MCP、扩展自动加载 |
| `pi-attach` | Pi 的 `@image` 附件输入 | 无工具 |
| `direct` | Python httpx 调 OpenAI 兼容 Chat Completions，base64 图像 | 无工具 |
| `direct-tuned` | 同 direct，但换成压缩后的提示词 | 无工具 |

**注意：** 当前子角色 `dispatch/transport.py` 已非常精简，非控制角色本来就没有 bash/edit/write。真正可能减少的主要是 Pi 读取图片时的工具轮次和额外上下文；文本成稿角色已是无工具 Pi，用直连测更有意义。全剧/单集主控仍有工具操作，不能直接把无工具模式套过去。

## 接口与能力

- OpenCode Go 官方当前文档给出 `https://opencode.ai/zen/go/v1/chat/completions`，模型 `deepseek-v4.1-flash`。
- Pi 模型目录展示过 `https://opencode.ai/inference/go/v1` 等实际基础地址。运行前须对照本机已安装 Pi 的模型信息及授权，确认 `--endpoint` 是**同一上游**；不能以 URL 不同的结果直接推断 Pi 有额外开销。
- Pi 模型目录将该模型定义为多模态并支持 high；直连脚本使用 `thinking.type=enabled` 和 `reasoning_effort=high`，但仍应在首个最小请求中确认网关兼容性，不能假定服务端接受并执行相同推理预算。
- 直接 API 需要同服务的 `OPENCODE_API_KEY` 环境变量。**不自动读取 `~/.pi/agent/auth.json`**，也不上传密钥或 Pi 会话数据到仓库。
- 所有结果保存在本机 `tmp/benchmark-dp/`；运行前避免把真实敏感视频素材输出提交 Git。
- **未指定 `--live` 时**：只做离线参数、文件与请求体预检，不发模型请求。

## 运行方式（在项目根目录）

首先在本机准备相同的测试材料：
- 一个**实际渲染、无 `{{占位}}` 的原版提示词**，例如单段动作识别派发时的 prompt 快照；
- 该段按时间顺序排列的 8～12 张图片（少量关键片段代表样本，不以抽样代替整集覆盖）；
- 如果测结构化返回，`--schema` 指向对应的 JSON Schema，要求两端输出完全相同的格式。

不调用模型的预检：

~~~bash
uv run python tools/benchmark_dp.py \
  --prompt tmp/bench-input/动作识别-P01.txt \
  --image tmp/bench-input/K001_01.jpg \
  --image tmp/bench-input/K001_02.jpg \
  --variants pi-read,pi-attach,direct \
  --repeat 3
~~~

在**本机**验证同上游授权后进行真实测试（会消耗额度）：

~~~bash
export OPENCODE_API_KEY="通过本地环境/安全凭据工具注入的值"
uv run python tools/benchmark_dp.py \
  --prompt tmp/bench-input/动作识别-P01.txt \
  --image tmp/bench-input/K001_01.jpg \
  --image tmp/bench-input/K001_02.jpg \
  --variants pi-read,pi-attach,direct \
  --repeat 3 \
  --thinking high \
  --schema tmp/bench-input/动作识别.schema.json \
  --output tmp/benchmark-dp/action-P01 \
  --live
~~~

**Schema 注意：** `role_schemas.json` 是按角色索引的聚合文件，而不是单个角色的 JSON Schema，实际测试应抽取对应角色 Schema 保存到 `tmp/bench-input/动作识别.schema.json` 后作为 `--schema` 传入；否则校验结果无意义。

测纯文本角色（如成稿）时不要使用 `pi-read`，可只传 `--variants pi-attach,direct`，不传 `--image`。

压缩提示词实验先保持输出 Schema 和全部质量要求不变，再加上：

~~~bash
--optimized-prompt tmp/bench-input/动作识别-精简.txt \
--variants direct,direct-tuned
~~~

## 比较口径

将实验拆成两个问题，避免把效率差异错误归因于模型或 Prompt：

1. **只换传输 / Harness，Prompt 不变：** 比较 `pi-read`、`pi-attach`、`direct`。记录耗时、返回 usage、工具读取次数、JSON 可解析率和 Schema 通过率；顺序每轮轮转。
2. **只改 Prompt，直连不变：** 比较 `direct`、`direct-tuned`。只能删除重复解释，不能删除事实证据规则、身份冲突处理、读帧要求和输出字段。以实际人工真值判定是否改善。

Pi 的工具读取模式和直传 API 的图像计费/缓存命中可能不同，所以需要**同时统计有效读取帧数与官方 usage**；不能只用总 token 或单轮耗时认定胜负。当前脚本不自动计算金额，账单费用应按实际网关结算口径另核。

质量需要建立人工真值：关键动作召回、无据动作、执行者/对象错配、说话人错归、短动作括号位置、全文称呼一致性。JSON 通过只是格式检查，不等于视觉识别正确。推荐 E01 跨 6～10 个困难片段、每种重复至少 3 次，再做全片验证。**不能通过删掉难动作换取更高通过率。**

## 决策条件

- 纯文本或少量帧任务：直连若同等质量且耗时更低，可直接替换子角色调用器，并保留现有 `assemble`、Schema 校验、缓存、日志、重试；
- 需要大量按需补帧的视觉角色：若 `pi-read` 的漏动作更少，继续 Pi 只读工具或者新增由 Python 自动补帧的直连模式；
- 主控角色：先保留 Pi，并考虑把 `read,bash,edit,write` 缩减为显式业务工具接口；不得让模型随意写公共角色卡；
- 正式更换生产路线之前，完成端到端模拟集成和真实素材回归，保留原 Pi 路线作为可回退的受控选项。

## 当前运行范围

本次新增的 `tests/test_benchmark_dp.py` 只使用图片夹具和 `httpx.MockTransport`，**不进行网络请求**；验证相同模型、鉴权头、图片载荷、未知模型拒绝、无密钥拒绝以及预检默认不调用。
