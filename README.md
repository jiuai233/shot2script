# shot2script

把短剧视频按镜头转写成制作剧本：人物明确、动作具体、对白逐字准确、画面和声音归属正确，经独立验收后交付单集剧本、全剧合并稿和 Word 稿。

产品要求见 `docs/视频逐镜头画声转写PRD.md`，当前待办见 `TODO.md`。

## 流程

```
视频
 ├─ 固定处理流程（本机工具）
 │   镜头切分与取帧 · 两路语音识别与交叉核对 · 按人声切句 · 声纹聚类 · 口型检测 · 字幕识别（仅参考）
 │
 ├─ 人工回听：两路不一致或只有一路识别出的句子，在离线回听页上逐句裁定
 │
 └─ 角色 agent（按集）
     动作识别 → 角色卡（认人、对白归属）→ 成稿 → 独立验收 → 返修（最多 3 轮）→ 正式稿
                                                         └─ 解决不了的进入“待人工裁定”
全剧主控：逐集调度、审核跨集角色变更、合并、生成 Word 稿
```

## 目录

| 路径 | 内容 |
|---|---|
| `docs/` | PRD、任务拆解、各项实测与对比报告 |
| `prompts/` | 各角色提示词与数据约定（`00-数据约定.md`） |
| `tools/` | 命令行工具；`tools/s2s/` 为 Python 包，用法见 `tools/README.md` |
| `config/models.json` | 模型与后端配置，可用环境变量临时覆盖 |
| `orchestrate/` | 开发任务单（`tasks/`）、执行报告（`reports/`）与派发脚本 |
| `eval/` | 人工标注规范、样本标注与评测 |
| `simple/` | Word 排版范本 |
| `tests/` | 单元测试与夹具 |

不入库的本机数据：`videos/`（视频）、`剧目/`（黑板运行数据，含声纹）、`models/`（模型文件）、`envs/`（隔离运行环境）、`tmp/`、`.env`。

## 模型

| 用途 | 当前配置 |
|---|---|
| 语音识别 | 主路本地 Qwen3-ASR 0.6B，副路 Grok `/v1/stt` |
| 人声检测 | 本地 FSMN-VAD |
| 声纹 | 本地 CAM++（sherpa-onnx） |
| 口型 | 本地 MediaPipe、YuNet、Light-ASD |
| 识图 | pi 调用 `opencode-go/deepseek-v4.1-flash`，思考级别 high |
| 角色 agent | pi 调用 `openai-codex/gpt-6.1-sol`，思考级别 high；看图的角色走识图配置 |

opencode-go 只允许 `deepseek-v4.1-flash` 和 `glm-5.3-flash`，列表写在 `config/models.json` 的 `allowed_models`，运行器调用前校验。

## 开始使用

```bash
uv sync                                  # 主环境
cp .env.example .env                     # 填写密钥；Grok 未配置密钥时使用 Pi 已登录的 xAI 凭据
uv run pytest -m "not live"              # 单元测试
```

本地语音识别和主动说话人检测使用独立环境，复建命令见 `tools/README.md`。模型文件下载到 `models/`，来源和校验值记录在各自的实测文档中。

## 开发方式

开发任务由编排者写成任务单，派给 pi 执行者：

```bash
orchestrate/dispatch.sh <任务编号> [补充说明]
```

执行者只改任务单列出的文件，完成后写报告；共享文件的改动由编排者合并。
