# T3e 本地语音识别后端与 Grok 对比

在本机运行中文语音识别，作为可切换的 `local` 后端，与 Grok 在同一样本上直接对比。总工已同意下载模型到本地。

## 负责文件

- `tools/s2s/asr/local.py` 以及 `tools/s2s/asr/local_*.py`（本任务新增的文件）
- `models/asr/`（模型文件）
- `docs/本地语音识别对比.md`
- `tests/test_t3e_*.py`
- `tmp/T3e/`

T3c 正在并行修改 `tools/s2s/asr/` 下已有的文件和黑板中的 `声音记录.json`。本任务不改这些文件。后端注册（`tools/s2s/asr/__init__.py`）和配置（`config/models.json` 的 `asr.backends.local`）写进报告，由编排者合并；测试时直接导入你的模块。识别结果写到 `tmp/T3e/`，不写黑板。

## 候选

| 候选 | 说明 |
|---|---|
| Paraformer-zh（FunASR） | 必测。带逐字时间戳的版本；有热词版本时一并测热词 |
| Qwen3-ASR 0.6B | 必测。有 MLX 或 CoreML 版本的优先 |
| Whisper large-v3-turbo | 参照。MLX、CoreML 或 whisper.cpp 任选 |
| SenseVoice-Small | 时间允许再测 |

- 先查清每个候选当前的官方发布渠道、在 Apple Silicon 上的推荐运行方式（MLX、CoreML、ONNX、PyTorch MPS），选最省依赖、能在本机稳定运行的方式。
- 模型下载到 `models/asr/<候选>/`，记录来源地址、版本、大小和校验值。
- 新依赖用 `uv run --with` 临时使用；需要 PyTorch 时说明安装体积。依赖冲突严重的候选可以用单独的 `uv` 虚拟环境（放在 `tmp/T3e/venv-<候选>/`），通过子进程调用。
- 不使用付费服务，不需要账号授权的模型优先。

## 后端接口

`local.py` 实现与其他后端相同的 `transcribe(audio_path, start=None, end=None) -> list[Sentence]`，配置项 `model` 选择候选，`hotwords` 传人名提示。词或字级时间戳按停顿和标点切句，切句参数与 T3c 保持可比（单句尽量不超过 6 秒）。本地后端不做说话人区分，`speaker` 填空，由 T3d 的说话人分离补上。

## 对比

在 `videos/sample/E01.mp4` 和 T3 的合成测试音频（见 `orchestrate/reports/T3.md` 找路径）上，每个候选和 Grok 各跑一遍：

| 项目 | 方法 |
|---|---|
| 用字与漏句 | 输出转成声音记录格式，用 `tools/sub_vs_asr.py` 与 `剧目/样本剧/E01/字幕记录.json` 对照，统计“用字不一致”“只有字幕”“只有声音”，并逐条列出人名（桂芝、朵朵）和已知错字（老师/老是、杵/处于、糊啦/胡辣）的识别情况 |
| 时间戳 | 合成音频上与真值比较句起止误差（平均、最大） |
| 人名提示 | 支持热词的候选，传入“桂芝、朵朵”前后各跑一次 |
| 配乐 | E01 中配乐较响的段落单独列出各候选结果 |
| 性能 | 冷启动加载时间、90 秒音频识别耗时、实时倍数、内存峰值 |

Grok 的结果用黑板中当前的 `声音记录.json`；报告完成前再读一次，如果 T3c 已经更新，用更新后的版本重新对比，注明使用的版本时间。

## 报告

`docs/本地语音识别对比.md` 最后给出结论：推荐的默认后端、推荐的本地候选，以及“两路交叉核对”（本地和 Grok 都识别出且一致的句子标为高可信）是否值得做，附依据。
