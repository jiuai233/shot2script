# T3e 报告

状态：部分完成

本地后端已实现，Paraformer-zh、Paraformer 热词版及 Qwen3-ASR 0.6B 均完成原片与合成音频实测；32 项单元测试、11 份声音记录格式校验通过。Whisper 权重续传超时未测，强配乐分层未完成。默认建议保留 Grok，本地首选 Qwen；两路一致只用于复核排序，不自动免验收。

## 完成内容

1. 新增 `LocalBackend`，接口与现有 `ASRBackend` 一致。支持 `model`、`hotwords`、独立 Python 环境、窗口识别；返回整集绝对时间和空 `speaker`。复用 T3c 切句实现，停顿阈值 0.35 秒、目标最长句 6 秒。
2. 下载并实跑官方 Paraformer 带字时间 ONNX 版本、官方热词 ONNX 版本及社区 MLX Qwen3-ASR 0.6B 4 位版本；辅助使用官方 FSMN-VAD 与 MLX Qwen3-ForcedAligner。完整权重已核对上游 SHA256，来源、版本、大小和逐文件摘要均已记录。没有直接用命令读取 `.env`，没有新增付费服务调用，没有使用识图模型。
3. 在 E01 90.27925 秒原片、T3 的 25.85 秒合成音频上完成三种本地配置的识别；热词版和 Qwen 另做“桂芝、朵朵”提示前后比较。用 `tools/sub_vs_asr.py` 对每份原片输出做相同参数对照，逐条记录姓名、易错字、漏句及原声背景窗口。
4. 完成时间、耗时、RSS 和 Metal 峰值统计。合成非重叠八句，各本地候选均文字一致；重叠两句均未完整保留。Grok 使用黑板当前 41 句版本及 T3 已有合成响应，没有重新请求付费接口。
5. 写入 `docs/本地语音识别对比.md`，给出默认、本地候选与交叉核对的结论；注册表、共享配置、现有 ASR 文件及黑板均未修改。

## 文件清单

### 代码与测试

- `tools/s2s/asr/local.py`：本地接口、独立环境子进程、窗口与切句。
- `tools/s2s/asr/local_runtime.py`：ONNX / MLX 加载、VAD、识别、强制对齐及零时长字处理。
- `tests/test_t3e_local.py`：32 项测试。

### 文档与模型

- `docs/本地语音识别对比.md`、`orchestrate/reports/T3e.md`。
- `models/asr/paraformer-zh/`、`paraformer-contextual/`：完整可用；热词版 `model.onnx` 为指向 `model_quant.onnx` 的符号链接。
- `models/asr/qwen3-asr-0.6b/`、`qwen3-aligner-0.6b/`、`fsmn-vad/`：完整可用。
- `models/asr/whisper-large-v3-turbo/`：不完整，不能推理。

### 运行材料

- `tmp/T3e/runs/paraformer-zh-vad/`、`runs/paraformer-contextual-final/`、`runs/qwen3-asr-0.6b-final/`：采用的识别文本、词时间、切句、耗时及内存结果。
- `tmp/T3e/comparison/`：Grok 快照、字幕对照、逐句误差和交叉一致性统计；`model_manifest.json`：完整下载来源、版本、大小和摘要。
- `tmp/T3e/venv-local/`、`versions.json`、`install.log`、`install-entry.log`：隔离环境与依赖记录。
- `tmp/T3e/run_bench.py`、`analyze.py`、`download_models.py`、`resume_models.py`：实测、对比与下载脚本；`sources/`：已读取的官方源码、GitHub 问题和社区全文。
- `tmp/T3e/test-results.log`、`validation.json`、`backend-window.json`、`grok-final-check.json`：定向测试、格式校验、真实窗口与最终版本复读结果。

## 测试与实际运行结果

测试日期：2026-10-17，所有时间均为 UTC+8。Apple M5、24 GiB 统一内存，macOS 27.0.1、Python 3.12.13。

### 单元测试与真实接口

```bash
uv run pytest tests/test_t3e_local.py -q --basetemp=tmp/T3e/pytest-final-32
```

结果：**32 passed，0.04 秒**。覆盖候选与参数检查、姓名提示、窗口偏移、空声线、停顿与六秒切句、标点恢复、零长字合并、VAD 偏移、子进程协议、失败与超时、缺模型不下载及 Qwen 300 秒对齐窗口限制。

11 份采用的声音记录均通过 `tools/validate.py`，明细见 `tmp/T3e/validation.json`。当前数据格式禁止空声线，因此评测文件通过现有 `records()` 写统一占位 `E01-V1`；不代表只有一名说话人。后端 `Sentence.speaker` 始终为空，应交由 T3d 补充。

真实执行 `LocalBackend` 的独立环境子进程，对 E01 10–15 秒窗口返回：

```json
[
  {"start":"00:10.3","end":"00:13.0","text":"你出什么事儿了吧能出什么事儿老师看着呢","speaker":"","clarity":"清楚"},
  {"start":"00:14.2","end":"00:14.8","text":"多多的","speaker":"","clarity":"清楚"}
]
```

这是窗口实际结果，未按字幕改字。“清楚”是默认标签，不代表准确率验收通过。

### E01 对照结果

| 后端 | 用字不一致 | 只有字幕 | 只有声音 |
|---|---:|---:|---:|
| Grok，当前 41 句 | 13 | 1 | 0 |
| Paraformer-zh | 8 | 1 | 1 |
| Paraformer 热词版，无 / 有提示 | 8 / 8 | 1 / 1 | 1 / 1 |
| Qwen，无 / 有提示 | 9 / 9 | 0 / 0 | 0 / 0 |

上述是多对多对照组数量，不能直接排名准确率。Qwen 和 Paraformer 均识别到当前 Grok 漏掉的“三缺一就差你一个”；Paraformer 片头只保留“奶奶”，Qwen 保留完整开场台词。热词版将“桂枝 → 桂芝”“朵多 → 朵朵”；Qwen 两处“慧芝”在姓名提示后仍未修正。

### 时间与性能

使用全部后端共同识别正确的八句非重叠合成台词，首尾字时间统一取 0.1 秒交付精度：

| 后端 | 起点平均 / 最大误差（毫秒） | 终点平均 / 最大误差（毫秒） |
|---|---:|---:|
| Grok | 21.3 / 50 | 38.8 / 50 |
| Paraformer-zh | 48.8 / 120 | 246.3 / 540 |
| Paraformer 热词版 | 51.2 / 120 | 83.8 / 150 |
| Qwen | 38.8 / 100 | 96.3 / 150 |

TTS 时间真值与 Grok STT 同厂商，不能视为独立精度排名；E01 尚无人声人工打点。热词前后合成结果相同。重叠两句：Grok、Paraformer 各保留不同的一句；Qwen 混为“我真的没有骗我了”。

| 本地候选 | 新进程加载（秒） | E01 识别含对齐（秒） | 实时速度 | RSS / Metal 峰值（GiB） |
|---|---:|---:|---:|---:|
| Paraformer-zh + VAD | 0.730 | 1.152 | 78.4× | 0.619 / — |
| Paraformer 热词版 + VAD + 对齐 | 3.578 | 2.532 | 35.7× | 2.472 / 1.871 |
| Qwen + 对齐 | 2.265 | 3.483 | 25.9× | 2.150 / 3.261 |

以上为最终无提示单次运行，不含下载与 ffmpeg 抽音频；系统缓存未清空，RSS 与 Metal 不能相加。完整热词、合成性能见对比文档。

### Grok 最终版本

比较使用 `剧目/样本剧/E01/声音记录.json`，41 句，修改时间 **2026-10-17 16:22:39.037155 UTC+8**。报告前 **16:33:10.494120 UTC+8** 再次读取，内容与已比较快照一致，无需重算。合成结果来自 `tmp/T3/grok-continuation/benchmark.json` 及对应原始响应。

## 需要编排者合并的共享改动

### 1. 后端注册

在 `tools/s2s/asr/__init__.py` 增加导入与注册，不改已有项：

```python
from .local import LocalBackend

BACKENDS["local"] = LocalBackend
```

### 2. 本地配置

在 `config/models.json` 的 `asr.backends` 增加下列 `local` 对象；`asr.provider` 暂时保留 `grok`：

```json
"local": {
  "model": "qwen3-asr-0.6b",
  "python_executable": "/Users/jiuai233/prj/shot2script/tmp/T3e/venv-local/bin/python",
  "hotwords": [],
  "pause_seconds": 0.35,
  "max_sentence_seconds": 6,
  "timeout_seconds": 300
}
```

模型默认从项目 `models/asr/<model>/` 读取。`model` 可切为 `paraformer-zh` 或 `paraformer-contextual`；热词版可设 `hotwords=["桂芝","朵朵"]`。`whisper-large-v3-turbo` 的调用分支已准备，但本机权重不完整，不能配置为实际可用候选。

当前解释器位于任务临时环境，后续正式部署需要保留或迁移该环境并更新绝对路径。注册合并后可在项目环境导入：

```python
from s2s.asr import get_backend
sentences = get_backend("local").transcribe("videos/sample/E01.mp4")
```

### 3. 命令行兼容点

`tools/asr.py` 和 `tools/asr_window.py` 的 `--provider` 目前仅允许两项，需在两处扩为：

```python
parser.add_argument("--provider", choices=("grok", "doubao", "local"))
```

`tools/asr.py` 给 `backend.settings["raw_dir"]` 赋值后，需要由固定流程创建该目录；本地模块不保存服务原始响应，若启用字幕回查，现有命令会写 `recovery.json`：

```python
Path(backend.settings["raw_dir"]).mkdir(parents=True, exist_ok=True)
```

这几处现有文件归 T3c，本任务未修改。`--keyterm` 当前专用于 Grok；本地姓名提示先走配置 `hotwords`，无须扩大本轮命令行接口。

### 4. 依赖

不建议直接把 FunASR 加入项目主环境：其 NumPy ≤1.26.4 与现有 OpenCV 的 NumPy ≥2 实际冲突。隔离环境的复建命令为：

```bash
uv venv --python 3.12 tmp/T3e/venv-local
uv pip install --python tmp/T3e/venv-local/bin/python \
  'funasr-onnx==0.4.3' 'mlx-audio==0.5.9' 'mlx-whisper==0.4.3' \
  python-dotenv httpx rapidfuzz pytest
```

完整版本记录为 `tmp/T3e/versions.json`。`mlx-whisper` 安装了 `torch==2.14.1`，文件总大小约 500 MiB；只部署已实测的 Paraformer / Qwen 时，不需要安装 `mlx-whisper` 及其 PyTorch 依赖。未修改 `pyproject.toml` 或 `uv.lock`。

### 5. tools/README.md 段落

可直接追加：

````markdown
## 本地语音识别

`local` 后端支持 `paraformer-zh`、`paraformer-contextual`、`qwen3-asr-0.6b`，通过 `asr.backends.local.model` 切换。模型保存在 `models/asr/`，独立运行环境通过 `python_executable` 指定；`hotwords` 为姓名列表。

```python
from s2s.asr import get_backend
sentences = get_backend("local").transcribe("videos/sample/E01.mp4", 10, 15)
```

时间相对整集，返回 `speaker` 为空，需由说话人分离补齐。当前声音记录导出中的统一声线标签仅为未知占位，不能认定为同一人物。Qwen 单次窗口最长 300 秒。Paraformer 原生时间版不支持热词，姓名提示需使用热词版。Whisper 的本地权重尚不完整，暂不可用。

实测对比、依赖与模型摘要见 `docs/本地语音识别对比.md`。Grok 仍为默认；两路一致不替代独立验收。
````

## 未完成与问题

1. Whisper 权重续传超过 240 秒后停止，未重试。已下载 1,337,407,110 / 1,613,977,612 字节，尚缺 276,570,502 字节；不完整文件保留在 `models/asr/whisper-large-v3-turbo/weights.safetensors`，摘要和来源见 manifest，不能推理，也没有填写虚构结果。
2. E01 没有独立确认的强配乐段落或配乐音轨，只列出原声窗口结果，未完成强配乐分层测试。SenseVoice-Small 未测。
3. 缺少 E01 人工逐字与时间真值及更多剧集；字幕差异不能当作真实声音错误率。默认后端替换结论尚无多集证据。
4. 本地不做说话人分离或重叠检测；合成重叠未完整识别，空声线和默认“清楚”标签不能当作身份或置信度。
5. 注册、配置及命令行兼容点待编排者合并；当前已通过直接导入模块验证。工作目录没有 Git 元数据，无法提供 `git diff`；本任务写入范围为任务负责文件及规定报告。

## 需要总工决定的事项

1. 是否批准继续补齐 Whisper 剩余约 277 MB 权重并完成参照实测；本轮已停止，不会自行重试。
2. 指定强配乐时间段及 E01 回听、时间标注负责人，以完成真实剧集质量验收。
3. 是否采用“Grok 默认 + Qwen 本地复核”的配置；一致句保留复核优先级提示，冲突句回听，不自动验收。
