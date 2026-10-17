# shot2script 工具

Python 包名为 `s2s`，源码位于 `tools/s2s/`。所有命令在项目根目录执行。

## 环境安装

需要 uv、Python 3.12 或以上、ffmpeg、ffprobe，以及至少一个已安装并完成模型授权的运行器：pi 或 opencode。

```bash
uv sync
uv run python -m s2s
uv run pytest -m "not live"
```

项目使用 Python 3.12。预装 pytest、jsonschema、httpx、Pillow、lxml、python-docx、PySceneDetect、python-dotenv、rapidfuzz。PySceneDetect 使用支持 `opencv` 扩展的 0.6 系列，OpenCV 随扩展安装。

### 服务变量

将 `.env.example` 复制为 `.env`，填写下列变量。`.env` 已在 `.gitignore` 中排除。

- `ARK_DOUBAO_API_KEY`：豆包方舟密钥。
- `ARK_DOUBAO_PROXY`：豆包请求使用的代理地址；不需要代理时留空。
- `XAI_API_KEY`：Grok 密钥。

配置模块自动读取项目根目录的 `.env`，已有进程环境变量优先。空的 `S2S_*` 变量不覆盖模型配置。ASR 返回配置只包含密钥变量名；读取实际密钥可调用 `s2s.config.get_environment()`，返回值不得记录到日志。

pi 和 opencode 使用各自已有的模型授权，运行接口不会提取或复制其登录凭据。

## 模型配置

默认文件：`config/models.json`。顶层包含五项：

- `asr`：`provider` 选择后端；`backends` 保存各后端的 `api_base`、`model`、`key_env` 和可选 `proxy_env`。
- `vision`：`runner`、`model`、`thinking`、`max_images_per_call`。
- `agents`：`default` 为默认角色配置；角色可用 `use: "default"` 或 `use: "vision"` 继承，角色自身字段覆盖继承字段。
- `runners`：以运行器名为键，`bin` 指向可执行文件名或绝对路径。
- 整个配置文件可用 `S2S_CONFIG=/绝对路径/models.json` 替换。

`get_asr_backend()`、`get_vision()`、`get_agent(角色)` 每次读取当前配置并返回独立的字典。

### 临时切换

```bash
# ASR 后端；实际识别实现由声音识别模块提供。
S2S_ASR_PROVIDER=doubao uv run python -c 'from s2s.asr import get_backend; print(type(get_backend()).__name__)'

# 识图模型。
S2S_VISION_MODEL=opencode-go/glm-5.3-flash uv run python -c 'from s2s import vision; print(vision.ask(["tmp/T1/frame20.jpg"], "描述画面"))'

# 默认角色模型及其运行器；使用 provider/model 格式填写模型。
S2S_AGENT_MODEL=提供方/模型 S2S_AGENT_RUNNER=opencode uv run python -c 'from s2s import agent; print(agent.run("角色卡", "只回复 OK"))'
```

`S2S_VISION_RUNNER` 切换识图运行器。`S2S_AGENT_MODEL`、`S2S_AGENT_RUNNER` 修改 `agents.default`，不影响继承 `vision` 的角色；角色显式配置的字段优先于继承字段。只需修改某个角色时，在 `agents.<角色>` 中写对应字段。

## 接口与模块

### `s2s.config`：配置读取

```python
from s2s.config import get_asr_backend, get_vision, get_agent

backend = get_asr_backend()
vision_settings = get_vision()
role_settings = get_agent("动作识别")
```

### `s2s.vision`：图片识别

```python
from s2s import vision

text = vision.ask(["tmp/T1/frame20.jpg"], "描述可见人物", timeout=90)
```

超过 `max_images_per_call` 时抛出 `ValueError`，不自动拆分。关键字参数可单次覆盖 `model`、`runner`、`thinking`、`max_images_per_call`、`timeout`、`tools`、`bin`。

### `s2s.agent`：角色调用

```python
from s2s import agent

text = agent.run("角色卡", "只回复 OK")
text = agent.run("动作识别", "核对图片与资料", attachments=["资料.json"], images=["帧.jpg"])
```

未知角色抛出配置错误。继承识图配置的角色同样受图片数量上限约束。单次覆盖参数与识图接口一致。

### `s2s.runners`：统一运行器

```python
from s2s.runners import run

text = run("只回复 OK", images=[], files=[], model="openai-codex/gpt-6.1-sol", thinking="high", runner="pi", tools=False, timeout=120)
```

- pi 使用非交互、无会话、无上下文文件模式，附件通过 `@绝对路径` 传入；默认加 `--no-tools`。
- opencode 使用 `run --format json`，图片和文件通过 `--file` 传入，返回 JSON 事件中的正文文本。`thinking != "off"` 时传递 `--variant`；具体模型是否支持该级别由提供方决定。
- opencode 的 `tools=False` 通过进程环境内的 `OPENCODE_CONFIG_CONTENT` 设置 `permission: "deny"`，不修改磁盘上的运行器配置。`tools=True` 使用运行器已有权限，不自动批准请求。
- 默认超时为 120 秒；超时、非零退出码、空正文和 JSON 事件错误均抛出 `RunnerError`，包含运行器、模型、退出码与 stderr 末尾 20 行。已知环境密钥与代理值在异常中隐藏；不自动重试。

### `s2s.asr`：声音识别接口

```python
from s2s.asr import get_backend, Sentence

backend = get_backend()          # 当前 asr.provider
backend = get_backend("doubao") # 显式选择后端
sentences = backend.transcribe("音频.wav", start=10.0, end=20.0)
```

当前 Grok、豆包为接口占位，调用 `transcribe` 会抛出明确的“未实现”异常。

`base.py` 定义 `ASRBackend` 和 `Sentence`，`grok.py`、`doubao.py` 保存对应后端，`__init__.py` 提供 `BACKENDS` 注册表和 `get_backend()`。后端构造参数为解析后的配置字典，可在 `.settings` 中读取。

`Sentence` 字段为 `start`、`end`、`text`、`speaker`、`clarity`；起止时间使用本集内 `MM:SS.s`，`clarity` 为“清楚”“重叠”或“听不清”。`transcribe` 的窗口参数以秒计，返回时间相对整集。编号及转换为声音记录的中文字段由后续保存流程处理。

## 实际调用测试

```bash
uv run pytest -m "not live"
uv run pytest -m live -v -s -x
```

实际测试要求 `videos/sample/E01.mp4` 存在，使用 `/opt/homebrew/bin/ffmpeg` 取第 20 秒帧，在 `tmp/T1/` 保存帧与响应。测试依次通过 `S2S_VISION_MODEL` 使用 `opencode-go/deepseek-v4.1-flash`、`opencode-go/glm-5.3-flash`，再调用默认角色卡模型并检查输出为 `OK`。每次模型调用最多 90 秒，不自动重试。

未显式传入 `-m` 时，`pytest` 会包含实际调用测试；离线开发使用 `-m "not live"`。

## 模型允许列表

`config/models.json` 的 `allowed_models` 按提供方登记允许的模型。opencode-go 只允许 `deepseek-v4.1-flash`、`glm-5.3-flash`，默认使用 `deepseek-v4.1-flash`。`runners.run` 调用前用 `config.check_model` 校验，列表外的模型抛出 `ConfigError`；未登记的提供方不受限制。

## 工具清单

### 镜头切分与取帧

```bash
uv run python tools/shots.py videos/sample/E01.mp4 剧目/样本剧/E01/
uv run python tools/shots.py videos/sample/E01.mp4 剧目/样本剧/E01/ --detector adaptive --threshold 3 --min-shot-seconds 0.3 --fps 2 --max-frames 12
```

生成本集 `镜头表.json` 与 `素材/frames/`。`--detector` 可选 `content`（默认，阈值 27）或 `adaptive`（默认阈值 3）；`--threshold` 覆盖对应检测阈值。`--min-shot-seconds` 默认 0.3，至少 0.1 秒。`--fps` 为补帧率，默认 2；`--max-frames` 默认 12，至少 3。首、中、尾优先，时间差小于 0.2 秒的帧去重，超过上限时均匀保留补帧。图片长边为 960 像素。

集号优先从输出目录名 `E01` 推断，其次从视频名推断；其他命名须传 `--episode 1`。标准输出为镜头数量、帧数和耗时的 JSON 摘要。

### 定向加密取帧

```bash
uv run python tools/dense_frames.py videos/sample/E01.mp4 00:11.0 00:12.0 5 剧目/样本剧/E01/素材/frames_dense/
```

依次传入视频、起点、终点、帧率和输出根目录。工具自动建立 `00-11.0-00-12.0/` 时间区间子目录，生成带时间码的 JPG 和 `帧.json`。时间区间为左闭右开；时间码精度为 0.1 秒，帧率范围为 `(0, 10]`。使用镜头表舍入后的全片终点时，超出真实终点的部分不取帧。

### 连续镜头分段

```bash
uv run python tools/segments.py 剧目/样本剧/E01/镜头表.json
```

输出同目录 `片段表.json`，通常每段 6–10 镜头，段间大小之差不超过 1。全片 1–5 个镜头只成一段；11 个镜头合为一段。`衔接前`、`衔接后` 引用邻段边界镜头，首尾为 `null`。

共享时间接口：

```python
from s2s.timefmt import fmt, to_sec

assert to_sec("01:23.4") == 83.4
assert fmt(83.4) == "01:23.4"
```

### 语音识别

```bash
uv run python tools/asr.py videos/sample/E01.mp4 剧目/示例/E01
uv run python tools/asr.py videos/sample/E01.mp4 剧目/示例/E01 --provider grok --keyterm 林舟 --keyterm 沈宁
uv run python tools/asr_window.py videos/sample/E01.mp4 00:10.0 00:15.0 --provider doubao --raw-dir tmp/asr-window
```

`asr.py` 根据配置或 `--provider` 切换 Grok / 豆包，生成本集 `声音记录.json`，原始响应保存在本集 `素材/asr_raw/`；已有声音记录时拒绝覆盖。集号默认从目录名 `E01` 取得，也可用 `--episode` 指定。

Grok 使用 `/v1/stt`，默认模型为 `grok-voice-transcribe-2.0`，`--keyterm` 可重复传入角色卡姓名。豆包采用音频理解模型，返回时间与声线标签必须回听复核。

`asr_window.py` 支持秒数或 `MM:SS.s`，仅向标准输出打印窗口识别结果，不修改声音记录；返回时间仍相对于整集。省略 `--raw-dir` 时，原始响应默认保存在输入文件同目录的 `asr_raw/`。

凭据通过 `s2s.config.get_environment()` 读取项目 `.env`。Grok 需要 `XAI_API_KEY`，豆包需要 `ARK_DOUBAO_API_KEY`，代理使用 `ARK_DOUBAO_PROXY`。

实测状态和限制见 `docs/语音识别能力实测.md`。

凭据经 `s2s.config.get_environment()` 读取。Grok 优先使用非空 `XAI_API_KEY`；未配置时调用 `pi auth print-bearer-token --provider xai`，复用 Pi 已登录凭据并由 Pi 刷新。令牌仅保存在进程内存，不写日志、文件或错误。豆包使用 `ARK_DOUBAO_API_KEY` 与 `ARK_DOUBAO_PROXY`。

Grok 声线标签不能直接认定为人物身份；“清楚”不表示无重叠或漏句。真实效果和限制见 `docs/语音识别能力实测.md`。

### 字幕识别与声音对照

字幕识别使用 macOS Apple Vision，首次运行需要可用的 `swiftc`。批量识别程序缺失或源码更新时自动用 `swiftc -O` 编译。

```bash
uv run python tools/subtitles.py 剧目/样本剧/E01
uv run python tools/subtitles.py tmp/T4/E01 --frames tmp/T4/frames --fps 2 --audit tmp/T4/字幕识别审计.json
uv run python tools/sub_vs_asr.py 剧目/样本剧/E01
```

- `subtitles.py` 默认读取本集 `素材/frames/`，使用 `镜头表.json` 中每帧的时间；镜头帧缺少时间表时会报错，不按文件序号猜时间。显式 `--frames` 表示输入等间隔帧，按文件名排序，从 0 秒按 `--fps` 推算时间。
- 输出本集 `字幕记录.json`。几何位置与文字特征用于初步分类；模糊文字、实物招牌和异常类型需回帧核对。`--audit` 可另存逐帧 OCR、来源帧、参数和过滤情况。
- `--similarity`、`--position-tolerance`、`--dialogue-top`、`--dialogue-bottom`、`--line-max-height` 等参数控制合并与分类；`--keep-watermarks` 关闭持续边缘文字过滤，`--help` 显示全部参数。
- `sub_vs_asr.py` 读取字幕与声音记录，输出本集 `字幕声音对照.json`；`--output` 可指定结果路径。结果分用字不一致、只有字幕、只有声音，另附断句提示。字幕编号为原始字幕数组从 1 开始的序号；声音编号沿用原记录 `id`。
- 差异位置为拼接文本从 0 开始、左闭右开的区间，同时提供去标点后的索引和原文索引。比较忽略标点、空格和全半角；不判说话人归属，不以字幕替代实际声音。

### 黑板与机械校验

- `uv run python tools/board_init.py <剧名> <集数>`：初始化 `剧目/<剧名>/`，保留已有文件。
- `uv run python tools/validate.py <文件> [--type <类型>]`：按文件名或显式类型执行 JSON Schema、编号、时间与结构化引用检查。
- `uv run python tools/coverage.py <本集目录> [--稿 工作稿.md]`：检查句号、关键动作覆盖及正文台词逐字一致性。
- `uv run python tools/lint_script.py <工作稿或正式稿> [--角色卡 角色卡.json] [--正式稿] [--词表 词表.json]`：检查剧本格式和非对白描述。

工具输出 JSON，退出码 0 表示通过，1 表示失败。JSON Schema 位于 `tools/s2s/schemas/`；记录类黑板文件的顶层为数组。`--type` 可使用对应文件名去掉 `.json` 后的名称。

默认词表为 `tools/s2s/schemas/lint_words.json`，结构为 `{"泛称": ["男人"], "动作词": ["抬头"]}`。未提供角色卡时，人物行中的具体名称用于保护描述名；提供角色卡时，仅卡中统一名称用于名称检查与描述名保护。

Python 接口：`s2s.board.initialize_board(name, episodes, root=None)`、`s2s.validate.validate_file(file, data_type=None)`、`s2s.coverage.check_coverage(directory, script="工作稿.md")`、`s2s.lint_script.lint_file(file, role_card=None, formal=False, word_file=None)`。

### Word 生成与校验

合并稿使用纯文本剧本行：`第1集`、`场景1-1：会议室，日，内`、`人物：……`、△ 和对白；支持场景行末尾 `（闪回）`。

```bash
uv run python tools/docx_build.py 合并稿.md 合集.docx '剧号剧名 剧本合集'
uv run python tools/docx_check.py 合集.docx 3
# 全剧总共三集，第二集未收录：
uv run python tools/docx_check.py 合集.docx 3 --缺失集 2
```

生成器复用 `simple/范本_TG26090377假穷真爱_剧本合集_20260916.docx` 的样式、主题、字体表及页面设置，生成标准 `.docx`。集标题、场景行采用两级导航，正文逐行保留。

校验器实际解析成品，输出 JSON；`通过=true` 时退出码为 0，否则为 1。集数参数为全剧总集数；缺失集不应出现在成品中。

Python 接口：`s2s.docx_build.build_docx(合并稿路径, 输出路径, 标题)`、`s2s.docx_check.check_docx(Word路径, 总集数, missing={缺失集号})`。

### 角色派发与主控

```bash
uv run python tools/run_series.py 样本剧 1
uv run python tools/run_episode.py 样本剧 E01
uv run python tools/run_episode.py 样本剧 E01 --模式 定向复核 --复核要求 '变更内容与受影响位置'
uv run python tools/run_episode.py 样本剧 E01 --完成步骤 5
uv run python tools/dispatch_role.py 动作识别 --集 剧目/样本剧/E01 --变量 vars.json --输出 剧目/样本剧/E01/派发结果/动作识别/E01-P01.json
```

动作识别变量示例：`{"片段号":"E01-P01"}`。定向复看、定向核对、定向改写及验收复核必须提供 `--范围`。加密复看可加入 `加密帧索引` 数组，值为相对本集的 `素材/frames_dense/<时间段>/帧.json`。跨集验收使用 `--模式 跨集`，变量增加 `跨集集号` 整数数组；复核可通过变量提供 `改动对照`。

子角色只读取授权图片，不能写黑板；派发结果通过校验后保存，由主控审核并合并到公共文件。成稿结果对象包含 `工作稿` 字符串和 `对照表` 数组。角色卡所需的画面人物和口型从 `派发结果/动作识别/*.json` 提取，片段结果应保存在该目录。

已有成功片段结果在输入未变化且重新校验通过时自动复用；定向复看不使用此缓存。派发日志保存至本集 `派发日志.jsonl`，记录 UTC+8 时间、角色、输入、模型、耗时、重答、用量与结果。无法通过校验的返回保存在输出目录的 `校验失败/` 下。

主控会话保存在 `会话/`；重启使用当前状态和文件清单创建新会话，不续接旧对话。`run_episode.py`、`run_series.py` 支持 `--超时`，默认 900 秒；`dispatch_role.py` 默认 180 秒。超时和运行器错误直接退出，只有格式校验错误触发最多两次重答。

### 口型证据

`uv run --with 'mediapipe==1.1.1' python tools/lipsync.py <视频> <本集目录> <输出.json> [--fps 12] [--复核]`

输入为镜头表与声音记录。输出口型记录及输出目录下的 `口型核对.html`，密集帧与运行快照置于同目录的独立证据子目录。已有输出不覆盖。模型需放在 `models/lipsync/face_landmarker.task`，模型来源与校验信息见 `models/lipsync/模型信息.json`。

`--复核` 只调用默认识图配置，对存疑句附加复核，不覆盖代码判断。分数为启发式证据，未开口不直接等于心声，看不到也不证明人物不在画面。实际能力与限制见 `docs/口型检测实测.md`。

### ASR 人名提示与字幕漏句回查

```bash
uv run python tools/asr.py videos/sample/E01.mp4 tmp/下一轮/E01 \
  --provider grok --keyterm-from 剧目/样本剧/E01/字幕记录.json \
  --recover-from 剧目/样本剧/E01/字幕记录.json

uv run python tools/asr_window.py videos/sample/E01.mp4 00:49.3 00:51.8 \
  --provider grok --keyterm-from 剧目/样本剧/E01/字幕记录.json \
  --raw-dir tmp/下一轮/window/asr_raw
```

`--keyterm-from` 可重复指定角色卡或字幕记录，与已有 `--keyterm`、配置提示词合并去重。角色卡读取姓名及别名称呼；字幕读取反复出现的人名候选。候选仅用于提示服务，不修改角色卡或按字幕替换台词。

整集工具在本集已有 `字幕记录.json` 时自动回查“只有字幕”的时间段，前后各扩 1 秒；`--recover-from` 指定其他字幕文件，`--no-recover` 关闭。过滤已有台词和窗口上下文；新增文本仅采用音频服务返回，标“听不清”，窗口声线不直接映射到整集人物。回查诊断保存在 `素材/asr_raw/recovery.json`；命令输出含回查、新增和失败窗口数量。服务没识别出的台词不会从字幕复制补齐。

Grok 参数位于 `asr.backends.grok`：`format=false`、`pause_seconds=0.35`、`max_sentence_seconds=6.0`、`min_speaker_seconds=0.3`；可选 `vad_threshold` 为 0–1，整集未配置时保留服务默认值。短声线跳变合并标“听不清”；独立短应答保留声线。详情见 `docs/语音识别能力实测.md` 第 8 节。

### 本地语音识别

`local` 后端支持 `paraformer-zh`、`paraformer-contextual`、`qwen3-asr-0.6b`，通过 `asr.backends.local.model` 切换。模型保存在 `models/asr/`，独立运行环境通过 `python_executable` 指定；`hotwords` 为姓名列表。

```python
from s2s.asr import get_backend
sentences = get_backend("local").transcribe("videos/sample/E01.mp4", 10, 15)
```

时间相对整集，返回 `speaker` 为空，需由说话人分离补齐。当前声音记录导出中的统一声线标签仅为未知占位，不能认定为同一人物。Qwen 单次窗口最长 300 秒。Paraformer 原生时间版不支持热词，姓名提示需使用热词版。Whisper 的本地权重尚不完整，暂不可用。

实测对比、依赖与模型摘要见 `docs/本地语音识别对比.md`。Grok 仍为默认；两路一致不替代独立验收。

独立运行环境位于 `envs/local-asr/`（不入 git），复建：`uv venv --python 3.12 envs/local-asr && uv pip install --python envs/local-asr/bin/python 'funasr-onnx==0.4.3' 'mlx-audio==0.5.9' python-dotenv httpx rapidfuzz`。`python_executable` 须写虚拟环境内的解释器路径，不能写解析符号链接后的路径。

### 双路识别与逐句回听

在 `asr.cross_check` 配置 `enabled: true`、`primary` 和 `secondary`。两个后端名称须不同；本地具体模型由 `asr.backends.local.model` 指定。单路工具仍使用 `asr.provider`。

```bash
uv run python tools/asr_cross.py videos/sample/E01.mp4 剧目/样本剧/E01
uv run python tools/asr_cross.py videos/sample/E01.mp4 剧目/样本剧/E01 \
  --主路结果 a.json --副路结果 b.json
uv run python tools/listen_sheet.py 剧目/样本剧/E01 tmp/E01回听.html
uv run python tools/listen_sheet.py 剧目/样本剧/E01 tmp/E01全部回听.html --全部
uv run python tools/apply_listen.py 剧目/样本剧/E01 回听结果.json
```

传入结果文件会直接复用，缺少的那一路才调用识别后端。合并保留主路句子的文本、时间和声线；仅副路句子按时间插入。非一致句生成阻断级待回听问题。合并和应用均会备份已有声音记录及待核对，并在写回前后校验。

回听页离线打开，音频位于本集 `素材/回听/`，页面使用相对路径。默认按优先级显示待处理句子；`--全部` 包含一致句和人工确认句。复制页面时保留所引用音频与代表帧的相对目录结构。

空格播放或暂停本句，C 播放上下文，1/2/3 采用主路/副路/字幕，J/K 切换句子，E 编辑，Esc 退出输入。选择文本或确认手改后计入进度；可调整听辨、备注或标记没有台词。浏览器不能存储进度时仍能操作，离开前导出。

导出文件包含声音快照与已完成决定。应用会拒绝已过期的快照；部分应用后继续处理，应重新生成页面。删除导致句号变化时，主控按 `素材/回听/句号对照.json` 更新下游引用，删除项为 `null`。归属表、口型记录、剧本和对照表不由本工具自动改写。

口型工具支持 `--variant baseline|side|model|fused`，默认 baseline。side 用 YuNet 补充 MediaPipe 未覆盖的人脸；model 在补检轨迹上使用 Light-ASD 分，fused 融合规则分与模型分。E01 四版及完整存疑复核实测见 `docs/口型检测实测.md`。

`--model-weight 0.65` 设置模型权重，`--conflict-gap 0.55` 设置冲突差，`--pixel-motion-scale 0.08` 设置无可靠唇点分支的像素变化尺度。ASD 默认独立 Python 为 `envs/asd/bin/python`，可用 `--asd-python` 指定；`--asd-timeout` 默认1200秒。主环境无需安装 torch。

`--reuse-source <side口型记录.json>` 复用侧脸版轨迹和规则证据，仅支持 model/fused，沿用源参数；`--asd-source <asd输出.json>` 复用输入清单及权重路径匹配的模型结果。输出 JSON 与核对页须不存在。

`--复核 --review-source <已有口型记录.json>` 只复核存疑句，使用默认识图配置、不指定模型，不覆盖代码结论。已有已复核回复可复用；不足4张证据的短句从原视频按原帧率补取句内帧，保存补帧索引和原始回复。输出到新目录。

ASD 独立环境复建：`uv venv --python 3.12 envs/asd && uv pip install --python envs/asd/bin/python torch==2.14.1 numpy==2.5.4 scipy==1.18.1 python-speech-features==0.6 opencv-python-headless==5.0.0.93`（约 800MB）。

### 约束声纹与跨集参考

在模型配置顶层设置 `diarize.enabled: true` 后，`asr_cross.py` 合并两路再聚类声线，保存本集 `声纹聚类.json`。缺失或禁用时不运行声纹。constraints 可选 grok、lipsync、max_speakers；已知人数上限加1为目标类数，硬约束冲突时保留分离。口型与当前句文本、时间不匹配时不使用。待定句保留旧标签，必须配合结果侧表。

```bash
S2S_CONFIG=tmp/T3g/models.json uv run python tools/asr_cross.py \
  videos/sample/E01.mp4 tmp/示例/E01 --主路结果 <主路.json> --副路结果 <副路.json>
uv run python tools/voiceprint.py 入库 <本地剧目目录> E01 --音频 <E01.mp4>
uv run python tools/voiceprint.py 比对 <本地剧目目录> E02 --音频 <E02.mp4>
uv run python tools/voiceprint.py 评测 <声纹聚类.json> <标注.json> --声线角色映射 <映射.json>
```

入库须有确认的角色编号归属表，跳过非清楚音频；重复入库替换该集贡献。比对只输出前三候选，不确认人物，不修改归属表。未指定音频时从镜头表读取视频路径。

独立 `tools/diarize.py` 新增 `--约束 grok lipsync max_speakers`、`--人数上限 6`、`--口型 <口型记录.json>`、`--Grok权重`、`--Grok纯度`、`--Grok最低相似度`、`--试听目录 <新目录>`。未指定约束时保留旧无约束聚类。原有 `--词级`、`--写回` 行为不变。

声纹数据只在本机黑板保存，不附到提示词、日志或报告。实验与使用边界见 `docs/说话人分离实测.md` 第9节。

### 派发与分段参数

`dispatch_role.py` 支持 `--提示词 <文件>`，默认正式提示词。全文验收自动执行整集文本检查与分段画面核对；结果旁的 `.coverage.json` 保存逐段实际读帧覆盖，`全文审查完成=false` 时不得标记全文完成。`segments.py` 支持 `--shots-per-segment`、`--soft-limit`、`--frame-budget`、`--context-frame-budget`，默认分别为 6、8、36、48。

### 按音频统一切句与回听

交叉核对使用本地 FSMN-VAD、短时能量及两路词级时间建立共享句界。字幕只在人工回听页作为参考展示。切句参数为 `asr.segmentation.min_pause=0.5`、`max_sentence=6`，可用同名命令行参数覆盖。

```bash
uv run python tools/asr_cross.py videos/sample/E01.mp4 tmp/新一轮/E01 \
  --主路词 主路原始响应.json --副路词 副路原始响应.json \
  --人声段 已有人声段.json
```

两份词级响应均提供时不调用识别后端；未提供人声段时执行本地检测。复用句级文件时，仍须同时提供 `--主路词`、`--副路词`；缺少原始词时间即报错，不通过重新识别补齐。新识别的本地后端将词级响应保存在 `素材/asr_raw/local-*.json`。

`asr.py --recover-from` 接受人声段，不接受字幕；`--secondary-words` 提供已有副路词级响应后，才允许对人声中两路均无词的区间补识别。`--keyterm-from` 只接受角色卡。补识别会调用所选后端，不用于禁止新识别的离线重切任务。

声音记录及人工裁定迁移完成后生成回听页：

```bash
uv run python tools/listen_sheet.py tmp/新一轮/E01 tmp/新一轮/E01/回听.html
uv run python tools/apply_listen.py tmp/新一轮/E01 回听结果.json
```

本句音频前后各补 0.2 秒，“带上下文”覆盖前一句开头至后一句结尾。原裁定发生拆句、合句、变界时在页上显示待确认。应用回听结果只校验句号、文本及起止摘要；声线等后续字段变化不会使页面失效，变化在命令输出中列出。
