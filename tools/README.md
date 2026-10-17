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

### 环境变量

`.env.example` 提供配置覆盖项，复制为 `.env` 后按本机路径填写。进程环境优先；空的 `S2S_*` 不覆盖配置。语音仅使用本地模型，不需要远程语音密钥。pi 和 opencode 使用各自已有的 DeepSeek 授权，运行接口不提取或复制登录凭据。

## 模型配置

默认文件：`config/models.json`。主要字段：

- `asr`：`provider=local`，`backends.local.model=qwen3-asr-0.6b`；`segmentation` 设置人声检测和切句参数。
- `vision`：`runner`、`model`、`thinking`、`max_images_per_call`。
- `agents`：`default` 为默认角色配置；角色可用 `use: "default"` 或 `use: "vision"` 继承，角色自身字段覆盖继承字段。
- `runners`：以运行器名为键，`bin` 指向可执行文件名或绝对路径。
- `diarize`：CAM++ 本地声纹、相似度阈值和 `lipsync` 约束。

整个配置文件可用 `S2S_CONFIG=/绝对路径/models.json` 替换，模型与 ASR 限制仍然生效。

`get_asr_backend()`、`get_vision()`、`get_agent(角色)` 每次读取当前配置并返回独立的字典。

### 配置覆盖

`S2S_VISION_RUNNER`、`S2S_AGENT_RUNNER` 可修改运行器。模型变量只接受 `opencode-go/deepseek-v4.1-flash`，`S2S_ASR_PROVIDER` 只接受 `local`。

`S2S_AGENT_MODEL`、`S2S_AGENT_RUNNER` 修改 `agents.default`，不影响继承 `vision` 的角色。角色显式字段优先于继承字段，模型仍须通过生产校验。

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

text = run("只回复 OK", images=[], files=[], model="opencode-go/deepseek-v4.1-flash", thinking="high", runner="pi", tools=False, timeout=120)
```

- pi 使用非交互、无会话、无上下文文件模式，附件通过 `@绝对路径` 传入；默认加 `--no-tools`。
- opencode 使用 `run --format json`，图片和文件通过 `--file` 传入，返回 JSON 事件中的正文文本。`thinking != "off"` 时传递 `--variant`；具体模型是否支持该级别由提供方决定。
- opencode 的 `tools=False` 通过进程环境内的 `OPENCODE_CONFIG_CONTENT` 设置 `permission: "deny"`，不修改磁盘上的运行器配置。`tools=True` 使用运行器已有权限，不自动批准请求。
- 默认超时为 120 秒；超时、非零退出码、空正文和 JSON 事件错误均抛出 `RunnerError`，包含运行器、模型、退出码与 stderr 末尾 20 行。已知环境密钥与代理值在异常中隐藏；不自动重试。

### `s2s.asr`：声音识别接口

```python
from s2s.asr import get_backend
backend = get_backend("local")
sentences = backend.transcribe("音频.wav", start=10.0, end=20.0)
```

生产注册表只含本地 Qwen3-ASR；其他后端、其他本地模型和启用双路的配置会被拒绝。Qwen 单次窗口最长 300 秒，整集入口按不重叠窗口处理。`last_words` 保存真实词时间，原始响应写入 `素材/asr_raw/`。

`Sentence` 包含 `start`、`end`、`text`、`speaker`、`clarity`，起止为本集内 `MM:SS.s`。本地 ASR 不识别人，声线为空；整集流程中的占位标签不代表人物，须结合声纹侧表和角色证据核对。

## 实际调用测试

```bash
uv run pytest -m "not live"
uv run pytest -m live -v -s -x
```

实际测试要求 `videos/sample/E01.mp4` 存在，使用 `/opt/homebrew/bin/ffmpeg` 取第 20 秒帧，在 `tmp/T1/` 保存帧与响应。测试依次通过 `S2S_VISION_MODEL` 使用 `opencode-go/deepseek-v4.1-flash`，再调用默认角色卡模型并检查输出为 `OK`。每次模型调用最多 90 秒，不自动重试。

未显式传入 `-m` 时，`pytest` 会包含实际调用测试；离线开发使用 `-m "not live"`。

## 模型允许列表

生产仅接受完整标识 `opencode-go/deepseek-v4.1-flash`。配置文件、环境变量、角色显式覆盖、单次参数、主控和重答路径都执行调用前检查；未知提供方、其他模型和带后缀标识均拒绝。配置中的 `allowed_models` 记录生产接入，不能扩展代码允许范围。失败直接报错，不回退模型。

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
uv run python tools/asr_window.py videos/sample/E01.mp4 00:10.0 00:15.0 --raw-dir tmp/asr-window
```

本集目录使用 `E01` 格式。整集入口生成 `声音记录.json`、`待核对.json`、人声段、声纹聚类与跨集候选。`diarize.enabled=false` 时不运行声纹与候选比对。已有声音和待核对记录在校验后备份，再写入新结果。

声音核对状态为 `单路识别`、`待核对`、`人工确认`。截断、句长异常、词时间与人声不匹配、重叠、声纹待定和口型冲突触发回听。有人声但没有识别词的区间单独登记待核对，不生成虚构台词。字幕不参与切句和补识别。

定向入口只输出 JSON 与保存原始响应，不改黑板；时间仍相对于整集。

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

### 单路重切与裁定迁移

```bash
uv run python tools/asr.py videos/sample/E01.mp4 剧目/示例/E01 \
  --words 剧目/示例/E01/素材/asr_raw/local-原始响应.json \
  --人声段 剧目/示例/E01/素材/人声段.json --原裁定 回听结果.json
```

`--words` 可重复传入多个 Qwen 窗口响应，窗口时间转换为本集时间；提供后不调用 ASR。省略 `--人声段` 会执行本地 VAD。切句参数默认为 `--min-pause 0.5`、`--max-sentence 6`。无真实词时间时报错，不从句级时间推造。

只有原文、起止和覆盖范围都未变化、且决定来源适用时，才沿用人工裁定；其余在 `素材/回听/裁定迁移.json` 保留待确认记录。旧双路选择不自动确认为单路结果。

重切或人工改字、删句后，须重建受影响的口型、归属、剧本与对照表。派发器会隔离过期证据并显式报告，不替主控重写这些内容。

### 本地语音识别环境

Qwen 模型、强制对齐模型及 FSMN-VAD 放在 `models/asr/`。`asr.backends.local.python_executable` 指向隔离解释器，`model_path`、`aligner_path` 和 `vad_path` 可指定模型目录。环境复建：

```bash
uv venv --python 3.12 envs/local-asr
uv pip install --python envs/local-asr/bin/python 'funasr-onnx==0.4.3' 'mlx-audio==0.5.9' python-dotenv httpx rapidfuzz
```

解释器须保留虚拟环境内路径，不解析成系统解释器路径。依赖和历史对比见 `docs/本地语音识别对比.md`。

### 逐句回听

```bash
uv run python tools/listen_sheet.py 剧目/示例/E01 剧目/示例/E01/回听.html
uv run python tools/listen_sheet.py 剧目/示例/E01 tmp/全部回听.html --全部
uv run python tools/apply_listen.py 剧目/示例/E01 回听结果.json
```

页面默认展示异常句、待确认裁定及疑似漏段音频；`--全部` 包含普通单路句和人工确认句。单路按钮为原文和参考字幕；人工操作才计入确认进度。漏段没有可选文本，须回听后定向识别或人工打点补录，再更新问题状态。

空格播放本句，C 播放上下文，1/2 采用原文/字幕，J/K 切换句子，E 编辑，Esc 退出输入。每个按钮可用 Tab、Enter 或空格操作。音频前后补 0.2 秒，上下文覆盖前一句到后一句。复制页面须保留音频和代表帧的相对路径。

导出包含声音快照与已完成决定；应用时核对句号、文本及起止摘要，声线等变化单独报告。删除后的句号映射写入 `素材/回听/句号对照.json`。未解决项继续保留，不能以单路识别状态替代人工确认。

### 口型变体与复核

口型工具支持 `--variant baseline|side|model|fused`，默认 baseline。side 用 YuNet 补充 MediaPipe 未覆盖的人脸；model 在补检轨迹上使用 Light-ASD 分，fused 融合规则分与模型分。E01 四版及完整存疑复核实测见 `docs/口型检测实测.md`。

`--model-weight 0.65` 设置模型权重，`--conflict-gap 0.55` 设置冲突差，`--pixel-motion-scale 0.08` 设置无可靠唇点分支的像素变化尺度。ASD 默认独立 Python 为 `envs/asd/bin/python`，可用 `--asd-python` 指定；`--asd-timeout` 默认1200秒。主环境无需安装 torch。

`--reuse-source <side口型记录.json>` 复用侧脸版轨迹和规则证据，仅支持 model/fused，沿用源参数；`--asd-source <asd输出.json>` 复用输入清单及权重路径匹配的模型结果。输出 JSON 与核对页须不存在。

`--复核 --review-source <已有口型记录.json>` 只复核存疑句，使用默认识图配置、不指定模型，不覆盖代码结论。已有已复核回复可复用；不足4张证据的短句从原视频按原帧率补取句内帧，保存补帧索引和原始回复。输出到新目录。

ASD 独立环境复建：`uv venv --python 3.12 envs/asd && uv pip install --python envs/asd/bin/python torch==2.14.1 numpy==2.5.4 scipy==1.18.1 python-speech-features==0.6 opencv-python-headless==5.0.0.93`（约 800MB）。

### 声纹、口型与跨集参考

`asr.py` 在切句后执行 CAM++ 聚类，再生成前三跨集候选；候选只用于核对身份。生产只使用 `lipsync` 约束，不从已有角色数量推导本集人数上限。旧 Grok 原始响应不参与生产声纹。

口型须与当前句号、原文、起止完整匹配才参与聚类。先生成单路声音，再用口型工具产出本集 `口型记录.json`，随后复用 `--words` 和 `--人声段` 重切，即可使用有效口型约束并刷新候选。缺失或过期口型在 `输入可用性` 中说明。

```bash
uv run python tools/voiceprint.py 入库 <本地剧目目录> E01 --音频 <E01.mp4>
uv run python tools/voiceprint.py 比对 <本地剧目目录> E02 --音频 <E02.mp4>
uv run python tools/voiceprint.py 评测 <声纹聚类.json> <标注.json> --声线角色映射 <映射.json>
uv run python tools/diarize.py <视频> <声音记录.json> <新声纹聚类.json> --约束 lipsync --口型 <口型记录.json>
```

入库须有确认的角色归属，清楚且非待核对的音频；重复入库替换该集贡献。比对排除待定、疑似混人/重叠及待核对句，不确认人物，不改归属表。候选保存当前声音摘要；原文、时间、句号或声线变化后须重新比对。

声纹向量与各集均值只保存本机，不附到提示词、日志或报告。

### 派发与分段参数

`dispatch_role.py` 支持 `--提示词 <文件>`，默认正式提示词。全文验收自动执行整集文本检查与分段画面核对；结果旁的 `.coverage.json` 保存逐段实际读帧覆盖，`全文审查完成=false` 时不得标记全文完成。`segments.py` 支持 `--shots-per-segment`、`--soft-limit`、`--frame-budget`、`--context-frame-budget`，默认分别为 6、8、36、48。

### 旧数据兼容

`asr_cross.py` 命令直接报停用，不读取素材或启动服务。历史后端代码、试验数据和双路 Schema 读取仍保留，供旧报告复核。生产识别和派发不调用历史后端，不把旧“两路一致”转换为新人工确认。

### 角色卡派发资料

角色卡建立可使用 `dispatch_role.py 角色卡 --模式 建立`。资料装配包含声音、字幕、人物投影、帧、句级与轨迹级口型证据、声纹聚类，以及存在时的声纹候选；不附加逐帧口型曲线或声纹库原始向量。`证据状态` 逐项报告口型、声纹聚类和跨集候选的有效、缺失或过期状态；过期证据不附入授权资料。缺失不构成否定事实。`--提示词 <文件>` 用于提示词对比，模型沿用配置默认。
