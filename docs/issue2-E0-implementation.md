# Issue #2 E0：固定单集实现与离线验证

日期：2026-10-18。范围仅为基线冻结、最薄固定单集调度和离线合同。没有启动E1真实整集，没有真实模型、内容质量、费用、时效或整集Word验收结论。

## 1. 基线与改动

- 基线为`23f2f26080269c0bc081b11d72235a440ef7fb23`，原分支和本地`baseline/issue2-23f2f260`引用保留。基线代码/提示词/配置内容SHA见`baselines/issue2-23f2f260.json`；冻结清单已先单独提交。
- 本克隆没有原视频、本地模型或历史E01产物；没有下载或覆盖这些资料，没有读取凭据、修改issue、push、开PR或merge。
- `fixed_pipeline.py`只串阶段，复用既有派发、两步角色卡、审核apply、机械校验、独立全文审查及交付门禁。`fixed_steps.py`负责确定编号、落板和终态。
- `run_record.py`保存内容冻结、阶段账本及已知前后散列的多文件发布；`request_ledger.py`提供每个HTTP请求前的统一预算边界。`offline_episode.py`是显式模拟入口；`action_placement.py`只应用同次成稿返回的排布决定，不改事件事实。
- 修改现有派发模块以支持签名缓存、预算传递及无工具文本直连；既有工具主控/定向角色仍保留Pi。纯文本成稿和整集文本验收的授权内容已完整放入提示词，原本无工具，不迁移需要工具读取的路径。
- `run_episode()`及单集CLI继续默认原Pi主控；固定E0须显式`scheduler='固定'`/`--调度 固定`，尚未可真实运行。默认兼容测试恢复原调用，并增加API/CLI默认入口断言。

## 2. 单集合同

固定预处理 → 按段动作 → 确定合并并编号A、登记T → 角色卡合并/冻结/归属 → 显式审核或延期审核apply → 成稿＋结构化排布 → 机械检查 → 独立全文审查 → 正式门禁或待人工裁定。

- 没有审核文件时不会批准任何N。明确延期审核，仅保存N、候选归属和阻断问题。`--审核`指定的文件尚不存在时停在apply前，返回审核模板；模板本身不是批准。审核需绑定本次结果输入版本。
- 画面人物T使用第一条非空外观作为登记投影，出现镜头按源顺序保存；不同镜头描述不因此拆人，逐镜头原始证据保留。
- 身份已确认但一句中声音形态变化时，允许保留身份与确认依据、形态null及非空`形态待核对`，须有同句阻断和草稿Q标记。混人/重叠仍须存疑或回听拆句，不伪造词时间。任何未决形态继续阻断正式稿。
- 新人未审核的关键动作保留原确认事实、执行者引用和依据帧，以对应身份阻断挂起；草稿不把该动作写成确定事实。正式检查仍要求完整覆盖，不能靠删事件放行。
- 固定成稿须为每个呈现动作恰好返回一次排布；缺失或不合法进入既有格式重答，仍受请求预算限制。排布按现有Schema、人物、时序、覆盖及独立验收校验。
- 口型失败可记录为“缺失；不能作为否定证据”，不造未开口/非同人结论。仍未实现后台口型和完整预处理并行。

## 3. 隔离、冻结与恢复

显式固定调度的默认输出在`剧目/剧名/隔离运行/版本散列/剧名/E01/`。显式运行根目录中已有未冻结黑板会被拒绝。既有原集目录不被接管或覆盖。

冻结视频内容、配置散列、全部提示词、代码散列/提交、继承角色卡以及离线夹具版本。运行源码包含Python、JSON、TypeScript和字幕OCR的`tools/ocr/vocr.swift`；依赖清单`pyproject.toml`及`uv.lock`另存明确内容SHA，任一变化拒绝恢复。本地权重存在时保存文件SHA；不存在时为null，远程只知道接入模型标识，不能冻结不可观测的远程权重版本。

完成阶段复用；证据/产物被外部修改或出现未记账验收文件时拒绝复用。角色卡内部合并、归属批次和独立审查文本/画面分段均可按输入签名复用成功部分。只续当前失败阶段，不自动重跑整链或无限返修。

多文件发布先持久化目标的前后SHA和暂存内容，再逐个替换。apply在六个文件边界中断后可接续同一已验证计划，不重分C/Q；目标出现第三种独立编辑时拒绝覆盖。预处理硬中断保留未发布素材及既有子阶段记录，必须恢复并经过输入校验后才派发语义任务。这是进程中断合同，不声明抵抗磁盘损坏或主机掉电。

账本区分阶段起止、墙钟、输入产物SHA、结果、失败/恢复历史；每次请求另列角色/阶段/片段/格式尝试、模型、图片数、HTTP重试、请求SHA、官方usage及金额。未知保留null。进程硬中断时不可测的原尝试耗时不会补成0。

## 4. 真实预算的明确阻塞

E0尚未实现可公开配置的真实官方计价/账单接入，这是代码缺项，并非只有环境缺失。显式固定入口在任何预处理/付费调用前预检失败；生产默认已恢复原主控；不能用旧Pi入口绕过这个缺口并宣称有预算保护。

保留的计价接口要求预检、单请求费用上界预留和官方结算；当前仅用明确测试替身验证。2美元警告、5美元停止是保护阈值，不是价格预测或已实测美元硬限。逐HTTP挂点包括每个初答、格式重答和网络重试，并发共享同一剩余额度。官方usage缺失不补0，已知金额小计与未知总额分开。Pi内部多请求不可观测，预算路径拒绝该运行器。

进入E1仍需：可核实的计价/结算来源及上界约束、原视频、本地Qwen/声纹/口型环境、单集真实审核和独立验收。该阻塞没有被离线夹具掩盖。

## 5. 可复现命令与结果

环境：Linux、Python 3.12.14；使用未改动的`uv.lock`，以`UV_CACHE_DIR=/tmp/shot2script-uv-cache uv sync --frozen`安装同一隔离venv。没有安装无关组件或运行live模型。

### E0合同及受影响旧门禁

```bash
.venv/bin/python -m pytest -q \
  tests/test_fixed_episode.py tests/test_fixed_episode_edges.py tests/test_request_ledger.py \
  tests/test_action_placement.py tests/test_production_script.py tests/test_rolecard_apply.py \
  tests/test_rolecard_two_stage.py tests/test_formal_publication.py \
  tests/test_production_pipeline.py tests/test_production_pipeline_edges.py tests/test_t1_runners.py
```

结果：252通过。其中46个新增E0合同通过。包含两个CLI影子终态、模拟不能放行正式稿、显式审核/延期、六个apply中断边界、独立编辑拒绝、预处理硬中断、角色卡与审查内部成功部分复用、输入/产物篡改、OCR源码/依赖变更拒绝恢复、未知usage、初答/重答/HTTP重试及并发预算边界、身份/形态和混人句、口型缺失与默认旧入口兼容。

### 两个CLI离线终态

```bash
.venv/bin/python tools/run_episode.py 离线测试 1 --调度 固定 --视频 tests/fixtures/fixed_episode/input.mp4 --离线夹具 tests/fixtures/fixed_episode/pass.json --运行目录 /tmp/s2s-e0-pass
.venv/bin/python tools/run_episode.py 离线测试 1 --调度 固定 --视频 tests/fixtures/fixed_episode/input.mp4 --离线夹具 tests/fixtures/fixed_episode/pending.json --运行目录 /tmp/s2s-e0-pending
```

均退出0、真实远程请求0。第一个影子终态为正式交付候选，第二个为待人工裁定；两个实际终态都为待人工裁定，`正式交付:false`，没有有效正式稿。正式发布正分支另用明确的测试验收夹具调用真实门禁，不把任何模拟回执改成真实。

```bash
.venv/bin/python tools/run_episode.py 离线测试 1 --调度 固定 --视频 tests/fixtures/fixed_episode/input.mp4 --预检
```

预期退出1，清楚报告真实计价/结算适配缺失；预处理和模型调用均未启动。

### 全量基线对照

```bash
.venv/bin/python -m pytest -m 'not live' -q --tb=short
```

基线`23f2f260`单独`git archive`解包，`.venv`链接同一安装、`PYTHONPATH`指向其自身`tools`；改后在当前工作树执行。额外只读pytest报告插件收集nodeid，不修改、跳过或xfail测试。

- 基线：1540通过、24失败、5error、5跳过、2个live未选
- 改后：1586通过、24失败、5error、5跳过、2个live未选
- 新失败nodeid、旧失败消失及失败阶段变化的差集均为空
- 完整nodeid、同一依赖版本和计数见`baselines/issue2-offline-regression.json`

既有24失败包括：旧角色卡单阶段材料断言13＋3项、旧黑板文件集合1项、macOS绝对ffmpeg路径相关7项。5个setup error也依赖该ffmpeg路径。5跳过涉及MediaPipe、独立ASD/真实输入及Apple Vision。本轮没有为凑全绿删改这些断言或添加xfail。

`git diff --check`及`.venv/bin/python -m compileall -q tools/s2s`均通过。新增`fixed-episode-offline` CI job执行定向合同和旧门禁；尚未push或触发GitHub Actions，不能报告远程CI通过。

## 6. 未验证/未实施

- 没有真实视频冷启动、真实Qwen/声纹/口型、远程模型/Pi或整集审查
- 没有真实身份/动作/对白质量、官方费用、真实总墙钟/180秒或跨视频泛化结论
- 没有新整集正式Markdown/Word交付；既有Word生成/解析仅有离线回归
- 没有自动三轮返修、全剧固定编排、跨集总结、UI、性能对比或消融实验
- 独立代码审查基于`4a0983f`复跑249项并指出下列两项；修正后的固定commit另行复验，本记录不替代复验结论

## 7. 独立审查后的最小修正

- P1：原固定E0入口缺少公开可配置的真实计价接入，把它设为默认会让现有真实CLI永远阻断。已恢复API/CLI的原主控默认，固定E0改为显式实验入口；预算关闭保持不变，没有借绕过保护修复可用性。
- P2：运行内容冻结原先漏了Swift OCR源码和根目录依赖文件。已补这三项及各自变更拒绝恢复测试。基线冻结清单本来已包含三项，已从原`23f2f260`的git对象逐项核验，没有使用改后文件冒充基线。
