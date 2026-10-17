"""主控只获得工具说明与当前状态，流程判断留在角色提示词中。"""

import argparse
import json
from pathlib import Path
import re

from s2s import agent, config
from s2s.board import initialize_board
from .materials import PROMPTS, dumps, load, render
from .transport import pi_options, decode_response

EPISODE_TOOLS = '''
## 固定工具
所有命令从项目根目录运行，Python 使用 uv run python。
- tools/dispatch_role.py <动作识别|角色卡|成稿|验收> --集 <本集目录> --变量 <vars.json> [--模式 <模式>] [--范围 <问题及位置>] --输出 <结果.json>
  变量 JSON 为对象；动作识别必须含“片段号”。加密复看时加入“加密帧索引”数组，填相对于本集的 素材/frames_dense/<时间段>/帧.json。
  动作结果统一放到 <本集目录>/派发结果/动作识别/<片段号>.json，其他子角色结果也放到派发结果下。
  子 agent 无写工具，只返回结果。你审核后合并、编号并写黑板；成稿返回对象含工作稿和对照表，分别写文件。
  动作片段已有通过校验且输入未变的成功结果时自动跳过；定向复看不复用缓存。
  角色卡有帧时自动使用 vision 配置；总结不得读工作稿。验收看不到动作记录、归属表。
  全文验收自动执行整集文本检查和逐片段画面核对；必须检查返回的“全文审查完成”和结果旁的 .coverage.json，画面未完整审查时不能标记全文审查完成。对比试验可用 --提示词 <文件>，默认仍用正式提示词。
  复核变量可含“改动对照”；跨集模式必须含“跨集集号”整数数组。
- tools/validate.py <JSON文件> [--type <数据类型>]
- tools/coverage.py <本集目录>
- tools/lint_script.py <工作稿.md>
- tools/dense_frames.py <视频> <起> <止> <帧率> <输出目录>
- tools/asr_window.py <视频> <起> <止>
- tools/run_episode.py <剧名> <集号> --完成步骤 <1到9>
  每步完成并把结果写入黑板后运行这条命令；不会启动新模型。状态.json 的“步骤”为1到9的字符串键与布尔值。
所有工具返回 JSON；必须读取通过/错误字段，命令退出不等于任务验收通过。
当前 validate.py 的状态 Schema 尚未加入“步骤”，会报额外字段错误；不要为消除它删掉步骤。
所有识图交给使用 vision 配置的子角色；主控不直接读取图片，包括冲突仲裁。
不要读取 .env、其他程序的凭据文件，不执行部署或线上变更。
'''
SERIES_TOOLS = '''
## 固定工具
所有命令从项目根目录运行，Python 使用 uv run python。
- tools/run_episode.py <剧名> <集号> [--模式 完整转写|定向复核] [--复核要求 <具体变更与位置>]
  按集顺序派发，等本集结束后读取状态.json；不要因为子进程正常退出就把本集标记为验收通过。
- tools/dispatch_role.py 验收 --集 <参考本集目录> --变量 <vars.json> --模式 跨集 --输出 <结果.json>
  vars.json 必须含“跨集集号”整数数组，只提供人物行、首尾两个场景及结尾衔接，并开放相关帧。
- tools/docx_build.py <合并稿.md> <输出.docx> <标题>
- tools/docx_check.py <输出.docx> <集数>
- tools/validate.py <JSON文件> [--type <数据类型>]
不要读取各集完整素材或正文；跨集验收交给使用 vision 配置的验收子 agent，不直接识图。
不要读取 .env、其他程序的凭据文件，不执行部署或线上变更。
'''


def episode_number(value):
    text = str(value)
    if not re.fullmatch(r'(?:E)?[0-9]+', text) or int(text.removeprefix('E')) < 1:
        raise ValueError('集号须为正整数或 E01')
    return int(text.removeprefix('E'))


def ensure_state(series: str, number: int, root: Path):
    if not series.strip() or series in ('.', '..') or '/' in series or '\\' in series:
        raise ValueError('剧名须为单个目录名')
    episode = root / series / f'E{number:02d}'
    path = episode / '状态.json'
    if not path.exists():
        initialize_board(series, number, root=root)
    state = load(path)
    if state.get('集') != number:
        raise ValueError('状态.json 集号与目录不一致')
    if '步骤' not in state:
        state['步骤'] = {str(i): False for i in range(1, 10)}
        path.write_text(dumps(state) + '\n', encoding='utf-8')
    if set(state['步骤']) != {str(i) for i in range(1, 10)} or any(type(v) is not bool for v in state['步骤'].values()):
        raise ValueError('步骤必须为1到9的字符串键与布尔值')
    return episode, state


def mark_step(series, number, step, *, root=None):
    if step not in range(1, 10):
        raise ValueError('步骤须为1到9')
    episode, state = ensure_state(series, episode_number(number), Path(root or config.PROJECT_ROOT / '剧目'))
    state['步骤'][str(step)] = True
    (episode / '状态.json').write_text(dumps(state) + '\n', encoding='utf-8')
    return {'通过': True, '步骤': state['步骤']}


def episode_prompt(series, number, *, mode='完整转写', review='', root=None):
    if mode not in ('完整转写', '定向复核'):
        raise ValueError('任务模式须为完整转写或定向复核')
    if mode == '定向复核' and not review.strip():
        raise ValueError('定向复核必须提供 --复核要求')
    number = episode_number(number)
    root = Path(root or config.PROJECT_ROOT / '剧目')
    episode, state = ensure_state(series, number, root)
    previous = root / series / f'E{number-1:02d}' / '状态.json'
    ending = load(previous).get('结尾衔接', '') if previous.is_file() else ''
    prompt = render((config.PROJECT_ROOT / 'prompts' / PROMPTS['单集主控']).read_text(encoding='utf-8'),
                    {'剧名': series, '集号': f'{number:02d}', '上集结尾衔接': ending,
                     '完整转写 | 定向复核': mode,
                     '全剧主控给出的变更内容和受影响位置，完整转写时为空': review})
    prompt += EPISODE_TOOLS + '\n## 当前状态\n' + dumps(state)
    files = [str(p.relative_to(episode)) for p in sorted(episode.rglob('*'))
             if p.is_file() and '会话' not in p.relative_to(episode).parts]
    prompt += '\n## 已有文件\n' + dumps(files)
    prompt += '\n从状态中尚未完成的步骤继续；不要重复派发已有完成结果的片段。第8和第9步是条件分支。'
    if mode == '定向复核':
        prompt += '\n本轮执行定向复核流程；历史完整转写步骤已完成不代表本次复核已完成。'
    prompt += '\n\n## 数据约定\n' + (config.PROJECT_ROOT / 'prompts/00-数据约定.md').read_text(encoding='utf-8')
    return episode, prompt


def start_controller(role, prompt, session_dir, *, timeout=900, call=None):
    settings = config.get_agent(role)
    config.check_model(settings['model'])
    with pi_options(settings, controller=True, session_dir=session_dir) as options:
        raw = (call or agent.run)(role, prompt, **{**settings, **options, 'timeout': timeout})
    response = decode_response(raw)
    return {'通过': True, '会话目录': str(session_dir), '返回': response.text, 'token用量': response.usage or None}


def episode_main():
    parser = argparse.ArgumentParser(description='启动可断点续做的单集主控 pi 会话')
    parser.add_argument('剧名')
    parser.add_argument('集号')
    parser.add_argument('--模式', choices=('完整转写', '定向复核'), default='完整转写')
    parser.add_argument('--复核要求', default='')
    parser.add_argument('--完成步骤', type=int, choices=range(1, 10))
    parser.add_argument('--超时', type=float, default=900)
    args = parser.parse_args()
    try:
        if args.完成步骤 is not None:
            result = mark_step(args.剧名, args.集号, args.完成步骤)
        else:
            episode, prompt = episode_prompt(args.剧名, args.集号, mode=args.模式, review=args.复核要求)
            result = start_controller('单集主控', prompt, episode / '会话', timeout=args.超时)
    except (OSError, ValueError, RuntimeError) as exc:
        result = {'通过': False, '错误': str(exc)}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['通过'] else 1


def series_main():
    parser = argparse.ArgumentParser(description='启动全剧主控 pi 会话')
    parser.add_argument('剧名')
    parser.add_argument('集数', type=int)
    parser.add_argument('--超时', type=float, default=900)
    args = parser.parse_args()
    try:
        initialize_board(args.剧名, args.集数)
        series = config.PROJECT_ROOT / '剧目' / args.剧名
        prompt = render((config.PROJECT_ROOT / 'prompts' / PROMPTS['全剧主控']).read_text(encoding='utf-8'),
                        {'剧名': args.剧名, '集数': args.集数, '素材目录': str(config.PROJECT_ROOT / 'videos')})
        prompt += SERIES_TOOLS + '\n## 全剧当前状态\n' + dumps(load(series / '全剧状态.json'))
        for number in range(1, args.集数 + 1):
            _, state = ensure_state(args.剧名, number, config.PROJECT_ROOT / '剧目')
            prompt += f'\n## E{number:02d} 当前状态\n' + dumps(state)
        prompt += '\n## 数据约定\n' + (config.PROJECT_ROOT / 'prompts/00-数据约定.md').read_text(encoding='utf-8')
        result = start_controller('全剧主控', prompt, series / '会话', timeout=args.超时)
    except (OSError, ValueError, RuntimeError) as exc:
        result = {'通过': False, '错误': str(exc)}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['通过'] else 1
