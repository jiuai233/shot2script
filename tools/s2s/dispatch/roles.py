"""子 agent 派发、重答、日志与片段结果复用。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time

from s2s import agent, config
from s2s.runners import _safe
from s2s.quality import review_version
from .materials import assemble, dumps, load
from .results import extract_result, validate_result
from .transport import decode_response, pi_options


class DispatchError(RuntimeError):
    pass


def settings_for(role, package):
    settings = config.get_agent(role)
    # 有帧时沿用 vision 的图片与运行器参数；所有角色仍只用 DeepSeek。
    if package.images:
        settings.update(config.get_vision())
    config.check_model(settings['model'])
    return settings


def cached_result(episode, role, package, signature):
    if role != '动作识别' or package.mode != '片段识别':
        return None
    log = episode / '派发日志.jsonl'
    if not log.is_file():
        return None
    for line in reversed(log.read_text(encoding='utf-8').splitlines()):
        try:
            row = json.loads(line)
        except ValueError:
            continue  # 中断可能留下未写完的最后一行。
        if row.get('角色') != role or row.get('片段') != package.segment['id']:
            continue
        if not row.get('成功') or row.get('输入版本') != signature:
            return None
        try:
            result = load(Path(row['结果文件']))
            if not validate_result(role, result, episode, package):
                return result
        except (OSError, ValueError, TypeError, KeyError):
            return None
    return None


def merge_review_issues(parts):
    issues, seen, events = [], {}, {}
    for part in parts:
        for source in part['问题']:
            issue = {**source}
            position = issue['位置']
            location = position.get('行', (position.get('镜头'), position.get('时间')))
            kind = (issue['类型'],)
            key = (*kind, location, ''.join(issue['原文'].split()))
            event = (*kind, position.get('镜头'), position.get('时间')) if position.get('镜头') and position.get('时间') else None
            target = seen.get(key) or (events.get(event) if event else None)
            if target is None:
                target = issue
                issues.append(target)
            else:
                if issue['级别'] == '阻断':
                    target['级别'] = '阻断'
                for field in ('原文', '依据', '修正要求'):
                    if issue[field] not in target[field]:
                        target[field] += '\n' + issue[field]
            seen[key] = target
            if event:
                events[event] = target
    return issues


def review_full(episode, variables, output, *, timeout, call, validator, prompt_file):
    """文本整集检查一次，画面逐段检查；覆盖回执独立保存。"""
    version = review_version(episode)
    folder = output.parent / (output.stem + '-分段')
    folder.mkdir(parents=True, exist_ok=True)
    parts, coverage = [], []
    def run_stage(stage, segment=None):
        name = segment['id'] if segment else '文本'
        path = folder / (name + '.json')
        v = {**variables, **({'片段号': name} if segment else {})}
        receipt = dispatch('验收', episode, v, path, mode='全文审查', timeout=timeout,
                           call=call, validator=validator, prompt_file=prompt_file, _review_stage=stage)
        return load(path), receipt, path

    text, _, _ = run_stage('文本')
    parts.append(text)
    segments = load(episode / '片段表.json')['片段']
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = pool.map(lambda segment: run_stage('画面', segment), segments)
        for segment, (part, receipt, path) in zip(segments, results):
            parts.append(part)
            coverage.append({'片段': segment['id'], '状态': '完成' if receipt['画面完整'] else '画面未完整审查',
                             '授权帧数': receipt['授权帧数'], '已读帧数': receipt['已读帧数'],
                             '已读帧': receipt['已读帧'], '结果文件': str(path)})
    return finish_review(episode, output, parts, coverage, version=version,
                         simulated=call is not None or validator is not None)


def finish_review(episode, output, parts, coverage, *, version, simulated=False):
    issues = merge_review_issues(parts)
    expected = [row['id'] for row in load(episode / '片段表.json')['片段']]
    complete = bool(expected) and [c['片段'] for c in coverage] == expected and all(c['状态'] == '完成' for c in coverage)
    unchanged = version == review_version(episode)
    shots = load(episode / '镜头表.json')['镜头']
    lines = (episode / '工作稿.md').read_text(encoding='utf-8').splitlines()
    result = {'模式': '全文审查', '集': int(episode.name[1:]),
              '审查范围': {'行': f'1-{len(lines)}', '场景': list(dict.fromkeys(s for part in parts for s in part['审查范围']['场景'])),
                           '镜头': f"{shots[0]['id']} 至 {shots[-1]['id']}" if complete else '画面未完整审查',
                           '对白句': parts[0]['审查范围']['对白句']},
              '问题': issues, '复核': [],
              '统计': {level: sum(i['级别'] == level for i in issues) for level in ('阻断', '规范')},
              '结论': '通过' if complete and unchanged and all(p['结论'] == '通过' for p in parts) and not issues else '不通过'}
    output.write_text(dumps(result) + '\n', encoding='utf-8')
    unique_read = {p for c in coverage for p in c['已读帧']}
    coverage_result = {'全文审查完成': complete, '片段': coverage, '实际读取不同帧数': len(unique_read),
                       '输入版本': version, '输入未变': unchanged, '模拟调用': simulated, '验收结果': result}
    output.with_suffix('.coverage.json').write_text(dumps(coverage_result) + '\n', encoding='utf-8')
    return {'通过': True, '结果文件': str(output), '全文审查完成': complete, '画面完整': complete}


def dispatch(role: str, episode: str | Path, variables: dict, output: str | Path,
             *, mode=None, scope=None, timeout=180, call=None, validator=None, prompt_file=None, _review_stage=None):
    episode, output = Path(episode).resolve(), Path(output).resolve()
    if role == '验收' and (mode or variables.get('模式', '全文审查')) == '全文审查' and _review_stage is None:
        return review_full(episode, variables, output, timeout=timeout, call=call, validator=validator, prompt_file=prompt_file)
    started = time.monotonic()
    package = None
    retries, success, skipped, usage = 0, False, False, {}
    coverage_complete, read_paths, missing = True, set(), set()
    settings = config.get_agent(role)
    row = {'时间': datetime.now(timezone(timedelta(hours=8))).isoformat(), '角色': role,
           '模式': mode or variables.get('模式', ''), '片段': variables.get('片段号'),
           '输入文件': [], '模型': settings.get('model'), '结果文件': str(output)}
    try:
        package = assemble(role, episode, variables, mode, scope, prompt_file=prompt_file, review_stage=_review_stage)
        settings = settings_for(role, package)
        signature = package.signature()
        signature['模型'] = settings
        row.update({'模式': package.mode, '范围': package.scope,
                    '片段': package.segment['id'] if package.segment else None,
                    '输入文件': [str(p) for p in [*package.files, *package.images]],
                    '输入版本': signature, '模型': settings['model']})
        result = cached_result(episode, role, package, signature)
        skipped = result is not None
        validate = validator or validate_result
        prompt = package.prompt
        if result is None:
            with pi_options(settings, read_paths=package.images) as options:
                for attempt in range(3):
                    retries = attempt
                    attempt_started = time.monotonic()
                    attempt_row = {'尝试': attempt + 1}
                    row.setdefault('尝试记录', []).append(attempt_row)
                    try:
                        config.check_model(settings['model'])
                        raw = (call or agent.run)(role, prompt, attachments=[], images=[],
                                               **{**settings, **options, 'timeout': timeout})
                        response = decode_response(raw)
                    finally:
                        attempt_row['耗时秒'] = round(time.monotonic() - attempt_started, 3)
                    read_paths = {str(Path(p).resolve()) for p in response.reads}
                    authorized = {str(p) for p in package.images}
                    read_paths &= authorized
                    missing = authorized - read_paths
                    attempt_row.update({'读取图片数': len(read_paths), '已读帧': sorted(read_paths),
                                        '授权帧数': len(authorized), '未读帧': sorted(missing),
                                        'token用量': response.usage or None})
                    row['读取图片数'] = len(read_paths)
                    for key, value in response.usage.items():
                        usage[key] = usage.get(key, 0) + value
                    try:
                        result = extract_result(role, response.text)
                        errors = validate(role, result, episode, package)
                        coverage_complete = not missing
                        if call is None and role in ('动作识别', '验收') and missing:
                            if attempt == 2 and role == '验收' and not errors:
                                result['结论'] = '不通过'
                                row['画面状态'] = '画面未完整审查'
                            else:
                                errors.append('尚未使用 read 读取全部帧：' + '、'.join(sorted(missing)))
                    except (ValueError, TypeError, KeyError) as exc:
                        errors = [str(exc)]
                    if not errors:
                        break
                    debug_dir = output.parent / '校验失败'
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    debug_path = debug_dir / f'{output.stem}-{time.time_ns()}-{attempt}.txt'
                    debug_path.write_text(response.text + '\n\n校验错误：\n' + dumps(errors), encoding='utf-8')
                    row.setdefault('失败返回', []).append(str(debug_path))
                    if attempt == 2:
                        raise DispatchError('格式校验失败，已重答 2 次：' + '; '.join(errors))
                    prompt = (package.prompt + '\n\n## 上次返回\n' + response.text
                              + '\n\n## 校验错误\n' + dumps(errors)
                              + '\n请修正这些错误，重新返回完整结果，不省略其他字段。每次重答都是新调用，必须重新读取全部授权帧，不能只补读遗漏帧。')
        output.parent.mkdir(parents=True, exist_ok=True)
        # 主控决定目标路径；先校验成功才替换旧结果。
        temporary = output.with_name(output.name + '.new')
        temporary.write_text(dumps(result) + '\n', encoding='utf-8')
        temporary.replace(output)
        success = True
        return {'通过': True, '结果文件': str(output), '跳过': skipped, '重答次数': retries,
                '画面完整': coverage_complete, '授权帧数': len(package.images),
                '已读帧数': len(read_paths), '已读帧': sorted(read_paths)}
    except Exception as exc:
        row['错误'] = _safe(str(exc), config.get_environment())
        raise
    finally:
        row.update({'耗时秒': round(time.monotonic() - started, 3), '重答次数': retries,
                    '成功': success, '跳过': skipped, 'token用量': usage or None})
        if episode.is_dir():
            with (episode / '派发日志.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='按资料隔离规则派发一个子 agent')
    parser.add_argument('角色', choices=('动作识别', '角色卡', '成稿', '验收'))
    parser.add_argument('--集', required=True)
    parser.add_argument('--变量', required=True, help='JSON 对象；动作识别必须包含片段号')
    parser.add_argument('--模式')
    parser.add_argument('--范围')
    parser.add_argument('--输出', required=True)
    parser.add_argument('--提示词', help='对比试验提示词文件，默认使用正式提示词')
    parser.add_argument('--超时', type=float, default=180)
    args = parser.parse_args()
    try:
        variables = load(Path(args.变量))
        if not isinstance(variables, dict):
            raise ValueError('变量文件必须为 JSON 对象')
        result = dispatch(args.角色, args.集, variables, args.输出,
                          mode=args.模式, scope=args.范围, timeout=args.超时, prompt_file=args.提示词)
    except (OSError, ValueError, RuntimeError) as exc:
        result = {'通过': False, '错误': _safe(str(exc), config.get_environment())}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['通过'] else 1
