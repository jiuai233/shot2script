"""交叉核对确认记录与剧本；素材真实性仍由独立验收判断。"""

import argparse
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator

from s2s.coverage import check_coverage, normalize_text
from s2s.lint_script import (lint_text, parse_dialogue,
                             PENDING_SPEAKER_RE, DEFAULT_WORDS, _mask_names)
from s2s.validate import read_json, time_seconds, validate_file, SCHEMA_DIR
from s2s.config import PROJECT_ROOT


def review_version(directory):
    """保存审查输入的内容和帧版本；角色性格总结不改变身份判断。"""
    directory = Path(directory)
    version = {}
    for name in ('工作稿.md', '对照表.json', '镜头表.json', '片段表.json', '声音记录.json',
                 '字幕记录.json', '动作记录.json', '归属表.json', '临时人物.json',
                 '口型记录.json', '声纹聚类.json', '声纹候选.json', '本集人物.json'):
        path = directory / name
        version[name] = path.read_text(encoding='utf-8') if path.is_file() else None
    table = json.loads(version['镜头表.json'])
    video = Path(table['视频'])
    video = video if video.is_absolute() else PROJECT_ROOT / video
    version['视频版本'] = ({'路径': str(video.resolve()), '字节': video.stat().st_size,
                            '修改时间ns': video.stat().st_mtime_ns} if video.is_file() else None)
    roles_path = directory.parent / '角色卡.json'
    roles = read_json(roles_path)
    referenced = set()
    for name, fields in (('归属表.json', ('说话人',)), ('动作记录.json', ('执行者', '对象')),
                         ('临时人物.json', ('映射',))):
        for row in json.loads(version[name] or '[]'):
            referenced.update(row.get(field) for field in fields)
    local_people = json.loads(version['本集人物.json'] or '{"人物": []}')['人物']
    referenced.update(row['审核C'] for row in local_people if row['审核C'])
    version['角色卡'] = [{k: v for k, v in role.items() if k != '性格总结'} for role in roles
                       if role['id'] in referenced or role['名称'] in (version['工作稿.md'] or '')]
    version['帧'] = [[str(path.relative_to(directory)), path.stat().st_size, path.stat().st_mtime_ns]
                     for folder in ('frames', 'frames_dense')
                     for path in sorted((directory / '素材' / folder).rglob('*')) if path.is_file()]
    for name in ('00-数据约定.md', '06-验收.md'):
        version[name] = (PROJECT_ROOT / 'prompts' / name).read_text(encoding='utf-8')
    return version


def check_review(directory):
    """只有当前稿件与证据的完整独立审查可用于正式稿。"""
    directory = Path(directory)
    version = review_version(directory)
    segments = read_json(directory / '片段表.json')['片段']
    shots = read_json(directory / '镜头表.json')['镜头']
    expected = {}
    for segment in segments:
        wanted = set(segment['镜头']) | {segment.get('衔接前'), segment.get('衔接后')}
        expected[segment['id']] = {str((directory / '素材/frames' / frame['文件']).resolve())
                                   for shot in shots if shot['id'] in wanted for frame in shot['帧']}
    available = {str((directory / item[0]).resolve()) for item in version['帧']
                 if Path(item[0]).suffix.lower() == '.jpg'}
    actions = read_json(directory / '动作记录.json')
    required_evidence = {segment['id']: [] for segment in segments}
    for segment in segments:
        wanted = set(segment['镜头']) | {segment.get('衔接前'), segment.get('衔接后')}
        for action in actions:
            if action['确定度'] != '确认' or action['镜头'] not in wanted:
                continue
            for filename in action['依据帧']:
                candidates = {path for path in available if Path(path).name == filename}
                if not candidates:
                    return False
                required_evidence[segment['id']].append(candidates)
    sounds = read_json(directory / '声音记录.json')
    sentence_range = f"{sounds[0]['id']} 至 {sounds[-1]['id']}" if sounds else '无对白'
    scenes = re.findall(r'^场景\d+-\d+', version['工作稿.md'], re.M)
    review_schema = read_json(Path(__file__).with_name('dispatch') / 'role_schemas.json')['验收']
    candidates = []
    for path in (directory / '派发结果').rglob('*.coverage.json'):
        try:
            receipt = read_json(path)
            if receipt.get('模拟调用') is False and receipt.get('输入版本') == version:
                candidates.append((path, receipt))
        except (OSError, ValueError, AttributeError):
            continue
    if not candidates:
        return False
    # 同一版本采用最近一次真实审查，失败或残缺不能回退到旧通过。
    path, receipt = max(candidates, key=lambda item: item[0].stat().st_mtime_ns)
    try:
        review = read_json(path.with_name(path.name.removesuffix('.coverage.json') + '.json'))
        rows = receipt.get('片段', [])
        if not Draft202012Validator(review_schema).is_valid(review):
            return False
        return bool(version['视频版本'] is not None
                    and receipt.get('全文审查完成') is True and receipt.get('验收结果') == review
                    and review.get('模式') == '全文审查' and review.get('集') == int(directory.name[1:])
                    and review['审查范围']['行'] == f"1-{len(version['工作稿.md'].splitlines())}"
                    and review['审查范围']['镜头'] == f"{shots[0]['id']} 至 {shots[-1]['id']}"
                    and set(review['审查范围']['场景']) == set(scenes)
                    and review['审查范围']['对白句'] == sentence_range
                    and review.get('结论') == '通过' and not review.get('问题')
                    and review.get('统计') == {'阻断': 0, '规范': 0}
                    and not any(row.get('结果') != '通过' for row in review.get('复核', []))
                    and expected and [row['片段'] for row in rows] == list(expected)
                    and all(row.get('状态') == '完成' and expected[row['片段']]
                            and expected[row['片段']].issubset(set(row.get('已读帧', [])))
                            and set(row.get('已读帧', [])).issubset(available)
                            and all(paths.intersection(row.get('已读帧', []))
                                    for paths in required_evidence[row['片段']])
                            and len(set(row.get('已读帧', []))) == len(row.get('已读帧', []))
                            == row.get('已读帧数') == row.get('授权帧数') for row in rows))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def issue_closed(issue):
    return issue['状态'] in ('已修正', '已裁定') and bool((issue['复核结果'] or '').strip())


def check_script(directory: str | Path, script='工作稿.md', *, formal=False) -> dict:
    directory = Path(directory)
    script = Path(script)
    script = script if script.is_absolute() else directory / script
    errors = []
    result = {'通过': False, '错误': errors, '需素材验收': [
        '动作、神态和情绪是否有实际依据，素材中的关键事件是否遗漏',
        '执行者、对象及说话人身份是否正确',
        '短动作是否应入对白括号，长过程及听者反应是否应独立写 △']}

    def fail(rule, position, message):
        errors.append({'检查项': rule, '位置': position, '问题': message, '级别': '阻断'})

    data = {}
    try:
        text = script.read_text(encoding='utf-8')
        for name in ('镜头表', '片段表', '声音记录', '动作记录', '归属表', '对照表', '待核对', '问题清单', '临时人物', '本集人物'):
            path = directory / (name + '.json')
            if name in ('临时人物', '本集人物') and not path.exists():
                data[name] = [] if name == '临时人物' else {'人物': []}
                continue
            checked = validate_file(path, name)
            for error in checked['错误']:
                fail(name, error['位置'], error['问题'])
            if checked['通过']:
                data[name] = read_json(path)
        roles_path = directory / '角色卡.json'
        if not roles_path.exists():
            roles_path = directory.parent / '角色卡.json'
        roles = read_json(roles_path)
        for error in Draft202012Validator(read_json(SCHEMA_DIR / '角色卡.schema.json')).iter_errors(roles):
            fail('角色卡结构', list(error.absolute_path), error.message)
    except (OSError, ValueError) as exc:
        fail('输入文件', {}, str(exc))
    if errors:
        return result

    if len({role['id'] for role in roles}) != len(roles):
        fail('角色卡编号重复', {}, '不同角色不能共用稳定角色 ID')
    for error in lint_text(text, roles, formal=formal)['错误']:
        fail(error['检查项'], {'行': error['行']}, error['命中词'])
    for error in check_coverage(directory, script)['错误']:
        fail(error['类型'], error['位置'], error['问题'])

    roles_by_id = {role['id']: role for role in roles}
    temporary = {row['id']: row['映射'] for row in data['临时人物']}
    pending = {row['id']: row for name in ('待核对', '问题清单') for row in data[name] if not issue_closed(row)}
    lines = text.splitlines()
    actions = {row['id']: row for row in data['动作记录']}
    sound = {row['id']: row for row in data['声音记录']}
    attribution = {row['句']: row for row in data['归属表']}
    names = {role['名称'] for role in roles}
    action_words = read_json(DEFAULT_WORDS)['动作词']

    local_people = {row['id']: row for row in data['本集人物']['人物']}

    def person(ident):
        ident = temporary.get(ident) or ident
        if ident in local_people:
            ident = local_people[ident]['审核C']
        return roles_by_id.get(ident)

    def confirmed(ident):
        role = person(ident)
        return role is not None and role['状态'] == '已确认'

    def has_pending(sentence):
        return any(row['位置'].get('句') == sentence for row in pending.values())

    for number, line in enumerate(lines, 1):
        for match in re.finditer(r'【(?:说话人)?待核对 (E[0-9]{2,}-Q[0-9]{3,})】', line):
            if match[1] not in pending:
                fail('待核对引用', {'行': number}, '标记未引用有效未决问题：' + match[1])
    for sentence in sound:
        row = attribution.get(sentence)
        if row is None or row['确定度'] != '确认' or not confirmed(row['说话人']):
            if formal or not has_pending(sentence):
                fail('对白归属未确认', {'句': sentence}, '正式稿须有已确认身份；草稿须登记对应句的待核对问题')
    for action in actions.values():
        if action['关键'] and action['确定度'] != '确认':
            registered = any(row['位置'].get('镜头') == action['镜头'] for row in pending.values())
            if formal or not registered:
                fail('关键动作未确认', {'动作': action['id']}, '存疑关键动作须保留问题，不能作为正式稿事实')

    rendered_events = {}
    for row in data['对照表']:
        for ident in row['动作']:
            action = actions.get(ident)
            if action is None:
                continue
            signature = tuple(action[k] for k in ('镜头', '起', '止', '执行者', '对象', '物件')) + (normalize_text(action['动作']),)
            previous = rendered_events.get(signature)
            if previous is not None and previous != ident:
                fail('重复动作事件', {'行': row['行'], '动作': ident}, '同一时间、人物及内容的动作事件已呈现：' + previous)
            rendered_events[signature] = ident

    latest_start = -1
    for row in sorted(data['对照表'], key=lambda row: row['行']):
        number = row['行']
        if number > len(lines):
            continue
        line = lines[number - 1].strip()
        dialogue = parse_dialogue(line)
        intervals = []
        for ident in row['句']:
            if ident not in sound:
                continue
            sentence = sound[ident]
            intervals.append((time_seconds(sentence['起']), time_seconds(sentence['止'])))
            owner = attribution.get(ident)
            if not dialogue:
                continue
            speaker, parenthesis, _ = dialogue
            if owner is None or owner['确定度'] != '确认' or not confirmed(owner['说话人']):
                if not PENDING_SPEAKER_RE.fullmatch(speaker):
                    fail('存疑归属冒充确认', {'行': number, '句': ident}, '使用已登记的说话人待核对标记，不猜测姓名')
                elif not any(q in pending and pending[q]['位置'].get('句') == ident
                             for q in re.findall(r'E[0-9]{2,}-Q[0-9]{3,}', speaker)):
                    fail('待核对引用', {'行': number, '句': ident}, '说话人标记须指向该句的问题')
                continue
            if speaker != person(owner['说话人'])['名称']:
                fail('说话人错归属', {'行': number, '句': ident}, '正文说话人与确认归属不一致')
            shape = re.match(r'^(画外|心声|电话|视频|广播)(?:[，,、]|$)', parenthesis)
            actual_shape = ('设备' if shape and shape[1] in ('电话', '视频', '广播') else
                            shape[1] if shape else '画内')
            if actual_shape != owner['形态']:
                fail('声音形态不一致', {'行': number, '句': ident}, '正文标记与确认声音形态不一致')

        for ident in row['动作']:
            action = actions.get(ident)
            if action is None:
                continue
            intervals.append((time_seconds(action['起']), time_seconds(action['止'])))
            if action['镜头'] not in row['镜头']:
                fail('动作镜头引用', {'行': number, '动作': ident}, '行映射未包含动作所属镜头')
            actor = person(action['执行者'])
            if action['确定度'] != '确认' or not confirmed(action['执行者']):
                fail('存疑动作进入正文', {'行': number, '动作': ident}, '正文动作须有确认记录和已确认角色')
                continue
            description = dialogue[1] if dialogue else line.removeprefix('△').strip()
            placement = action.get('排布')
            if placement is None:
                if formal:
                    fail('动作排布未确认', {'行': number, '动作': ident}, '须审核动作与对白关系并保存排布依据，不按字数或关键词自动决定')
            else:
                actual_position = '对白括号' if dialogue else '独立△'
                if placement['位置'] != actual_position:
                    fail('动作排布不一致', {'行': number, '动作': ident}, '确认位置为 ' + placement['位置'])
                if dialogue and placement['句'] not in row['句']:
                    fail('动作关联句不一致', {'行': number, '动作': ident}, '括号动作未对应排布记录中的对白句')
            expression = (placement or {}).get('表述', action['动作'])
            for role_id, role in roles_by_id.items():
                expression = expression.replace(role_id, role['名称'])
            if not normalize_text(expression) or normalize_text(expression) not in normalize_text(description):
                fail('动作呈现待核实', {'行': number, '动作': ident}, '正文未包含确认动作表述；须保留事件，改写需先审核排布中的表述')
            if dialogue:
                if dialogue[0] != actor['名称']:
                    fail('括号动作错归属', {'行': number, '动作': ident}, '括号动作执行者不是该行说话人')
                linked = (placement or {}).get('句')
                sentences = [sound[s] for s in row['句'] if s in sound and (linked is None or s == linked)]
                # 排布审核决定能否简写；时间仍须真实重叠，不能提前呈现说完后的动作。
                a, b = time_seconds(action['起']), time_seconds(action['止'])
                if sentences and not any((time_seconds(s['起']) <= a == b < time_seconds(s['止'])) or
                                         (a < time_seconds(s['止']) and time_seconds(s['起']) < b <=
                                          time_seconds(s['止'])) for s in sentences):
                    fail('动作对白时序', {'行': number, '动作': ident}, '动作不在关联对白时段内，不能合并进括号')
                if any(attribution.get(s, {}).get('形态') != '画内' for s in row['句']):
                    fail('画外动作合并', {'行': number, '动作': ident}, '非画内对白不能合并可见动作')
                if not description:
                    fail('动作未呈现', {'行': number, '动作': ident}, '对白行引用动作但括号为空')
            else:
                first = next((name for name in sorted(names, key=len, reverse=True) if description.startswith(name)), None)
                if actor['名称'] not in description or (len(row['动作']) == 1 and first and first != actor['名称']):
                    fail('动作执行者不一致', {'行': number, '动作': ident}, '正文人物与确认执行者不一致')
            target = person(action['对象'])
            if target is not None and target['名称'] not in description:
                fail('动作对象未呈现', {'行': number, '动作': ident}, '描述缺少确认的动作对象名称')
        if dialogue and not row['动作']:
            masked = _mask_names(dialogue[1], names)
            if any(word in masked for word in action_words):
                fail('括号动作缺少记录', {'行': number}, '可见动作括号没有对应的确认动作记录')
        if intervals:
            start, end = min(t[0] for t in intervals), max(t[1] for t in intervals)
            if end < latest_start:
                fail('事件时序错误', {'行': number}, '该行事件完全早于前文已呈现的事件')
            latest_start = max(latest_start, start)

    if formal:
        for row in pending.values():
            fail('未决问题', {'问题': row['id']}, '问题须修正或裁定并留下复核结果')
        for role in [*roles, *[r for r in local_people.values() if r['审核C'] is None]]:
            if role.get('状态') != '已确认':
                descriptions = [line.split('：', 1)[1] if line.startswith('人物：') else
                                parse_dialogue(line)[0] + parse_dialogue(line)[1] if parse_dialogue(line)
                                else line if line.startswith('△') and not line.startswith('△ 画面字幕：') else ''
                                for line in lines]
                if any(role['名称'] in description for description in descriptions):
                    fail('角色身份未确认', {'角色': role['id']}, '待确认人物不能进入正式稿')
        for sentence in sound.values():
            if sentence.get('核对', {}).get('状态') == '待核对':
                fail('声音待核对', {'句': sentence['id']}, '声音异常尚未核实')
        try:
            if script.read_text(encoding='utf-8') != (directory / '工作稿.md').read_text(encoding='utf-8'):
                fail('交付版本不一致', {}, '正式稿须与实际验收的工作稿完全一致')
            if not check_review(directory):
                fail('独立验收缺失或过期', {}, '需要当前稿件和证据的完整非模拟审查，且结论通过')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            fail('独立验收输入', {}, str(exc))
    result['通过'] = not errors
    return result


def main():
    parser = argparse.ArgumentParser(description='检查人物、动作、声音与剧本的一致性')
    parser.add_argument('本集目录')
    parser.add_argument('--稿', default='工作稿.md')
    parser.add_argument('--正式稿', action='store_true')
    args = parser.parse_args()
    result = check_script(args.本集目录, args.稿, formal=args.正式稿)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['通过'] else 1
