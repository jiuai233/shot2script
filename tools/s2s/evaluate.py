"""人工标注与正文的可复查机械评测；动作语义结果为人工复核候选。"""

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re

from jsonschema import Draft202012Validator

from .coverage import normalize_text
from .lint_script import DEFAULT_WORDS, parse_dialogue
from .timefmt import to_sec

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / 'eval' / '标注.schema.json'
METRICS = ('台词逐字一致率', '说话人错归率', '关键动作漏写率', '无据动作率', '人物错配率')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_annotation(data: dict, *, complete=False) -> None:
    errors = sorted(Draft202012Validator(read_json(SCHEMA)).iter_errors(data), key=lambda e: str(list(e.path)))
    if errors:
        error = errors[0]
        raise ValueError(f'标注 Schema：{list(error.path)} {error.message}')
    people = {row['id'] for row in data['人物']}
    for key in ('人物', '台词', '关键动作'):
        seen = set()
        for row in data[key]:
            if row['id'] in seen:
                raise ValueError(f'标注编号重复：{row["id"]}')
            seen.add(row['id'])
            if '起' in row and to_sec(row['止']) < to_sec(row['起']):
                raise ValueError(f'{row["id"]} 结束早于开始')
            for span in row.get('出现', []):
                start, end = span.split('-')
                if to_sec(end) < to_sec(start):
                    raise ValueError(f'{row["id"]} 出现时间倒置')
            for field in ('说话人', '执行者', '对象'):
                value = row.get(field)
                if value and re.fullmatch(r'H[0-9]+', value) and value not in people:
                    raise ValueError(f'{row["id"]} 引用未登记人物：{value}')
            if data.get('校对完成') and (row.get('待核对') or
                    (key == '台词' and (not row['说话人'] or not row['形态'])) or
                    (key == '关键动作' and not row['执行者'])):
                raise ValueError(f'{row["id"]} 仍有待核对内容')
    if complete and not data.get('校对完成', False):
        raise ValueError('标注尚未人工校对完成，不能作为评测答案；请在工作表完成回看、回听后勾选校对完成')


def similarity(left, right):
    left, right = normalize_text(left or ''), normalize_text(right or '')
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def action_similarity(left, right):
    a, b = normalize_text(left or ''), normalize_text(right or '')
    if a and b and (a in b or b in a):
        return 1.0
    return similarity(left, right)


def nearby(left, right, tolerance):
    return (left.get('起') is not None and right.get('起') is not None and
            min(to_sec(left['止']), to_sec(right['止'])) - max(to_sec(left['起']), to_sec(right['起'])) >= -tolerance)


def align_dialogue(gold, actual, tolerance, threshold, max_span):
    """按播放顺序配对；只有逐字相同才跨断句合并，避免一条错句牵连邻句。"""
    n, m = len(gold), len(actual)
    dp = [[-math.inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    def update(i, j, score, previous):
        if score > dp[i][j]:
            dp[i][j], back[i][j] = score, previous

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == -math.inf:
                continue
            if i < n:
                update(i + 1, j, dp[i][j], (i, j))
            if j < m:
                update(i, j + 1, dp[i][j], (i, j))
            for a in range(1, min(max_span, n - i) + 1):
                for b in range(1, min(max_span, m - j) + 1):
                    left, right = gold[i:i + a], actual[j:j + b]
                    if (any(not any(nearby(x, y, tolerance) for y in right) for x in left) or
                            any(not any(nearby(x, y, tolerance) for x in left) for y in right)):
                        continue
                    lt = ''.join(normalize_text(row['文本']) for row in left)
                    rt = ''.join(normalize_text(row['文本']) for row in right)
                    score = similarity(lt, rt)
                    if score < threshold or ((a > 1 or b > 1) and lt != rt):
                        continue
                    reward = score * (len(lt) + len(rt)) - .1 * (a + b - 2)
                    update(i + a, j + b, dp[i][j] + reward, (i, j))
    groups = []
    i, j = n, m
    while i or j:
        pi, pj = back[i][j]
        groups.append((gold[pi:i], actual[pj:j]))
        i, j = pi, pj
    return list(reversed(groups))


def pair_actions(gold, actual, tolerance, threshold, same_actor=False):
    choices = []
    for i, left in enumerate(gold):
        for j, right in enumerate(actual):
            if not nearby(left, right, tolerance):
                continue
            if same_actor and left['执行者'] != right['执行者']:
                continue
            score = action_similarity(left['动作'], right['动作'])
            if score >= threshold:
                distance = abs(to_sec(left['起']) - to_sec(right['起']))
                choices.append((-score, distance, i, j))
    used_left, used_right, pairs = set(), set(), []
    for _, _, i, j in sorted(choices):
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            pairs.append((i, j))
    return pairs


def metric(numerator, denominator, errors, **extra):
    return {'分子': numerator, '分母': denominator, '比率': numerator / denominator if denominator else None,
            '错误': errors, **extra}


def evidence(gold=None, actual=None, **extra):
    position = {'标注': [{k: row[k] for k in ('id', '起', '止') if k in row} for row in gold or []],
                '正文': [{k: row[k] for k in ('行', '场景', '句', 'id', '起', '止') if k in row} for row in actual or []]}
    return {'位置': position, '标注': gold, '正文': actual, **extra}


def evaluate(annotation: str | Path, episode: str | Path, *, script='正式稿.md', mapping=None,
             time_tolerance=.3, text_threshold=.55, action_threshold=.55, max_span=3) -> dict:
    if (not math.isfinite(time_tolerance) or time_tolerance < 0 or
            not 0 < text_threshold <= 1 or not 0 < action_threshold <= 1 or max_span < 1):
        raise ValueError('时间容差须为有限非负数，相似度须在 (0, 1]，最大断句合并数须大于 0')
    annotation, episode = Path(annotation), Path(episode)
    gold = read_json(annotation)
    validate_annotation(gold, complete=True)
    mapping_path = Path(mapping) if mapping is not None else annotation.with_name('人物映射.json')
    role_map = read_json(mapping_path)
    roles = read_json(episode.parent / '角色卡.json')
    role_ids = {role['id'] for role in roles}
    if not isinstance(role_map, dict) or any(role_map.get(row['id']) not in role_ids for row in gold['人物']):
        raise ValueError('人物映射须为 {"H01": "C001"}，并覆盖全部标注人物，目标须存在于角色卡')
    names = defaultdict(set)
    for role in roles:
        for name in [role['名称'], *role.get('别名称呼', [])]:
            if name:
                names[name].add(role['id'])
    name_pattern = re.compile('|'.join(re.escape(name) for name in sorted(names, key=len, reverse=True))) if names else None

    def entity(value):
        return role_map.get(value, value)

    def role_id(name):
        ids = names.get(name, set())
        return next(iter(ids)) if len(ids) == 1 else f'未识别：{name}'

    sounds = read_json(episode / '声音记录.json')
    actions = read_json(episode / '动作记录.json')
    attribution = read_json(episode / '归属表.json')
    references = read_json(episode / '对照表.json')
    sound_by_id = {row['id']: row for row in sounds}
    action_by_id = {row['id']: row for row in actions}
    attribution_by_id = {row['句']: row for row in attribution}
    if len(sound_by_id) != len(sounds) or len(action_by_id) != len(actions) or len(attribution_by_id) != len(attribution):
        raise ValueError('黑板存在重复声音、动作或归属编号')
    script = Path(script)
    script = script if script.is_absolute() else episode / script
    lines = script.read_text(encoding='utf-8').splitlines()
    by_line = {}
    for ref in references:
        number = ref['行']
        if number in by_line or not isinstance(number, int) or not 1 <= number <= len(lines):
            raise ValueError(f'对照表行号重复或越界：{number}')
        if any(x not in sound_by_id for x in ref['句']) or any(x not in action_by_id for x in ref['动作']):
            raise ValueError(f'对照表第 {number} 行引用未知声音或动作')
        by_line[number] = ref
    warnings, actual_dialogue, body_actions = [], [], []
    words = read_json(DEFAULT_WORDS)['动作词']
    seen_sentences, seen_actions = set(), set()
    scene = None
    for number, raw in enumerate(lines, 1):
        text = raw.strip()
        if text.startswith('场景'):
            scene = text.split('：', 1)[0]
        ref = by_line.get(number, {'句': [], '动作': []})
        dialogue = parse_dialogue(text)
        description = ''
        speaker = None
        if dialogue:
            name, bracket, spoken = dialogue
            speaker = role_id(name)
            ids = ref['句']
            if seen_sentences.intersection(ids) or len(ids) != len(set(ids)):
                raise ValueError(f'对白句重复映射：第 {number} 行')
            seen_sentences.update(ids)
            source = [sound_by_id[x] for x in ids]
            row = {'行': number, '场景': scene, '句': ids, '文本': spoken, '说话人': speaker,
                   '起': min((x['起'] for x in source), key=to_sec, default=None),
                   '止': max((x['止'] for x in source), key=to_sec, default=None),
                   '归属记录': [attribution_by_id.get(x) for x in ids]}
            actual_dialogue.append(row)
            if not source:
                warnings.append({'类型': '对白缺时间映射', '行': number, '正文': raw})
            if any(x not in attribution_by_id for x in ids):
                warnings.append({'类型': '对白缺归属记录', '行': number, '句': ids})
            if ref['动作'] or any(word in bracket for word in words):
                description = bracket
        elif text.startswith('△') and not text[1:].strip().startswith('画面字幕：'):
            description = text[1:].strip()
        if ref['句'] and not dialogue:
            raise ValueError(f'对照表第 {number} 行引用对白，但正文不是对白')
        if ref['动作'] and not description:
            warnings.append({'类型': '动作映射没有正文动作', '行': number, '动作': ref['动作']})
        if not description:
            continue
        ids = ref['动作'] or [None]
        if any(x in seen_actions for x in ids if x is not None) or len(ids) != len(set(ids)):
            raise ValueError(f'动作重复映射：第 {number} 行')
        seen_actions.update(x for x in ids if x is not None)
        for ident in ids:
            source = action_by_id.get(ident, {})
            action_text = description
            if len(ids) > 1:
                clauses = [clause.strip() for clause in re.split(r'[。；;]', description) if clause.strip()]
                action_text = max(clauses, key=lambda clause: action_similarity(source.get('动作'), clause), default=description)
            actor = speaker or source.get('执行者')
            # 正文明确写出的人名优先于动作记录，防止正文换人后被原记录遮住。
            mentioned = [(match.start(), role_id(match[0])) for match in name_pattern.finditer(action_text)] if name_pattern else []
            subject_written = bool(mentioned and mentioned[0][0] == 0)
            if subject_written:
                actor = mentioned[0][1]
            targets = mentioned[1:] if subject_written else mentioned
            target = targets[0][1] if targets else source.get('对象')
            body_actions.append({'id': ident or f'正文行{number}', '行': number, '场景': scene,
                                 '起': source.get('起'), '止': source.get('止'), '动作': action_text,
                                 '记录动作': source.get('动作'), '执行者': actor, '对象': target,
                                 '原文': raw})
            if not ident:
                warnings.append({'类型': '动作缺时间映射', '行': number, '正文': raw})

    gold_dialogue = sorted(gold['台词'], key=lambda row: (to_sec(row['起']), to_sec(row['止'])))
    groups = align_dialogue(gold_dialogue, actual_dialogue, time_tolerance, text_threshold, max_span)
    correct, dialogue_errors, speaker_errors = 0, [], []
    for left, right in groups:
        if left:
            exact = (right and normalize_text(''.join(x['文本'] for x in left)) ==
                     normalize_text(''.join(x['文本'] for x in right)))
            if exact:
                correct += len(left)
            else:
                for row in left:
                    dialogue_errors.append(evidence([row], right, 类型='漏句' if not right else '台词不一致'))
        if not left and right:
            warnings.append(evidence([], right, 类型='正文台词无匹配标注'))
        for row in right:
            if not left:
                continue
            units = [{**row, '句': [ident], '起': sound_by_id[ident]['起'], '止': sound_by_id[ident]['止']}
                     for ident in row['句']] or [row]
            for unit in units:
                candidates = [x for x in left if nearby(x, unit, time_tolerance)]
                expected = {entity(x['说话人']) for x in candidates}
                if len(expected) != 1:
                    warnings.append(evidence(candidates, [unit], 类型='合并对白说话人需人工复核'))
                elif unit['说话人'] not in expected:
                    speaker_errors.append(evidence(candidates, [unit], 类型='说话人错归', 期望说话人=sorted(expected)))

    gold_actions = [{**row, '执行者': entity(row['执行者']), '对象': entity(row['对象'])} for row in gold['关键动作']]
    body_pairs = pair_actions(gold_actions, body_actions, time_tolerance, action_threshold)
    matched_gold = {i for i, _ in pair_actions(gold_actions, body_actions, time_tolerance, action_threshold, True)}
    matched_gold.update(i for i, _ in pair_actions(gold_actions, actions, time_tolerance, action_threshold, True))
    missed = [evidence([row], [], 类型='关键动作漏写') for i, row in enumerate(gold_actions) if i not in matched_gold]
    supported = {j for _, j in body_pairs}
    unsupported = [evidence([], [row], 类型='无据动作候选', 需要人工复核=True)
                   for j, row in enumerate(body_actions) if j not in supported]
    mismatches = [evidence([gold_actions[i]], [body_actions[j]], 类型='人物错配')
                  for i, j in body_pairs if any(gold_actions[i][k] != body_actions[j][k] for k in ('执行者', '对象'))]
    metrics = {
        '台词逐字一致率': metric(correct, len(gold_dialogue), dialogue_errors),
        '说话人错归率': metric(len(speaker_errors), sum(max(1, len(row['句'])) for row in actual_dialogue), speaker_errors),
        '关键动作漏写率': metric(len(missed), len(gold_actions), missed),
        '无据动作率': metric(len(unsupported), len(body_actions), unsupported, 需要人工复核=True),
        '人物错配率': metric(len(mismatches), len(body_pairs), mismatches),
    }
    return {'集': gold['集'], '标注': str(annotation), '稿': str(script), '人物映射': str(mapping_path),
            '参数': {'时间容差秒': time_tolerance, '台词相似度阈值': text_threshold,
                     '动作相似度阈值': action_threshold, '最大断句合并数': max_span},
            '指标': metrics, '待复核': warnings,
            '口径': {'说话人分母': '正文引用的对白句数，未映射对白行按一句计；缺句只计入台词漏句，无匹配标注的正文对白列入待复核',
                     '关键动作': '正文或动作记录任一覆盖即可；须在时间容差内、执行者一致、动作文本达到阈值',
                     '人物错配分母': '与标注按时间和动作文本匹配的正文动作数',
                     '动作单位': '正文对照表的动作引用；未引用动作记录的 △ 或含动作词的对白括号作为一个候选',
                     '无据动作': '标注只含关键动作，非关键动作也可能成为候选；候选率不等于素材事实判定',
                     '空分母': '比率为 null，不记为满分'}}


def aggregate(results):
    output = {}
    for key in METRICS:
        numerator = sum(row['指标'][key]['分子'] for row in results)
        denominator = sum(row['指标'][key]['分母'] for row in results)
        errors = [{**error, '集': row['集']} for row in results for error in row['指标'][key]['错误']]
        output[key] = metric(numerator, denominator, errors, **({'需要人工复核': True} if key == '无据动作率' else {}))
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotation', type=Path)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--稿', default='正式稿.md')
    parser.add_argument('--人物映射', type=Path)
    parser.add_argument('--time-tolerance', type=float, default=.3)
    parser.add_argument('--text-threshold', type=float, default=.55)
    parser.add_argument('--action-threshold', type=float, default=.55)
    parser.add_argument('--max-span', type=int, default=3)
    parser.add_argument('--追加', nargs=2, action='append', metavar=('标注', '本集目录'), default=[])
    args = parser.parse_args(argv)
    try:
        pairs = [(args.annotation, args.episode), *args.追加]
        results = [evaluate(annotation, episode, script=args.稿, mapping=args.人物映射,
                            time_tolerance=args.time_tolerance, text_threshold=args.text_threshold,
                            action_threshold=args.action_threshold, max_span=args.max_span)
                   for annotation, episode in pairs]
        if len({row['集'] for row in results}) != len(results):
            raise ValueError('多集评测不可重复集号')
        print(json.dumps({'各集': results, '合计': aggregate(results)}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'错误': str(exc)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
