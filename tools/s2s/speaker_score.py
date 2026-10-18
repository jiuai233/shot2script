"""已有人工标注的说话人/声音形态时间区间子集评分；不作正式整集验收。"""

import argparse
import json
from pathlib import Path
import re
import sys

from jsonschema import Draft202012Validator

from .evaluate import read_json, validate_annotation
from .timefmt import fmt, to_sec
from .validate import SCHEMA_DIR


_ROLE = re.compile(r'(?:C[0-9]{3,}|H[0-9]{2,}|E[0-9]{2,}-(?:N[0-9]{3,}|P[0-9]{2,}-T[1-9][0-9]*))')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ticks(value):
    # 现有输入 Schema 精确到 0.1 秒；用整数边界避免相加/比较的浮点漂移。
    return round(to_sec(value) * 10)


def _predictions(sounds, result, annotation):
    errors = list(Draft202012Validator(read_json(SCHEMA_DIR / '声音记录.schema.json')).iter_errors(sounds))
    _require(not errors, f'声音记录 Schema：{errors[0].message}' if errors else '')
    by_sound = {}
    episode = f'E{annotation["集"]:02d}'
    for row in sounds:
        ident = row['id']
        _require(ident not in by_sound, f'声音句号重复：{ident}')
        _require(ident.split('-')[0] == episode, f'声音句号非本集：{ident}')
        _require(_ticks(row['起']) < _ticks(row['止']), f'声音句非正时长：{ident}')
        by_sound[ident] = row
    ordered = sorted(sounds, key=lambda r: _ticks(r['起']))
    for left, right in zip(ordered, ordered[1:]):
        _require(_ticks(left['止']) <= _ticks(right['起']), f'声音句时间重叠：{left["id"]} / {right["id"]}')
    _require(isinstance(result, dict) and isinstance(result.get('对白归属'), list), '结果须含对白归属数组')
    people = {r['id'] for r in annotation['人物']}
    two_stage = result.get('结果版本') == 2
    registered = set()
    if two_stage:
        table = result.get('本集人物')
        _require(isinstance(table, dict) and isinstance(table.get('人物'), list), '两阶段结果缺本集人物表')
        for row in table['人物']:
            _require(isinstance(row, dict) and isinstance(row.get('id'), str), '本集人物编号无效')
            ident = row['id']
            _require(re.fullmatch(re.escape(episode) + r'-N[0-9]{3,}', ident), f'本集人物编号非本集N：{ident}')
            _require(ident not in registered, f'本集人物编号重复：{ident}')
            registered.add(ident)

    def check_role(ident):
        _require(isinstance(ident, str) and _ROLE.fullmatch(ident), f'非法人物引用：{ident}')
        if ident.startswith('H'):
            _require(ident in people, f'人物引用未登记：{ident}')
        elif ident.startswith('E'):
            _require(ident.split('-')[0] == episode, f'人物引用非本集：{ident}')
            if '-N' in ident:
                _require(two_stage and ident in registered, f'N人物引用未登记：{ident}')

    aliases = {}
    raw_aliases = result.get('临时人物映射', [])
    _require(isinstance(raw_aliases, list), '临时人物映射须为数组')
    for row in raw_aliases:
        _require(isinstance(row, dict) and '临时' in row and '映射' in row, '临时人物映射字段缺失')
        source, target = row['临时'], row['映射']
        check_role(source)
        check_role(target)
        _require(source not in aliases, f'临时人物映射重复：{source}')
        _require(row.get('确定度') in ('确认', '存疑'), '临时人物映射确定度缺失或非法')
        aliases[source] = (target, row['确定度'])

    def resolve(ident):
        check_role(ident)
        seen, certain = set(), True
        while ident in aliases:
            _require(ident not in seen, f'临时人物映射循环：{ident}')
            seen.add(ident)
            target, confidence = aliases[ident]
            certain = certain and confidence == '确认'
            if target == ident:
                break
            ident = target
        _require(not (two_stage and '-T' in ident), f'两阶段终端T人物未登记到N/C：{ident}')
        return ident, certain

    # 即使某别名未被预测引用，结构错误也不能变成排除难例。
    for ident in aliases:
        resolve(ident)
    blocked = set()
    pending = result.get('待核对', [])
    _require(isinstance(pending, list), '待核对须为数组')
    for row in pending:
        _require(isinstance(row, dict) and isinstance(row.get('位置', {}), dict), '待核对位置无效')
        sentence = row.get('位置', {}).get('句')
        if sentence is not None:
            _require(sentence in by_sound, f'待核对引用未知句：{sentence}')
            if row.get('级别') == '阻断':
                blocked.add(sentence)
    attribution = {}
    for row in result['对白归属']:
        _require(isinstance(row, dict) and all(k in row for k in ('句', '说话人', '形态', '确定度')), '对白归属字段缺失')
        sentence = row['句']
        _require(isinstance(sentence, str) and sentence in by_sound, f'归属引用未知句：{sentence}')
        _require(sentence not in attribution, f'对白归属重复句号：{sentence}')
        _require(row['确定度'] in ('确认', '存疑'), f'归属确定度非法：{sentence}')
        _require(row['形态'] in (None, '画内', '画外', '心声', '设备'), f'归属形态非法：{sentence}')
        terminal, alias_certain = resolve(row['说话人']) if row['说话人'] is not None else (None, False)
        attribution[sentence] = {**row, '终端': terminal, '别名确认': alias_certain,
                                 '句确认': row['确定度'] == '确认' and sentence not in blocked}
    return ordered, attribution, resolve, check_role, two_stage


def _mapping(mapping, resolve, check_role, people):
    _require(mapping is None or isinstance(mapping, dict), '人物映射须为对象')
    normalized = {}
    for left, right in (mapping or {}).items():
        _require(isinstance(left, str) and isinstance(right, str), '人物映射键值须为编号字符串')
        if left.startswith('H') and not right.startswith('H'):
            prediction, human = right, left
        else:
            prediction, human = left, right
        _require(human in people, f'人物映射目标未登记：{human}')
        try:
            check_role(prediction)
            terminal, alias_certain = resolve(prediction)
        except ValueError as exc:
            raise ValueError(f'人物映射：{exc}') from exc
        _require(terminal == prediction or alias_certain,
                 f'人物映射不能经存疑别名链转移：{prediction}；请直接提供终端编号 {terminal} 的独立映射')
        _require(terminal not in normalized or normalized[terminal] == human, f'人物映射终端冲突：{terminal}')
        normalized[terminal] = human
    return normalized


def _score_field(annotation, sounds, attribution, mapping, mapping_provided, field):
    eligible, excluded = [], []
    for row in annotation['台词']:
        reasons = []
        if _ticks(row['起']) == _ticks(row['止']):
            reasons.append('零时长')
        if not row[field]:
            reasons.append('空' + field)
        if row.get('待核对'):
            reasons.append('待核对')
        if reasons:
            excluded.append({'来源': row['id'], '原因': reasons})
        else:
            eligible.append(row)
    boundaries = sorted({_ticks(r[k]) for r in eligible + sounds for k in ('起', '止')})
    details, conflicts, uncovered = [], [], []
    used, missing = set(), set()
    totals = dict.fromkeys(('正确', '错误', '未确认', '无法映射'), 0)
    for start, end in zip(boundaries, boundaries[1:]):
        active = [r for r in eligible if _ticks(r['起']) <= start and end <= _ticks(r['止'])]
        if not active:
            continue
        values = sorted({r[field] for r in active})
        span = {'起': fmt(start / 10), '止': fmt(end / 10), '秒数': (end - start) / 10,
                '来源': [r['id'] for r in active]}
        if len(values) != 1:
            conflicts.append({**span, '真值': values})
            continue
        used.update(r['id'] for r in active)
        prediction = next((r for r in sounds if _ticks(r['起']) <= start and end <= _ticks(r['止'])), None)
        attr = attribution.get(prediction['id']) if prediction else None
        status, reason, predicted, terminal = '未确认', '', None, None
        if prediction is None:
            reason = '缺声音句'
            uncovered.append(span)
        elif attr is None:
            reason = '缺归属'
        else:
            predicted = attr[field]
            terminal = attr['终端'] if field == '说话人' else None
            if not attr['句确认']:
                reason = '归属存疑' if attr['确定度'] == '存疑' else '同句阻断待核对'
            elif predicted is None:
                reason = '空预测' + field
            elif field == '说话人' and not attr['别名确认']:
                reason = '别名映射存疑'
            elif field == '说话人' and terminal not in mapping:
                status, reason = '无法映射', '缺独立人物映射（非系统存疑）'
                missing.add(terminal)
            else:
                actual = mapping[terminal] if field == '说话人' else predicted
                status = '正确' if actual == values[0] else '错误'
        totals[status] += end - start
        detail = {**span, '真值': values[0], '句': prediction['id'] if prediction else None,
                  '预测': predicted, '状态': status, '原因': reason}
        if field == '说话人':
            detail.update({'预测终端': terminal, '独立映射人物': mapping.get(terminal)})
        details.append(detail)
    denominator = sum(totals.values())
    confirmed = denominator - totals['未确认']
    comparable = field != '说话人' or (mapping_provided and not missing)

    def ratio(numerator, divisor, needs_mapping=False):
        return numerator / divisor if divisor and (comparable or not needs_mapping) else None

    return {'有效标注条数': len(used), '候选标注条数': len(eligible), '分母秒数': denominator / 10,
            **{k + '秒数': v / 10 for k, v in totals.items()}, '确认秒数': confirmed / 10,
            '确认覆盖率': ratio(confirmed, denominator),
            '确认精度': ratio(totals['正确'], confirmed, True),
            '有效正确覆盖率': ratio(totals['正确'], denominator, True),
            '错归率': ratio(totals['错误'], confirmed, True),
            '整体正确性可计算': comparable,
            '排除条目': excluded, '冲突秒数': round(sum(r['秒数'] for r in conflicts), 1),
            '冲突区间': conflicts, '未覆盖秒数': round(sum(r['秒数'] for r in uncovered), 1),
            '未覆盖区间': uncovered, '逐区间': details, '缺失映射编号': sorted(missing)}


def score_speaker(annotation: dict, sounds: list[dict], result: dict,
                  mapping: dict[str, str] | None = None) -> dict:
    """只读评分；mapping 必须独立提供，不从本次预测/真值共现反推身份。"""
    validate_annotation(annotation, complete=False)
    sounds, attribution, resolve, check_role, two_stage = _predictions(sounds, result, annotation)
    normalized = _mapping(mapping, resolve, check_role, {r['id'] for r in annotation['人物']})
    metrics = {field: _score_field(annotation, sounds, attribution, normalized, bool(mapping), field)
               for field in ('说话人', '形态')}
    return {'评测范围': '已有标注子集（非正式整集验收）', '源校对完成': annotation.get('校对完成', False),
            '口径': {
                '单位': '有效人工时间区间；边界交集、左闭右开、按秒加权；同值重叠去重，异值冲突区间排除',
                '范围': '说话人与形态独立使用非空、正时长且台词条目无待核对字段的标注；条目待核对同时排除两项',
                '确认': '归属确定度确认且无同句阻断；说话人另需非空及别名链确认；形态不依赖身份或人物映射',
                '比率': '确认覆盖率=确认/有效标注时长；确认精度=正确/确认；有效正确覆盖率=正确/有效标注时长；错归率=错误确认秒数/已确认秒数（形态项表示已确认形态错误率）；空分母为null',
                '映射缺口': '正确/错误秒数只计可映射已知部分；缺独立映射或已确认预测存在缺映射时，身份整体依赖映射比率为null；无法映射不等于系统存疑',
                '旧登记边界': '旧结果合法同集T及C只作终端编号，登记未验证；不据名称或共现猜身份。两阶段N须在本集人物表登记',
            }, '输入声音句数': len(sounds), '源台词条数': len(annotation['台词']),
            '结果类型': '两阶段' if two_stage else '旧单次',
            '人物映射': {'提供内容': mapping, '归一化预测到人工': normalized,
                         '缺失编号': metrics['说话人']['缺失映射编号'],
                         '状态': '未提供或为空' if not mapping else ('部分缺失' if metrics['说话人']['缺失映射编号'] else '覆盖本次已确认可评分预测')},
            '指标': metrics}


def _unique_object(pairs):
    data = {}
    for key, value in pairs:
        _require(key not in data, f'JSON字段重复：{key}')
        data[key] = value
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='已有人工标注子集的说话人/声音形态时间加权评分；不调用模型')
    parser.add_argument('声音记录')
    parser.add_argument('归属结果')
    parser.add_argument('标注')
    parser.add_argument('输出', help='必须为新文件，不覆盖已有文件')
    parser.add_argument('--人物映射', help='独立确认的H→C或预测C/N/T/H→H映射；缺失不推断身份')
    args = parser.parse_args(argv)
    try:
        output = Path(args.输出)
        _require(not output.exists(), f'输出已存在：{output}')
        sources, loaded = {}, {}
        for name in ('声音记录', '归属结果', '标注', '人物映射'):
            path = getattr(args, name)
            if path is None:
                loaded[name] = None
                continue
            path = Path(path).resolve()
            raw = path.read_bytes()
            sources[name] = {'路径': str(path)}
            loaded[name] = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object)
        report = score_speaker(loaded['标注'], loaded['声音记录'], loaded['归属结果'], loaded['人物映射'])
        report['来源'] = sources
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as file:
            file.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        summary = {'评测范围': report['评测范围'], '输出': str(output.resolve()),
                   '人物映射': report['人物映射'],
                   '指标': {field: {k: v for k, v in metric.items() if k not in
                                   ('排除条目', '冲突区间', '未覆盖区间', '逐区间')}
                            for field, metric in report['指标'].items()}}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
