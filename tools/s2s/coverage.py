"""核对句号、关键动作覆盖与正文对白原文。"""

import argparse
from collections import Counter
import json
from pathlib import Path
import unicodedata

from s2s.lint_script import line_type, parse_dialogue, PENDING_NOTE_RE
from s2s.validate import read_json, time_seconds


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    return ''.join(ch for ch in text if not ch.isspace() and not unicodedata.category(ch).startswith('P'))


def check_coverage(directory: str | Path, script: str | Path = '工作稿.md') -> dict:
    directory = Path(directory)
    script = Path(script)
    script = script if script.is_absolute() else directory / script
    errors = []
    result = {'目录': str(directory), '稿': str(script), '通过': False, '错误': errors}

    def fail(kind, location, message):
        errors.append({'类型': kind, '位置': location, '问题': message})

    def load(name):
        try:
            data = read_json(directory / name)
            if not isinstance(data, list) or any(not isinstance(r, dict) for r in data):
                raise ValueError('须为对象数组')
            return data
        except (OSError, ValueError) as exc:
            fail('输入文件', {'文件': name}, str(exc))
            return None

    sound = load('声音记录.json')
    actions = load('动作记录.json')
    mapping = load('对照表.json')
    try:
        lines = script.read_text(encoding='utf-8').splitlines()
    except (OSError, ValueError) as exc:
        fail('输入文件', {'文件': str(script)}, str(exc))
        return result
    if any(data is None for data in (sound, actions, mapping)):
        return result
    try:
        for entry in sound:
            if not isinstance(entry.get('id'), str) or not isinstance(entry.get('文本'), str):
                raise ValueError('声音记录缺少字符串 id 或文本')
            time_seconds(entry['起'])
            time_seconds(entry['止'])
        for entry in actions:
            if not isinstance(entry.get('id'), str) or not isinstance(entry.get('关键'), bool):
                raise ValueError('动作记录缺少 id 或布尔字段关键')
        for entry in mapping:
            if not isinstance(entry.get('行'), int) or isinstance(entry['行'], bool) or entry['行'] < 1:
                raise ValueError('对照表行号须为正整数')
            if entry.get('类型') not in ('场景行', '人物行', '△', '对白'):
                raise ValueError('对照表类型无效')
            for key in ('句', '动作'):
                if not isinstance(entry.get(key), list) or any(not isinstance(v, str) for v in entry[key]):
                    raise ValueError(f'对照表 {key} 须为字符串数组')
    except (KeyError, ValueError) as exc:
        fail('输入文件', {'文件': '声音记录/动作记录/对照表'}, str(exc))
        return result

    for name, entries in (('声音记录.json', sound), ('动作记录.json', actions)):
        for ident, count in Counter(entry['id'] for entry in entries).items():
            if count > 1:
                fail('编号重复', {'文件': name, '编号': ident}, f'编号出现 {count} 次')
    sound_by_id = {entry['id']: entry for entry in sound}
    action_by_id = {entry['id']: entry for entry in actions}
    mapping = sorted(mapping, key=lambda entry: entry['行'])
    counts = Counter(ident for entry in mapping for ident in entry['句'])
    for ident in sound_by_id:
        count = counts[ident]
        if count == 0:
            fail('漏句', {'文件': '声音记录.json', '句': ident}, '句号未在对照表中出现')
        elif count > 1:
            positions = [entry['行'] for entry in mapping if ident in entry['句']]
            fail('重复句', {'句': ident, '行': positions}, f'句号出现 {count} 次')
    for entry in mapping:
        for ident in entry['句']:
            if ident not in sound_by_id:
                fail('未知句', {'行': entry['行'], '句': ident}, '句号不在声音记录中')
        for ident in entry['动作']:
            if ident not in action_by_id:
                fail('未知动作', {'行': entry['行'], '动作': ident}, '动作编号不在动作记录中')
    chronological = sorted(sound, key=lambda entry: (time_seconds(entry['起']), time_seconds(entry['止'])))
    rank = {entry['id']: i for i, entry in enumerate(chronological)}
    previous = None
    for entry in mapping:
        for ident in entry['句']:
            if ident not in rank:
                continue
            if previous is not None and rank[ident] < rank[previous]:
                fail('句序错乱', {'行': entry['行'], '句': ident}, f'{ident} 排在时间更晚的 {previous} 后面')
            previous = ident
    action_counts = Counter(ident for entry in mapping for ident in entry['动作'])
    for ident, count in action_counts.items():
        if count > 1:
            fail('重复动作', {'动作': ident, '行': [r['行'] for r in mapping if ident in r['动作']]},
                 f'同一动作引用 {count} 次')
    covered_actions = set(action_counts)
    for entry in actions:
        if entry['关键'] and entry.get('确定度') != '存疑' and entry['id'] not in covered_actions:
            fail('关键动作未覆盖', {'文件': '动作记录.json', '动作': entry['id']}, '关键动作未在对照表中出现')

    mapped_lines = Counter(entry['行'] for entry in mapping)
    for number, count in mapped_lines.items():
        if count > 1:
            fail('重复行映射', {'行': number}, f'同一正文行有 {count} 条对照记录')
    dialogue_lines = {i: parse_dialogue(line) for i, line in enumerate(lines, 1) if parse_dialogue(line)}
    mapped_dialogues = set()
    for entry in mapping:
        number = entry['行']
        if number > len(lines):
            fail('行号越界', {'行': number}, f'正文只有 {len(lines)} 行')
            continue
        actual_type = line_type(lines[number - 1])
        if entry['类型'] != actual_type:
            fail('正文行类型', {'行': number}, f'对照表为 {entry["类型"]}，正文为 {actual_type}')
        if entry['动作'] and entry['类型'] not in ('△', '对白'):
            fail('非正文动作映射', {'行': number}, '动作只能映射到 △ 或对白括号')
        if entry['类型'] != '对白':
            if entry['句']:
                fail('非对白句映射', {'行': number}, '句号只能映射到对白行')
            continue
        mapped_dialogues.add(number)
        if number not in dialogue_lines:
            fail('对白行类型', {'行': number}, '对照表指向的正文行不是对白')
            continue
        if any(ident not in sound_by_id for ident in entry['句']):
            continue
        expected = ''.join(sound_by_id[ident]['文本'] for ident in entry['句'])
        actual = PENDING_NOTE_RE.sub('', dialogue_lines[number][2])
        if normalize_text(actual) != normalize_text(expected):
            fail('台词不一致', {'行': number, '句': entry['句']},
                 f'正文：{actual}；声音记录：{expected}')
    for number in dialogue_lines.keys() - mapped_dialogues:
        fail('对白未映射', {'行': number}, '正文对白行未在对照表中出现')
    scene = None
    scenes_by_line = {}
    for number, line in enumerate(lines, 1):
        kind = line_type(line)
        if kind == '场景行':
            scene = line.split('：', 1)[0].strip()
        scenes_by_line[number] = scene
        if kind in ('场景行', '人物行', '△', '对白') and number not in mapped_lines:
            fail('正文未映射', {'行': number}, '正文行缺少对照记录，不能跳过独立画面核对')
    for entry in mapping:
        if entry['行'] in scenes_by_line and entry.get('场景') != scenes_by_line[entry['行']]:
            fail('行场景不一致', {'行': entry['行']}, '对照表场景与该行实际场景不一致')
    result['通过'] = not errors
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='检查对白、关键动作覆盖和逐字一致')
    parser.add_argument('本集目录')
    parser.add_argument('--稿', default='工作稿.md')
    args = parser.parse_args()
    result = check_coverage(args.本集目录, args.稿)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['通过'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
