"""提取角色返回，并用返回包 Schema 与黑板校验器检查结构及引用。"""

import json
from pathlib import Path
import re
import tempfile

from jsonschema import Draft202012Validator
from s2s.validate import validate_file, time_seconds
from s2s.names import name_errors
from s2s.quality import check_script
from .materials import dumps, load

# 实测正常片段中有人物而无动作的镜头最多约一半，整段漏看的片段超过八成。
IDLE_MIN_SHOTS = 8
IDLE_MAX_SHARE = 0.7


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    blocks = re.findall(r'```(?:json)?\s*\n?([\s\S]*?)```', text)
    candidates = []
    for block in blocks:
        try:
            candidates.append(json.loads(block))
        except ValueError:
            continue
    if candidates:
        if len(candidates) != 1:
            raise ValueError('返回含多个 JSON，无法确定结果')
        return candidates[0]
    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        if text[cursor] not in '[{':
            cursor += 1
            continue
        try:
            value, length = decoder.raw_decode(text[cursor:])
            candidates.append(value)
            cursor += length
        except ValueError:
            cursor += 1
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError('返回中没有唯一、可解析的 JSON')


def extract_result(role: str, text: str):
    if role != '成稿':
        return extract_json(text)
    try:
        value = extract_json(text)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    # 兼容提示词原本要求的“工作稿 + 对照表”双块输出。
    marker = re.search(r'(?:^|\n)[#*\s]*(?:2[.、]\s*)?对照表[^\n]*\n', text)
    if not marker:
        raise ValueError('成稿缺少工作稿与对照表两部分')
    before = text[:marker.start()]
    match = re.search(r'第[0-9]+集\s*\n', before)
    if not match:
        raise ValueError('工作稿缺少集标题')
    script = before[match.start():].strip().rstrip('`').strip()
    table = extract_json(text[marker.end():])
    if not isinstance(table, list):
        raise ValueError('对照表必须是数组')
    return {'工作稿': script, '对照表': table}


def validate_result(role: str, data, episode: Path, package) -> list[str]:
    if role == '角色卡' and package.mode in ('建立', '人物合并'):
        return ['建立必须通过dispatch两阶段建立入口，不接受旧单次返回']
    schemas = load(Path(__file__).with_name('role_schemas.json'))
    errors = [f'{list(e.absolute_path)}: {e.message}'
              for e in Draft202012Validator(schemas[role]).iter_errors(data)]
    if errors:
        return errors
    if data.get('模式', package.mode) != package.mode:
        errors.append('返回模式与派发模式不一致')
    staging = episode / '派发结果' / '.validation'
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='validate-', dir=staging) as folder:
        context = Path(folder) / episode.name
        context.mkdir()
        for name in ('镜头表.json', '片段表.json', '声音记录.json', '字幕记录.json',
                     '动作记录.json', '归属表.json', '待核对.json', '问题清单.json', '工作稿.md', '临时人物.json', '本集人物.json'):
            source = episode / name
            if source.is_file():
                (context / name).symlink_to(source.resolve())
        if (episode / '素材/frames_dense').is_dir():
            (context / '素材').mkdir()
            (context / '素材/frames_dense').symlink_to((episode / '素材/frames_dense').resolve())
        roles = episode.parent / '角色卡.json'
        (context / '角色卡.json').write_text(roles.read_text(encoding='utf-8') if roles.is_file() else '[]', encoding='utf-8')

        def check(kind, rows):
            target = context / (kind + '.json')
            if target.is_symlink():
                target.unlink()
            target.write_text(dumps(rows), encoding='utf-8')
            result = validate_file(target, kind)
            errors.extend(f'{kind}{item["位置"]}: {item["问题"]}' for item in result['错误'])

        if role == '动作识别':
            segment = package.segment
            if data['片段'] != segment['id']:
                errors.append('片段号不一致')
            if [s['镜头'] for s in data['镜头']] != segment['镜头']:
                errors.append('输出镜头必须按顺序覆盖本段一次，不含衔接镜头')
            shot_map = {s['id']: s for s in load(episode / '镜头表.json')['镜头']}
            sent = {s['id']: s for s in package.shots}
            known = {r['id'] for r in load(context / '角色卡.json')}
            declared = {p['人物'] for s in data['镜头'] for p in s['画面人物']}
            for person in declared - known:
                if not person.startswith(segment['id'] + '-T'):
                    errors.append('未登记人物：' + person)
                elif not any(p.get('外观') for s in data['镜头'] for p in s['画面人物'] if p['人物'] == person):
                    errors.append('临时人物缺少外观：' + person)
            rows = []
            for shot in data['镜头']:
                source = shot_map.get(shot['镜头'])
                if source is None:
                    continue
                if (shot['起'], shot['止']) != (source['起'], source['止']):
                    errors.append(shot['镜头'] + ' 起止与镜头表不一致')
                authorized = sent.get(shot['镜头'])
                if authorized is None:
                    errors.append(shot['镜头'] + ' 缺少本次授权镜头')
                frames = {f['文件'] for f in authorized['帧']} if authorized else set()
                if authorized is not None:
                    frames.update(f['文件'] for f in package.dense if f['镜头'] == shot['镜头'])
                for entry in [*shot['画面人物'], *shot['动作'], *shot['神态']]:
                    if not set(entry['依据帧']).issubset(frames):
                        errors.append(shot['镜头'] + ' 引用了非本镜头依据帧')
                local_people = {p['人物'] for p in shot['画面人物']}
                for action in shot['动作']:
                    if action['执行者'] not in local_people:
                        errors.append('动作执行者不在本镜头人物中')
                    if not (time_seconds(source['起']) <= time_seconds(action['起']) <= time_seconds(action['止']) <= time_seconds(source['止'])):
                        errors.append('动作时间越出镜头范围')
                    rows.append({'id': f'{episode.name}-A{len(rows)+1:04d}', '镜头': shot['镜头'], **action})
                for mouth in shot['口型']:
                    if not {p['人物'] for p in mouth['人物状态']}.issubset(local_people):
                        errors.append('口型人物不在本镜头人物中')
                    start, end = mouth['人声时间段'].split('-')
                    if not (time_seconds(source['起']) <= time_seconds(start) < time_seconds(end) <= time_seconds(source['止'])):
                        errors.append('人声时间段越出当前镜头')
                    people = [p['人物'] for p in mouth['人物状态']]
                    if len(people) != len(set(people)):
                        errors.append('同一人声时间段的嘴部状态人物重复')
            # 大段曾出现整段只写“无明显动作”的漏看；有人物镜头较多且大半无动作时要求重看。
            people = [s for s in data['镜头'] if s['画面人物']]
            idle = sum(1 for s in people if not s['动作'])
            if len(people) >= IDLE_MIN_SHOTS and idle > len(people) * IDLE_MAX_SHARE:
                errors.append(f'有画面人物的 {len(people)} 个镜头中 {idle} 个没有动作条目，疑似漏看；'
                              '逐镜头重新查看，只写画面中看得到的动作，确实没有动作的镜头保持为空')
            temporary = []
            for person in sorted(declared - known):
                temporary.append({'id': person, '片段': segment['id'],
                                  '外观': next((p['外观'] for s in data['镜头'] for p in s['画面人物']
                                               if p['人物'] == person and p.get('外观')), ''),
                                  '出现镜头': [s['镜头'] for s in data['镜头']
                                             if any(p['人物'] == person for p in s['画面人物'])],
                                  '映射': None, '映射依据': ''})
            check('临时人物', temporary)
            check('动作记录', rows)
        elif role == '成稿':
            script = context / '工作稿.md'
            if script.is_symlink():
                script.unlink()
            script.write_text(data['工作稿'], encoding='utf-8')
            if not data['工作稿'].splitlines() or data['工作稿'].splitlines()[0].strip() != f'第{int(episode.name[1:])}集':
                errors.append('工作稿首行集标题不正确')
            check('对照表', data['对照表'])
            if any(row['行'] > len(data['工作稿'].splitlines()) for row in data['对照表']):
                errors.append('对照表行号超出工作稿')
            quality = check_script(context)
            errors.extend(f'{e["检查项"]}{e["位置"]}: {e["问题"]}' for e in quality['错误'])
        elif role == '角色卡':
            if data['临时人物映射'] or data['新增人物']:
                errors.append('定向核对/总结不得改变冻结分组或新增人物')
            if package.mode == '总结' and data['对白归属']:
                errors.append('总结不得重新分配对白归属')
            for _, message in name_errors(data['新增人物']):
                errors.append('新增人物名称：' + message)
            from .rolecard import _validate_attribution
            table_path = episode / '本集人物.json'
            table = load(table_path) if table_path.exists() else {'人物': []}
            confirmed = [r for r in load(context / '角色卡.json') if r['状态'] == '已确认']
            allowed = {r['id'] for r in confirmed} | {r['id'] for r in table['人物']}
            for row in data['补充资料']:
                if row['角色'] not in allowed or not row['依据'].strip():
                    errors.append('补充资料须绑定冻结人物表且有依据')
                if '声线' in row and row['角色'] not in {r['id'] for r in confirmed}:
                    errors.append('声线建议只能涉及已获批C')
            sounds = load(episode / '声音记录.json')
            shots = {s['id'] for s in load(episode / '镜头表.json')['镜头']}
            references = {'句': {s['id'] for s in sounds}, '镜头': shots}
            for row in data['待核对']:
                position = row['位置']
                if not set(position) <= references.keys() or any(
                        not isinstance(value, str) or value not in references[kind]
                        for kind, value in position.items() if kind in references):
                    errors.append('资料待核对位置须引用合法本集句或镜头')
                if not row['问题'].strip():
                    errors.append('资料待核对问题不能为空')
            wanted = {r['句'] for r in data['对白归属']}
            batch = [s for s in sounds if s['id'] in wanted]
            # 资料问题不必伪造对白修改；真正改归属的句仍走完整批次/同句阻断校验。
            attribution_pending = [r for r in data['待核对']
                                   if isinstance(r['位置'].get('句'), str) and r['位置']['句'] in wanted]
            try:
                _validate_attribution({'对白归属': data['对白归属'], '补充建议': [], '待核对': attribution_pending},
                                      batch, table, {'已确认角色': confirmed}, shots)
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(str(exc))
        elif role == '验收':
            if data['集'] != int(episode.name[1:]):
                errors.append('验收返回集号不一致')
            issues = [{'id': f'{episode.name}-Q{i:03d}', '来源': '验收',
                       '处理职责': '成稿', '状态': '待修正', '复核结果': None, **row}
                      for i, row in enumerate(data['问题'], 1)]
            check('问题清单', issues)
            for level in ('阻断', '规范'):
                if data['统计'][level] != sum(r['级别'] == level for r in data['问题']):
                    errors.append('问题统计不一致：' + level)
            if (data['问题'] or any(r['结果'] == '未通过' for r in data['复核'])) and data['结论'] == '通过':
                errors.append('有未解决问题时不能判定通过')
    return errors
