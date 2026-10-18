"""JSON Schema 与跨文件编号、时间校验。"""

import argparse
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator
from s2s.names import name_errors

SCHEMA_DIR = Path(__file__).with_name('schemas')
TYPE_ALIASES = {'roles': '角色卡', 'shots': '镜头表', 'segments': '片段表',
    'sound': '声音记录', 'subtitles': '字幕记录', 'actions': '动作记录',
    'attribution': '归属表', 'pending': '待核对', 'issues': '问题清单',
    'mapping': '对照表', 'state': '状态', 'series_state': '全剧状态',
    'temporary_people': '临时人物', 'role_changes': '角色变更申请', 'dense_frames': '加密帧索引',
    '帧': '加密帧索引', 'lipsync': '口型记录', 'diarization': '声纹聚类',
    'voiceprint_candidates': '声纹候选', 'speech_regions': '人声段', 'decision_migrations': '裁定迁移'}
CURVE_TIME_RE = re.compile(r'^[0-9]{2}:[0-5][0-9]\.(?:[0-9]|[0-9]{3})$')
TIME_RE = re.compile(r'^[0-9]{2}:[0-5][0-9]\.[0-9]$')
ID_RE = re.compile(r'^E([0-9]{2,})-([SKAQPVTRN])([0-9]+)(?:-T([1-9][0-9]*))?$')
ID_FIELDS = {'id', '标签', '声线', '镜头', '句', '动作', '执行者', '对象', '说话人',
             '相关人物', '衔接前', '衔接后', '首次出场', '未决问题', '本集新增人物',
             '片段', '出现镜头', '映射', '涉及角色', '新句号', 'cannot-link', '成员', '审核C', '候选C'}


def time_seconds(value: str) -> float:
    if not isinstance(value, str) or not TIME_RE.fullmatch(value):
        raise ValueError(f'时间须为 MM:SS.s：{value}')
    return int(value[:2]) * 60 + float(value[3:])


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _location(parts) -> str:
    return '$' + ''.join(f'[{part}]' if isinstance(part, int) else f'.{part}' for part in parts)


def _episode_from_path(path: Path) -> int | None:
    for parent in path.parents:
        if re.fullmatch(r'E[0-9]{2,}', parent.name):
            return int(parent.name[1:])
    return None


def _walk(value, path=()):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            if key != '输入版本':
                yield from _walk(child, (*path, key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _walk(child, (*path, i))


def _items(data, field=None):
    if field:
        data = data.get(field, []) if isinstance(data, dict) else []
    return data if isinstance(data, list) else []


def _check_frames(record, path, fail, dense=False):
    start, end = time_seconds(record['起']), time_seconds(record['止'])
    previous = None
    for i, frame in enumerate(record['帧']):
        frame_path = (*path, '帧', i)
        moment = time_seconds(frame['时间'])
        if not start <= moment <= end:
            fail((*frame_path, '时间'), '帧时间超出起止范围')
        if previous is not None and moment < previous:
            fail((*frame_path, '时间'), '帧未按时间升序排列')
        previous = moment
        if dense:
            expected = f't{frame["时间"].replace(":", "-")}.jpg'
            if frame['文件'] != expected:
                fail((*frame_path, '文件'), '加密帧文件名与时间不一致')
        elif frame['文件'].split('_')[0] != record['id'].split('-')[-1]:
            fail((*frame_path, '文件'), '帧文件名与镜头编号不一致')


def _check_shots(data, fail):
    shots = data['镜头']
    if not shots:
        if data['时长'] != '00:00.0':
            fail(('镜头',), '非零时长的镜头表不能为空')
        return
    if shots[0]['起'] != '00:00.0':
        fail(('镜头', 0, '起'), '首镜头必须从 00:00.0 开始')
    if shots[-1]['止'] != data['时长']:
        fail(('镜头', len(shots) - 1, '止'), '末镜头结束时间必须等于时长')
    for i, shot in enumerate(shots):
        if i and shot['起'] != shots[i - 1]['止']:
            fail(('镜头', i, '起'), '相邻镜头必须首尾相接')
        _check_frames(shot, ('镜头', i), fail)


def _check_segments(data, source, fail):
    shots, segments = source['镜头'], data['片段']
    shot_ids = [shot['id'] for shot in shots]
    if [ident for segment in segments for ident in segment['镜头']] != shot_ids:
        fail(('片段',), '全部片段必须按顺序恰好覆盖镜头表一次')
    by_id = {shot['id']: shot for shot in shots}
    for i, segment in enumerate(segments):
        ids = segment['镜头']
        if ids[0] in by_id and segment['起'] != by_id[ids[0]]['起']:
            fail(('片段', i, '起'), '片段起点必须等于首镜头起点')
        if ids[-1] in by_id and segment['止'] != by_id[ids[-1]]['止']:
            fail(('片段', i, '止'), '片段终点必须等于末镜头终点')
        before = segments[i - 1]['镜头'][-1] if i else None
        after = segments[i + 1]['镜头'][0] if i + 1 < len(segments) else None
        for field, expected in (('衔接前', before), ('衔接后', after)):
            if segment[field] != expected:
                fail(('片段', i, field), f'{field}必须是相邻片段边界镜头：{expected}')


def _check_lipsync(data, sound, file, reference, fail):
    by_id = {r['id']: r for r in _items(sound)
             if isinstance(r, dict) and isinstance(r.get('id'), str)}
    for i, row in enumerate(_items(data, '句')):
        if not isinstance(row, dict) or not isinstance(row.get('句'), str):
            continue
        source = by_id.get(row['句'])
        if source:
            for field in ('文本', '起', '止'):
                if field in row and row[field] != source.get(field):
                    fail(('句', i, field), f'{field}与同目录声音记录不一致')
        seen = set()
        for j, track in enumerate(_items(row, '轨迹')):
            if not isinstance(track, dict) or not isinstance(track.get('id'), str):
                continue
            if not re.fullmatch(r'E[0-9]{2,}-K[0-9]{3,}-F[1-9][0-9]*', track['id']):
                continue
            path = ('句', i, '轨迹', j, 'id')
            if track['id'] in seen:
                fail(path, f'轨迹编号重复：{track["id"]}')
            seen.add(track['id'])
            shot_id = track['id'].rsplit('-F', 1)[0]
            reference(path, shot_id, '镜头表')
            if isinstance(row.get('镜头'), list) and shot_id not in row['镜头']:
                fail(path, '轨迹所属镜头未列在本句镜头中')
    # 路径保留生产端写法：绝对路径、本集相对路径、项目相对路径。
    for path, value in _walk(data):
        if len(path) >= 2 and path[-2] == '依据帧' and isinstance(value, str):
            frame = Path(value)
            candidates = [frame] if frame.is_absolute() else [file.parent / frame, frame]
            if not any(p.is_file() for p in candidates):
                fail(path, f'依据帧文件不存在：{value}')


def _check_diarization(data, fail):
    rows, categories = data['句列表'], data['类别']
    voices = [c['声线'] for c in categories]
    for i, row in enumerate(rows):
        if not row['待定'] and row['声线'] not in voices:
            fail(('句列表', i, '声线'), '声线未在类别中声明')
    for i, category in enumerate(categories):
        members = [r for r in rows if r['声线'] == category['声线']]
        if category['句数'] != len(members):
            fail(('类别', i, '句数'), '类别句数与句列表不一致')
        if category['聚类句数'] > category['句数']:
            fail(('类别', i, '聚类句数'), '聚类句数不得超过类别句数')
        if all('参与聚类' in r for r in members):
            if category['聚类句数'] != sum(r['参与聚类'] for r in members):
                fail(('类别', i, '聚类句数'), '聚类句数与句列表不一致')
        ids = {r['句'] for r in members}
        for j, row in enumerate(category['代表句']):
            if row['句'] not in ids:
                fail(('类别', i, '代表句', j, '句'), '代表句不属于当前类别')
    order = data['矩阵声线顺序']
    if set(order) != set(voices):
        fail(('矩阵声线顺序',), '矩阵声线顺序必须恰好包含全部类别声线')
    matrix = data['类间相似度矩阵']
    if len(matrix) != len(order) or any(len(row) != len(order) for row in matrix):
        fail(('类间相似度矩阵',), '类间相似度矩阵维度与矩阵声线顺序不一致')


def episode_people_errors(directory, table):
    """N只在本集登记；原始输入变化使登记失效，自身审核落板不改变原始声明。"""
    directory = Path(directory)
    errors = []
    schema = read_json(SCHEMA_DIR / '本集人物.schema.json')
    if not Draft202012Validator(schema).is_valid(table):
        return ['本集人物结构不合法']
    rows = table['人物']
    if not rows and table['输入版本'] is None:
        return []  # 初始化空表不是一次成功识别。
    version = table['输入版本']
    if not isinstance(version, dict) or not {'规则', '文件', '临时声明', '已确认角色', '模型', '候选', '人物文件', '冻结合并'} <= version.keys():
        return ['本集人物缺少完整输入版本']
    if version['规则'] != '角色卡两阶段-v2-起点owner-整句帧-固定N':
        errors.append('本集人物规则版本过期')
    try:
        for filename, size, mtime in version['文件']:
            path = Path(filename)
            current = (path.stat().st_size, path.stat().st_mtime_ns) if path.is_file() else (None, None)
            if current != (size, mtime):
                errors.append('本集人物输入版本过期：' + filename)
        temporary = read_json(directory / '临时人物.json')
        declaration = [{k: p[k] for k in ('id', '片段', '外观', '出现镜头')} for p in temporary]
        if declaration != version['临时声明']:
            errors.append('本集人物临时声明版本过期')
        roles_path = directory / '角色卡.json'
        roles = read_json(roles_path if roles_path.exists() else directory.parent / '角色卡.json')
        known = {r['id']: r for r in roles if r['状态'] == '已确认'}
        for source in version['已确认角色']:
            if source != {k: v for k, v in known.get(source['id'], {}).items() if k != '性格总结'}:
                errors.append('本集人物已确认C版本过期')
        ids, members = set(), set()
        for row in rows:
            if row['id'] in ids or not row['id'].startswith(directory.name + '-N'):
                errors.append('本集人物编号重复或跨集')
            ids.add(row['id'])
            if any(p in members or p not in {t['id'] for t in temporary} for p in row['成员']):
                errors.append('本集人物成员重复或未登记')
            members.update(row['成员'])
            if row['候选C'] is not None and row['候选C'] not in known:
                errors.append('本集人物候选C不是已确认角色')
            if row['审核C'] is not None:
                if row['审核C'] not in known or row['状态'] != '已审核' or not (row['审核依据'] or '').strip():
                    errors.append('本集人物审核C/状态/依据不合法')
            elif row['状态'] != '待审核' or row['审核依据'] is not None:
                errors.append('本集人物未审核状态不合法')
        from s2s.dispatch.rolecard import _freeze
        from s2s.dispatch.roles import settings_for
        from s2s.dispatch.materials import Package
        if settings_for('角色卡', Package('', view_all=True)) != version['模型']:
            errors.append('本集人物模型配置版本过期')
        expected, _ = _freeze(directory, version['冻结合并'], {'临时人物': version['候选']}, version)
        normalized = [{**r, '审核C': None, '审核依据': None, '状态': '待审核'} for r in rows]
        if normalized != expected['人物']:
            errors.append('本集人物与冻结合并不一致')
        source_dir = next(Path(p).parent for p, _, _ in version['人物文件'] if Path(p).name == '临时人物.json')
        declared_paths = {p for p, _, _ in version['文件'] if Path(p).parent == (source_dir / '派发结果/动作识别').resolve()}
        if declared_paths != {str(p.resolve()) for p in (source_dir / '派发结果/动作识别').glob('*.json')}:
            errors.append('本集人物动作声明来源版本过期')
        errors.extend(message for _, message in name_errors(rows))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        errors.append('本集人物版本不可核验：' + str(exc))
    return errors


def validate_file(file: str | Path, data_type: str | None = None) -> dict:
    file = Path(file)
    errors = []

    def fail(path, message):
        errors.append({'位置': _location(path), '问题': message})

    requested_kind = data_type if data_type else file.stem
    kind = TYPE_ALIASES.get(requested_kind, requested_kind)
    result = {'文件': str(file), '通过': False, '错误': errors}
    if not kind or '/' in kind or '\\' in kind:
        fail((), '未知数据类型')
        return result
    schema_path = SCHEMA_DIR / f'{kind}.schema.json'
    if not schema_path.is_file():
        fail((), f'未知数据类型：{kind}；使用 --type 指定')
        return result
    try:
        data = read_json(file)
    except (OSError, ValueError) as exc:
        fail((), f'无法读取 JSON：{exc}')
        return result
    schema = read_json(schema_path)
    for error in Draft202012Validator(schema).iter_errors(data):
        messages = {
            'required': f'必填字段不齐：{error.validator_value}',
            'type': f'字段类型错误，应为 {error.validator_value}',
            'enum': f'取值不在允许范围：{error.validator_value}',
            'pattern': f'格式不符：{error.instance}',
            'additionalProperties': '包含数据约定未定义的字段',
            'anyOf': '取值不符合任一允许类型或编号格式',
            'minItems': f'数组条目至少 {error.validator_value} 个',
            'maxItems': f'数组条目最多 {error.validator_value} 个',
            'minimum': f'数值不得小于 {error.validator_value}',
            'maximum': f'数值不得大于 {error.validator_value}',
        }
        fail(tuple(error.absolute_path), messages.get(error.validator, '字段不符合 JSON Schema'))
    schema_valid = not errors
    # 类型错误由 Schema 报出；只对类型正确的局部值进行语义检查。
    parent_ep = _episode_from_path(file)
    top_ep = data.get('集') if isinstance(data, dict) and isinstance(data.get('集'), int) else None
    if kind == '声纹候选' and isinstance(data, dict) and isinstance(data.get('集'), str):
        if re.fullmatch(r'E[0-9]{2,}', data['集']):
            top_ep = int(data['集'][1:])
    episode = parent_ep if parent_ep is not None else top_ep
    if parent_ep is not None and top_ep is not None and parent_ep != top_ep:
        fail(('集',), f'集号 {top_ep} 与目录 E{parent_ep:02d} 不一致')
    if episode is None and kind != '角色卡':
        for path, value in _walk(data):
            field = next((p for p in reversed(path) if isinstance(p, str)), None)
            is_id = field in ID_FIELDS and not (field == '动作' and kind != '对照表')
            if is_id and isinstance(value, str) and (match := ID_RE.fullmatch(value)):
                episode = int(match[1])
                break

    def times_and_episode(value, path=(), local_ep=episode):
        if isinstance(value, dict):
            own_ep = value.get('集')
            if kind in ('角色卡', '全剧状态') and isinstance(own_ep, int):
                local_ep = own_ep
            if '提出集' in value and isinstance(value['提出集'], int) and local_ep is not None:
                if value['提出集'] != local_ep:
                    fail((*path, '提出集'), f'提出集与 E{local_ep:02d} 不一致')
            start, end = value.get('起'), value.get('止')
            if isinstance(start, str) and isinstance(end, str) and TIME_RE.fullmatch(start) and TIME_RE.fullmatch(end):
                if time_seconds(start) > time_seconds(end):
                    fail(path, '起晚于止')
            for key, child in value.items():
                if key == '输入版本':
                    continue
                time_pattern = (CURVE_TIME_RE if kind == '口型记录' and key == '时间'
                                and any(p in ('曲线', '模型曲线') for p in path) else TIME_RE)
                if key in ('起', '止', '时间', '时长') and isinstance(child, str) and not time_pattern.fullmatch(child):
                    fail((*path, key), '曲线时间须为 MM:SS.s 或 MM:SS.sss' if time_pattern is CURVE_TIME_RE else '时间须为 MM:SS.s')
                times_and_episode(child, (*path, key), local_ep)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                times_and_episode(child, (*path, i), local_ep)
        elif isinstance(value, str):
            field = next((p for p in reversed(path) if isinstance(p, str)), None)
            is_id = field in ID_FIELDS and not (field == '动作' and kind != '对照表')
            match = ID_RE.fullmatch(value) if is_id else None
            if match and local_ep is not None and int(match[1]) != local_ep:
                fail(path, f'编号集号与 E{local_ep:02d} 不一致')
            if field == '场景' and re.fullmatch(r'场景[1-9][0-9]*-[1-9][0-9]*', value) and local_ep is not None:
                if int(value.split('-')[0][2:]) != local_ep:
                    fail(path, f'场景集号与第 {local_ep} 集不一致')
    times_and_episode(data)

    item_field = {'镜头表': '镜头', '片段表': '片段', '全剧状态': '集',
                  '口型记录': '句', '声纹聚类': '句列表', '声纹候选': '声线', '本集人物': '人物'}.get(kind)
    records = _items(data, item_field)
    key = {'归属表': '句', '对照表': '行', '全剧状态': '集', '口型记录': '句',
           '声纹聚类': '句', '声纹候选': '声线', '裁定迁移': '新句号'}.get(kind, 'id')
    seen = set()
    for i, record in enumerate(records):
        if isinstance(record, dict) and isinstance(record.get(key), (str, int)):
            ident = record[key]
            if ident in seen:
                fail((item_field, i, key) if item_field else (i, key), f'{key}重复：{ident}')
            seen.add(ident)

    companions = {}
    script_scenes = None
    board_dir = file.parent
    if kind in ('人声段', '裁定迁移'):
        board_dir = next((p.parent for p in file.parents if p.name == '素材'), file.parent)
    candidate_roles = (board_dir.parent if re.fullmatch(r'E[0-9]{2,}', board_dir.name)
                       else board_dir) / '角色卡.json'

    def load_companion(target_kind, target_ep=None):
        target_file = board_dir / f'{target_kind}.json'
        if target_kind == '角色卡':
            target_file = board_dir / '角色卡.json'
            if kind == '声纹候选':
                target_file = candidate_roles
            elif not target_file.exists():
                target_file = board_dir.parent / '角色卡.json'
        elif target_ep is not None and (episode != target_ep or kind == '角色卡'):
            series = board_dir.parent if _episode_from_path(file) is not None else board_dir
            target_file = series / f'E{target_ep:02d}' / f'{target_kind}.json'
        if target_kind == kind and target_file.resolve() == file.resolve():
            source = data
        elif target_file in companions:
            source = companions[target_file]
        else:
            try:
                source = read_json(target_file)
            except (OSError, ValueError) as exc:
                source = f'无法读取引用文件 {target_file}：{exc}'
            companions[target_file] = source
        if isinstance(source, str):
            return None, source
        return source, None

    def load_ids(target_kind, target_ep=None, id_field='id'):
        source, problem = load_companion(target_kind, target_ep)
        if problem:
            return None, problem
        field = {'镜头表': '镜头', '片段表': '片段', '本集人物': '人物'}.get(target_kind)
        return {r[id_field] for r in _items(source, field)
                if isinstance(r, dict) and isinstance(r.get(id_field), str)}, None

    def reference(path, ident, target_kind, id_field='id'):
        if not isinstance(ident, str):
            return
        match = ID_RE.fullmatch(ident)
        target_ep = int(match[1]) if match and target_kind != '临时人物' else None
        ids, problem = load_ids(target_kind, target_ep, id_field)
        if problem:
            fail(path, problem)
        elif ident not in ids:
            fail(path, f'引用编号不存在：{ident}（{target_kind}）')

    # 只检查结构化引用，不把依据、说明中的自然语言当成编号。
    ref_fields = {'镜头': '镜头表', '衔接前': '镜头表', '衔接后': '镜头表',
                  '首次出场': '镜头表', '句': '声音记录', '动作': '动作记录',
                  '未决问题': '问题', '本集新增人物': '角色卡',
                  '片段': '片段表', '出现镜头': '镜头表', '新句号': '声音记录',
                  'cannot-link': '声音记录'}
    for path, value in _walk(data):
        if not path or not isinstance(value, str):
            continue
        field = next((p for p in reversed(path) if isinstance(p, str)), None)
        if field == '场景' and kind in ('对照表', '待核对', '问题清单'):
            if script_scenes is None:
                script_scenes = set()
                for script_name in ('工作稿.md', '正式稿.md'):
                    script_path = file.parent / script_name
                    if script_path.is_file():
                        try:
                            text = script_path.read_text(encoding='utf-8')
                        except (OSError, ValueError) as exc:
                            fail(path, f'无法读取场景引用稿：{exc}')
                            break
                        if text.strip():
                            script_scenes = set(re.findall(r'^(场景[1-9][0-9]*-[1-9][0-9]*)：', text, re.MULTILINE))
                            break
            if value not in script_scenes:
                fail(path, f'引用场景不存在：{value}（工作稿或正式稿）')
        if kind == '声纹候选' and field == '角色' and candidate_roles.exists():
            reference(path, value, '角色卡')
        # id、描述、依据等都不是引用字段。
        if field == '动作' and kind != '对照表':
            continue
        if field in ref_fields and (ID_RE.fullmatch(value) or field == '本集新增人物'):
            target = ref_fields[field]
            if target == '问题':
                ids1, err1 = load_ids('待核对')
                ids2, err2 = load_ids('问题清单')
                if value not in (ids1 or set()) | (ids2 or set()):
                    fail(path, f'引用问题不存在：{value}' + (f'；{err1 or err2}' if err1 or err2 else ''))
            else:
                reference(path, value, target)
        if kind == '角色卡' and field == '标签':
            reference(path, value, '声音记录', '声线')
        if kind == '本集人物' and field == '成员':
            reference(path, value, '临时人物')
        if field in ('执行者', '对象', '说话人', '相关人物', '映射', '涉及角色', '审核C', '候选C'):
            if re.fullmatch(r'C[0-9]{3,}', value):
                reference(path, value, '角色卡')
            elif re.fullmatch(r'E[0-9]{2,}-P[0-9]{2,}-T[1-9][0-9]*', value):
                reference(path, value, '临时人物')
            elif re.fullmatch(r'E[0-9]{2,}-N[0-9]{3,}', value):
                reference(path, value, '本集人物')
                table, problem = load_companion('本集人物')
                if not problem:
                    for message in episode_people_errors(board_dir, table):
                        fail(path, message)
    if kind == '本集人物' and schema_valid:
        for message in episode_people_errors(board_dir, data):
            fail((), message)
    if kind == '归属表' and schema_valid:
        for i, row in enumerate(data):
            if not row['依据'].strip():
                fail((i, '依据'), '归属依据不能为空')
            if row['说话人'] is None:
                pending, problem = load_companion('待核对')
                if row['确定度'] != '存疑' or problem or not any(
                        p.get('位置', {}).get('句') == row['句'] and p.get('级别') == '阻断'
                        and p.get('状态') not in ('已修正', '已裁定') for p in pending or []):
                    fail((i, '说话人'), 'null归属仅限存疑且须有同句未决阻断')
            if row['形态'] is None or row.get('形态待核对'):
                pending, problem = load_companion('待核对')
                if problem or not any(p.get('位置', {}).get('句') == row['句'] and p.get('级别') == '阻断'
                                      and p.get('状态') not in ('已修正', '已裁定') for p in pending or []):
                    fail((i, '形态'), '形态待核对须有同句未决阻断，已确认身份证据仍保留')
    if kind == '口型记录':
        source, problem = load_companion('声音记录')
        if problem:
            fail(('句',), problem)
        _check_lipsync(data, source, file, reference, fail)
    if schema_valid:
        if kind == '角色卡':
            for i, message in name_errors(data):
                fail((i, '名称'), message)
        elif kind == '动作记录':
            shots, problem = load_companion('镜头表')
            if problem:
                fail((), problem)
            elif not Draft202012Validator(read_json(SCHEMA_DIR / '镜头表.schema.json')).is_valid(shots):
                fail((), '引用镜头表结构不符合数据约定')
            else:
                by_id = {s['id']: s for s in shots['镜头']}
                dense_frames = {}
                for index in (board_dir / '素材/frames_dense').rglob('帧.json'):
                    try:
                        dense = read_json(index)
                        if not validate_file(index, '加密帧索引')['通过'] or dense['视频'] != shots['视频']:
                            continue
                        for frame in dense['帧']:
                            dense_frames.setdefault(frame['文件'], []).append(time_seconds(frame['时间']))
                    except (OSError, ValueError):
                        continue
                for i, action in enumerate(data):
                    if not action['动作'].strip():
                        fail((i, '动作'), '动作内容不能为空')
                    shot = by_id.get(action['镜头'])
                    if shot is None:
                        continue
                    start, end = time_seconds(shot['起']), time_seconds(shot['止'])
                    if not start <= time_seconds(action['起']) <= time_seconds(action['止']) <= end:
                        fail((i,), '动作时间越出镜头范围')
                    frames = {f['文件'] for f in shot['帧']}
                    for j, frame in enumerate(action['依据帧']):
                        if frame not in frames and not any(start <= t < end for t in dense_frames.get(frame, [])):
                            fail((i, '依据帧', j), '依据帧不在本镜头或有效加密帧索引中：' + frame)
        if kind == '镜头表':
            _check_shots(data, fail)
        elif kind == '片段表':
            source, problem = load_companion('镜头表')
            if problem:
                fail(('片段',), problem)
            elif not list(Draft202012Validator(read_json(SCHEMA_DIR / '镜头表.schema.json')).iter_errors(source)):
                _check_segments(data, source, fail)
            else:
                fail(('片段',), '引用镜头表结构不符合数据约定')
        elif kind == '加密帧索引':
            _check_frames(data, (), fail, dense=True)
        elif kind == '临时人物':
            for i, person in enumerate(data):
                if person['id'].rsplit('-T', 1)[0] != person['片段']:
                    fail((i, '片段'), '临时人物编号与所属片段不一致')
        elif kind == '声纹聚类':
            _check_diarization(data, fail)
            seen_voices = set()
            for i, category in enumerate(data['类别']):
                if category['声线'] in seen_voices:
                    fail(('类别', i, '声线'), f'声线重复：{category["声线"]}')
                seen_voices.add(category['声线'])
        elif kind == '人声段':
            for i, region in enumerate(data):
                if region['起'] > region['止']:
                    fail((i,), '起晚于止')
                if i and region['起'] < data[i - 1]['起']:
                    fail((i, '起'), '人声段未按起点排序')
                if i and region['起'] < data[i - 1]['止']:
                    fail((i, '起'), '人声段重叠')
        if kind == '状态':
            seen_requests = set()
            for i, request in enumerate(data['角色变更申请']):
                if request['id'] in seen_requests:
                    fail(('角色变更申请', i, 'id'), f'id重复：{request["id"]}')
                seen_requests.add(request['id'])
    result['通过'] = not errors
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='校验黑板 JSON 格式与引用')
    parser.add_argument('文件')
    parser.add_argument('--type', dest='data_type')
    args = parser.parse_args()
    result = validate_file(args.文件, args.data_type)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['通过'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
