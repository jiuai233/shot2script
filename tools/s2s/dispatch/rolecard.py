"""角色卡建立的固定两阶段流程；模型判断人物，代码负责覆盖、编号和受控保存。"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re
import tempfile
import time

from jsonschema import Draft202012Validator

from s2s import agent, config
from s2s.evidence import evidence_problem
from s2s.names import name_errors
from s2s.run_record import save
from s2s.validate import SCHEMA_DIR, time_seconds, validate_file
from .materials import Package, dumps, frame_path, load, render
from .results import extract_json
from .rolecard_materials import build_rolecard_materials
from .transport import decode_response, pi_options

RULE = '角色卡两阶段-v2-起点owner-整句帧-固定N'
SCHEMAS = Path(__file__).with_name('role_schemas.json')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _schema(kind, value):
    schema = load(SCHEMAS)[kind] if kind in ('人物合并', '对白归属') else load(SCHEMA_DIR / f'{kind}.schema.json')
    errors = [f'{list(e.absolute_path)}: {e.message}' for e in Draft202012Validator(schema).iter_errors(value)]
    _require(not errors, kind + '校验失败：' + '; '.join(errors))


def _confirmed(episode):
    return [{k: v for k, v in row.items() if k != '性格总结'}
            for row in load(episode.parent / '角色卡.json') if row['状态'] == '已确认']


def _declarations(episode, people, shots):
    by_id = {p['id']: p for p in people}
    _require(len(by_id) == len(people), '临时人物 ID 重复')
    segments = {s['id']: s for s in load(episode / '片段表.json')['片段']}
    for person in people:
        _require(person['片段'] in segments and person['片段'].startswith(episode.name + '-P'),
                 '临时人物片段不属于本集或不存在')
        _require(re.fullmatch(re.escape(person['片段']) + r'-T[1-9][0-9]*', person['id']),
                 '临时人物ID与本集片段不一致')
    declared = {}
    known = {r['id'] for r in _confirmed(episode)}
    for path in sorted((episode / '派发结果/动作识别').glob('*.json')):
        result = load(path)
        _require(result['片段'] in segments and result['片段'].startswith(episode.name + '-P'),
                 '动作声明片段不属于本集或不存在')
        for row in result['镜头']:
            _require(row['镜头'] in segments[result['片段']]['镜头'], '动作声明镜头不属于本片段，不能使用衔接镜头')
            local = set()
            for person in row['画面人物']:
                ident = person['人物']
                _require(ident not in local, '同镜头人物重复声明')
                local.add(ident)
                if ident in known:
                    continue
                _require(re.fullmatch(re.escape(result['片段']) + r'-T[1-9][0-9]*', ident), '非法临时人物声明：' + ident)
                target = declared.setdefault(ident, {'片段': result['片段'], '外观': person.get('外观'), '镜头': set()})
                if not target['外观'] and person.get('外观'):
                    target['外观'] = person['外观']
                _require(target['片段'] == result['片段'], '人物片段冲突')
                target['镜头'].add(row['镜头'])
    _require(set(declared) == set(by_id), '临时清单与动作声明不完整一致')
    for ident, row in declared.items():
        person = by_id[ident]
        _require(person['片段'] == row['片段'] and person['外观'] == row['外观']
                 and person['出现镜头'] == [s['id'] for s in shots if s['id'] in row['镜头']],
                 '临时清单与动作声明片段/出现镜头/外观不一致：' + ident)


def _materials(episode):
    people = load(episode / '临时人物.json')
    _schema('临时人物', people)
    for kind in ('镜头表', '片段表', '声音记录', '字幕记录'):
        checked = validate_file(episode / f'{kind}.json', kind)
        _require(checked['通过'], '角色卡输入不合法：' + dumps(checked['错误']))
    shots = load(episode / '镜头表.json')['镜头']
    _declarations(episode, people, shots)
    if not people:
        return {'临时人物': [], '身份证据字幕': [r for r in load(episode / '字幕记录.json')
                if r['类型'] in ('人物字卡', '地点字幕')], 'images': [], 'labels': [], '来源文件': []}
    materials = build_rolecard_materials(episode)
    _require(not materials['缺失项'], '缺少合法人物候选帧')
    _require(not materials['统计']['超过图片上限'], '人物合并图片超过配置上限')
    return materials


def _stats(paths):
    return [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] if p.is_file()
            else [str(p.resolve()), None, None] for p in sorted(set(paths))]


def _apply_target_version(episode):
    # 只保存apply整体覆盖的目标，不能放入落板后仍须有效的本集人物源版本。
    return _stats([episode.parent / '角色卡.json',
                   *[episode / f'{kind}.json' for kind in ('本集人物', '临时人物', '归属表', '待核对', '状态')]])


def source_version(episode, materials, *, prompt_file=None):
    """沿用路径/size/mtime快照；可写人物文件另存原始版本与声明投影。"""
    template = Path(prompt_file).resolve() if prompt_file else config.PROJECT_ROOT / 'prompts/03-角色卡.md'
    paths = [episode / f'{name}.json' for name in ('镜头表', '片段表', '声音记录', '字幕记录',
             '口型记录', '声纹聚类', '声纹候选')]
    paths += list((episode / '派发结果/动作识别').glob('*.json'))
    paths += [episode / p for p in materials['来源文件'] if Path(p).name != '临时人物.json']
    paths += [Path(p) for p in materials['images']]
    paths += [frame_path(episode / '素材/frames', f['文件'])
              for s in load(episode / '镜头表.json')['镜头'] for f in s['帧']]
    paths += [template, config.PROJECT_ROOT / 'prompts/00-数据约定.md', SCHEMAS,
              *[SCHEMA_DIR / f'{k}.schema.json' for k in ('本集人物', '临时人物', '归属表', '待核对')]]
    from .roles import settings_for
    return {'规则': RULE, '模型': settings_for('角色卡', Package('', view_all=True)),
            '文件': _stats(paths), '人物文件': _stats([episode / '临时人物.json', episode.parent / '角色卡.json']),
            '临时声明': [{k: p[k] for k in ('id', '片段', '外观', '出现镜头')}
                        for p in load(episode / '临时人物.json')],
            '已确认角色': _confirmed(episode), '候选': materials['临时人物']}


def _freeze(episode, merge, materials, version):
    _schema('人物合并', merge)
    people = {r['id']: r for r in materials['临时人物']}
    known = {r['id'] for r in version['已确认角色']}
    members = [p for g in merge['分组'] for p in g['成员']]
    _require(len(members) == len(set(members)) and set(members) == set(people), '人物分组必须恰好覆盖每个T一次')
    shots = {s['id']: i for i, s in enumerate(load(episode / '镜头表.json')['镜头'])}
    groups = sorted(merge['分组'], key=lambda g: (min(shots[s] for p in g['成员'] for s in people[p]['出现镜头']), sorted(g['成员'])))
    used, rows, mappings = set(), [], []
    for group in groups:
        _require(group['依据'].strip() and group['外观'].strip(), '合并依据/外观不能为空')
        _require(group['已有角色'] is None or group['已有角色'] in known, '非法已有C引用')
        _require(not name_errors([group]), '人物名称不合法')
        _require(group['确定度'] != '存疑' or group['待核对'], '存疑分组必须说明待核对')
        adopted = group['已有角色'] if group['确定度'] == '确认' and not group['待核对'] else None
        if adopted:
            _require(adopted not in used, '同一C不得重复占用不同确认组')
            used.add(adopted)
            ident = adopted
        else:
            ident = f'{episode.name}-N{len(rows) + 1:03d}'
            first = min((people[p]['候选帧'][0] for p in group['成员']),
                        key=lambda f: (shots[f['镜头']], time_seconds(f['时间']), f['文件']))
            rows.append({'id': ident, **{k: group[k] for k in ('名称', '名称类型', '外观', '依据', '确定度', '待核对')},
                         '成员': sorted(group['成员']), '参考帧': first['文件'], '候选C': group['已有角色'],
                         '审核C': None, '审核依据': None, '状态': '待审核'})
        mappings.extend({'临时': p, '映射': ident, '依据': group['依据'], '确定度': group['确定度']}
                        for p in sorted(group['成员']))
    _require(not name_errors(rows), '本集人物名称重复或不合法')
    table = {'版本': 1, '输入版本': version, '人物': rows}
    _schema('本集人物', table)
    return table, mappings


def sentence_batches(episode):
    sounds = load(episode / '声音记录.json')
    segments = load(episode / '片段表.json')['片段']
    _require(len({s['id'] for s in sounds}) == len(sounds), '声音句号重复')
    _require(len({s['id'] for s in segments}) == len(segments), '片段编号重复')
    for i, segment in enumerate(segments):
        _require(time_seconds(segment['起']) < time_seconds(segment['止']), '片段边界不合法')
        if i:
            _require(segments[i - 1]['止'] == segment['起'], '片段有空隙或重叠')
    batches = {s['id']: [] for s in segments}
    for sound in sounds:
        start, end = time_seconds(sound['起']), time_seconds(sound['止'])
        _require(start < end, '声音句边界不合法')
        owners = [s for s in segments if time_seconds(s['起']) <= start < time_seconds(s['止'])]
        _require(len(owners) == 1 and end <= time_seconds(segments[-1]['止']), '声音句必须恰好有一个片段owner且完整落在本集')
        batches[owners[0]['id']].append(sound)
    return [(s, batches[s['id']]) for s in segments if batches[s['id']]]


def _evidence(episode, sounds, wanted):
    data, states = {}, {}
    for name in ('口型记录', '声纹聚类', '声纹候选'):
        path = episode / f'{name}.json'
        if not path.is_file():
            states[name] = '缺失；不能作为否定证据'
            continue
        value = load(path)
        try:
            problem = evidence_problem(name, value, sounds)
        except (ValueError, TypeError, KeyError, AttributeError):
            problem = '证据结构无效，须重新生成'
        if problem:
            states[name] = '过期；' + problem + '；不能作为否定证据'
            continue
        states[name] = '有效候选，仍须逐句核对'
        if name == '口型记录':
            data[name] = {'句': [{**{k: v for k, v in row.items() if k not in ('轨迹', '声线')},
                                  '声线': next(s['声线'] for s in sounds if s['id'] == row['句']),
                                  '轨迹': [{k: v for k, v in t.items() if k not in ('曲线', '模型曲线')}
                                           for t in row.get('轨迹', [])]}
                                 for row in value['句'] if row['句'] in wanted]}
        elif name == '声纹聚类':
            data[name] = {'句列表': [r for r in value['句列表'] if r['句'] in wanted]}
        else:
            voices = {s['声线'] for s in sounds if s['id'] in wanted}
            data[name] = {'声线': [r for r in value.get('声线', []) if r['声线'] in voices]}
    return {'证据状态': states, **data}


def _package(episode, stage, data, images, labels, *, prompt_file=None):
    images = [Path(p) for p in images]
    _require(len(images) <= config.get_vision()['max_images_per_call'], stage + '图片超过配置上限')
    _require(all(p.is_file() for p in images), '图片不存在')
    template = Path(prompt_file).resolve() if prompt_file else config.PROJECT_ROOT / 'prompts/03-角色卡.md'
    prompt = render(template.read_text(encoding='utf-8'), {
        '集号': episode.name[1:], '剧名': episode.parent.name, '建立 | 定向核对 | 总结': '建立',
        '定向核对时填写问题编号、句号和镜头范围': ''})
    prompt += '\n\n## 本次阶段：' + stage + '\n' + dumps(data)
    schema = load(SCHEMAS)[stage]
    return Package(prompt, images=images, labels=labels, view_all=True, mode=stage,
                   system='只执行当前阶段；必须查看全部附图。只返回精确Schema的JSON：\n' + dumps(schema))


def _attribution_package(episode, segment, batch, table, mappings, version, *, prompt_file=None):
    sounds = load(episode / '声音记录.json')
    wanted = {s['id'] for s in batch}
    indexes = [i for i, s in enumerate(sounds) if s['id'] in wanted]
    neighbors = {j for i in indexes for j in (i - 1, i + 1) if 0 <= j < len(sounds)}
    shots = [s for s in load(episode / '镜头表.json')['镜头']
             if any(time_seconds(s['起']) < time_seconds(b['止']) and time_seconds(s['止']) > time_seconds(b['起']) for b in batch)]
    _require(shots, '对白句没有覆盖镜头')
    images, labels = {}, {}
    for shot in shots:
        for frame in shot['帧']:
            path = frame_path(episode / '素材/frames', frame['文件'])
            images[path] = None
            labels.setdefault(path, []).append(f"{frame['文件']}｜{shot['id']}｜{frame['时间']}")
    data = {'片段': segment['id'], '冻结人物表': {'已确认角色': version['已确认角色'], '本集人物': table['人物']},
            '临时人物映射': mappings, '负责完整句': batch,
            '相邻句仅参考不得输出': [sounds[i] for i in sorted(neighbors) if sounds[i]['id'] not in wanted],
            '镜头表': shots, '身份证据字幕': [r for r in load(episode / '字幕记录.json') if r['类型'] in ('人物字卡', '地点字幕')],
            **_evidence(episode, sounds, wanted)}
    package = _package(episode, '对白归属', data, list(images), ['；'.join(labels[p]) for p in images], prompt_file=prompt_file)
    package.segment = segment
    return package


def _validate_attribution(data, batch, table, version, shots):
    _schema('对白归属', data)
    expected = [s['id'] for s in batch]
    got = [r['句'] for r in data['对白归属']]
    _require(len(got) == len(set(got)) and set(got) == set(expected), '对白归属必须恰好覆盖本批句一次，不输出上下文句')
    known = {r['id'] for r in version['已确认角色']} | {r['id'] for r in table['人物']}
    blocked = {r['位置']['句'] for r in data['待核对'] if r['级别'] == '阻断'}
    for row in data['待核对']:
        _require(row['位置']['句'] in expected and row['问题'].strip(), '待核对引用非本批句或空问题')
        _require(not row['位置'].get('镜头') or row['位置']['镜头'] in shots, '待核对镜头引用未授权')
    for row in data['对白归属']:
        _require(row['依据'].strip(), '对白归属依据不能为空')
        _require(row['说话人'] is None or row['说话人'] in known, '说话人不在冻结人物表')
        _require(row['说话人'] is not None or row['确定度'] == '存疑', 'null说话人仅限存疑')
        _require(row['确定度'] != '存疑' or row['句'] in blocked, '存疑归属必须有同句阻断')
        _require(not (row['形态'] is None or row.get('形态待核对')) or row['句'] in blocked,
                 '形态未定须有同句阻断；可保留已确认身份，不伪造词界')
        sound = next(s for s in batch if s['id'] == row['句'])
        mixed = any('换人' in reason or '重叠' in reason for reason in sound.get('核对', {}).get('异常', []))
        _require(not mixed or row['确定度'] == '存疑' and row['句'] in blocked,
                 '句内换人或重叠须拆句或保留同句阻断，不能整句确认一个人')
    for row in data['补充建议']:
        _require(row['角色'] in known and row['依据'].strip(), '补充建议人物/依据不合法')


def _run_stage(episode, package, validate, output, *, timeout, call, validator, records,
               request_budget=None, resume=False):
    from .roles import DispatchError, settings_for
    settings = settings_for('角色卡', package)
    cached = output.parent / (output.stem + '-阶段') / (package.mode + '-' + (package.segment['id'] if package.segment else '整集') + '.json')
    signature = {'资料': package.signature(), '模型': settings, '模拟调用': call is not None or validator is not None}
    if resume and cached.is_file():
        previous = load(cached)
        if previous.get('输入版本') == signature:
            validate(previous['结果'])
            if not validator or not validator('角色卡', previous['结果'], episode, package):
                records.append({'阶段': package.mode, '片段': package.segment['id'] if package.segment else None, '复用': True})
                return previous['结果']
    prompt = package.prompt
    with pi_options(settings) as options:
        for attempt in range(3):
            config.check_model(settings['model'])
            record = {'阶段': package.mode, '片段': package.segment['id'] if package.segment else None,
                      '尝试': attempt + 1, '授权帧数': len(package.images)}
            records.append(record)
            started = time.monotonic()
            try:
                response = decode_response((call or agent.run)('角色卡', prompt, attachments=[], images=package.images,
                           labels=package.labels, system=agent.ROLE_SYSTEM['角色卡'] + '\n' + package.system,
                           **({'request_budget': request_budget.scoped(
                               f'角色卡/{package.mode}/{record["片段"] or "整集"}/格式尝试{attempt+1}')}
                              if request_budget is not None else {}),
                           **{**settings, **options, 'timeout': timeout}))
                record.update({'token用量': response.usage or None, '已读帧': list(map(str, package.images))})
            finally:
                record['耗时秒'] = round(time.monotonic() - started, 3)
            try:
                result = extract_json(response.text)
                validate(result)
                if validator:
                    _require(not validator('角色卡', result, episode, package), '附加校验失败')
                if resume:
                    save(cached, {'输入版本': signature, '结果': result})
                return result
            except (ValueError, TypeError, KeyError) as exc:
                debug = output.parent / '校验失败'
                debug.mkdir(parents=True, exist_ok=True)
                (debug / f'{output.stem}-{package.mode}-{time.time_ns()}-{attempt}.txt').write_text(response.text + '\n' + str(exc), encoding='utf-8')
                if attempt == 2:
                    raise DispatchError('格式校验失败，已重答 2 次：' + str(exc)) from exc
                prompt = package.prompt + '\n上次返回：\n' + response.text + '\n校验错误：' + str(exc) + '\n重新查看同一批附图，完整重答；不得改变冻结人物表。'


def validate_two_stage_result(episode, result, *, prompt_file=None):
    _require(isinstance(result, dict) and set(result) == {'结果版本', '模式', '输入版本', '覆盖目标版本', '合并结果', '本集人物', '临时人物映射', '对白归属', '补充建议', '待核对'}, '角色卡结果字段不完整')
    _require(result['结果版本'] == 2 and result['模式'] == '建立', '只接受两阶段建立结果版本2')
    _require(result['覆盖目标版本'] == _apply_target_version(episode), '角色卡覆盖目标版本过期')
    materials = _materials(episode)
    version = {**source_version(episode, materials, prompt_file=prompt_file), '冻结合并': result['合并结果']}
    _require(result['输入版本'] == version, '角色卡输入版本过期')
    table, mappings = _freeze(episode, result['合并结果'], materials, version)
    _require(result['本集人物'] == table and result['临时人物映射'] == mappings, '冻结人物表/映射不一致')
    batch = load(episode / '声音记录.json')
    _require([r['句'] for r in result['对白归属']] == [r['id'] for r in batch], '最终归属必须按声音记录原顺序全覆盖')
    _validate_attribution({k: result[k] for k in ('对白归属', '补充建议', '待核对')}, batch, table, version,
                          {s['id'] for s in load(episode / '镜头表.json')['镜头']})
    return materials


def establish_rolecard(episode, variables, output, *, timeout, call, validator, prompt_file,
                       request_budget=None, resume=False):
    """仅发布通过总体校验的最终结果；不修改人物登记、归属或状态。"""
    success, version, records = False, None, []
    started = time.monotonic()
    try:
        materials = _materials(episode)
        batches = sentence_batches(episode)
        version = source_version(episode, materials, prompt_file=prompt_file)
        target_version = _apply_target_version(episode)
        merge = {'分组': []}
        if materials['临时人物']:
            package = _package(episode, '人物合并', {'候选人物': materials['临时人物'],
                       '已确认角色': version['已确认角色'], '身份证据字幕': materials['身份证据字幕']},
                       materials['images'], materials['labels'], prompt_file=prompt_file)
            merge = _run_stage(episode, package, lambda r: _freeze(episode, r, materials, version), output,
                               timeout=timeout, call=call, validator=validator, records=records,
                               request_budget=request_budget, resume=resume)
        version = {**version, '冻结合并': deepcopy(merge)}
        table, mappings = _freeze(episode, merge, materials, version)
        packages = [_attribution_package(episode, segment, batch, table, mappings, version, prompt_file=prompt_file)
                    for segment, batch in batches]

        def run(item):
            package, (_, batch) = item
            shots = {s['id'] for s in load(episode / '镜头表.json')['镜头']
                     if any(time_seconds(s['起']) < time_seconds(b['止']) and time_seconds(s['止']) > time_seconds(b['起']) for b in batch)}
            return _run_stage(episode, package, lambda r: _validate_attribution(r, batch, table, version, shots),
                              output, timeout=timeout, call=call, validator=validator, records=records,
                              request_budget=request_budget, resume=resume)

        with ThreadPoolExecutor(max_workers=3) as pool:
            parts = list(pool.map(run, zip(packages, batches)))
        by_sentence = {r['句']: r for p in parts for r in p['对白归属']}
        result = {'结果版本': 2, '模式': '建立', '输入版本': version, '覆盖目标版本': target_version, '合并结果': merge,
                  '本集人物': table, '临时人物映射': mappings,
                  '对白归属': [by_sentence[s['id']] for s in load(episode / '声音记录.json')],
                  '补充建议': [r for p in parts for r in p['补充建议']], '待核对': [r for p in parts for r in p['待核对']]}
        validate_two_stage_result(episode, result, prompt_file=prompt_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.new')
        temporary.write_text(dumps(result) + '\n', encoding='utf-8')
        temporary.replace(output)
        success = True
        return {'通过': True, '结果文件': str(output), '结果版本': 2, '批次数': len(parts)}
    finally:
        with (episode / '派发日志.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'时间': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                                     '角色': '角色卡', '模式': '建立', '结果版本': 2, '成功': success,
                                     '耗时秒': round(time.monotonic() - started, 3), '输入版本': version,
                                     '模型': version['模型']['model'] if version else None,
                                     '尝试记录': records, '结果文件': str(output)}, ensure_ascii=False) + '\n')


def apply_rolecard_result(episode, result, approvals, *, publish=None, defer_unreviewed=False):
    """主控显式审核后落板；识别/校验失败零落板，逐文件替换，状态最后写。"""
    episode = Path(episode).resolve()
    result = load(Path(result)) if isinstance(result, (str, Path)) else deepcopy(result)
    approvals = load(Path(approvals)) if isinstance(approvals, (str, Path)) else deepcopy(approvals)
    validate_two_stage_result(episode, result)
    deferred = approvals is None and defer_unreviewed
    if deferred:
        # 明确延期审核，不构造获批身份；N和同句阻断全部保留。
        approvals = {'结果版本': 2, '输入版本': result['输入版本'], '批准': []}
    _require(isinstance(approvals, dict) and set(approvals) == {'结果版本', '输入版本', '批准'}, '审核必须显式提供版本、输入版本和批准数组')
    _require(approvals['结果版本'] == 2 and approvals['输入版本'] == result['输入版本'], '审核输入版本过期')
    _require(isinstance(approvals['批准'], list), '批准必须为数组')
    table = deepcopy(result['本集人物'])
    people = {p['id']: p for p in table['人物']}
    roles = load(episode.parent / '角色卡.json')
    _schema('角色卡', roles)
    known = {r['id'] for r in roles if r['状态'] == '已确认'}
    next_id = max((int(r['id'][1:]) for r in roles), default=0) + 1
    approved, targets, new_ids = {}, set(), []
    for decision in approvals['批准']:
        _require(isinstance(decision, dict) and {'人物', '依据'} <= decision.keys()
                 and set(decision) <= {'人物', '依据', '已有角色'}, '审核条目字段不合法')
        ident = decision['人物']
        _require(ident in people and ident not in approved, '审核人物未登记或重复')
        _require(isinstance(decision['依据'], str) and decision['依据'].strip(), '审核依据不能为空')
        target = decision.get('已有角色')
        _require(target is None or target in known, '审核只能指向已确认C')
        person = people[ident]
        if target is not None:
            _require(target not in {m['映射'] for m in result['临时人物映射'] if m['映射'].startswith('C')},
                     '审核不能与已采用C的冻结组重合')
        if target is None:
            target = f'C{next_id:03d}'
            next_id += 1
            new_ids.append(target)
            shots = [s['id'] for s in load(episode / '镜头表.json')['镜头']
                     if any(s['id'] in t['出现镜头'] for t in result['输入版本']['临时声明'] if t['id'] in person['成员'])]
            roles.append({'id': target, '名称': person['名称'], '名称类型': person['名称类型'], '状态': '已确认',
                          '别名称呼': [], '外观': [{'集': int(episode.name[1:]), '描述': person['外观'], '参考帧': person['参考帧']}],
                          '声线': [], '关系': [], '身份依据': [person['依据'], decision['依据']],
                          '性格总结': [], '首次出场': shots[0], '待核对': []})
        _require(target not in targets, '审核不能把两个冻结组直接合并到同一C')
        targets.add(target)
        approved[ident] = target
        person.update({'审核C': target, '审核依据': decision['依据'], '状态': '已审核'})
    _require(not name_errors(roles), '审核后全剧名称重复或不合法')
    _schema('角色卡', roles)
    temporary = load(episode / '临时人物.json')
    mappings = {r['临时']: r for r in result['临时人物映射']}
    for person in temporary:
        mapping = mappings[person['id']]
        person.update({'映射': approved.get(mapping['映射'], mapping['映射']), '映射依据': mapping['依据']})
    attribution = [{**r, '说话人': approved.get(r['说话人'], r['说话人'])} for r in result['对白归属']]
    pending = load(episode / '待核对.json')
    issues = load(episode / '问题清单.json')
    number = max((int(r['id'].rsplit('Q', 1)[1]) for r in [*pending, *issues]), default=0)

    def add_pending(position, problem, related, *, level='阻断', need='主控核对身份证据或回听拆句'):
        nonlocal number
        number += 1
        pending.append({'id': f'{episode.name}-Q{number:03d}', '来源': '归属', '位置': position,
                        '原文': next((s['文本'] for s in load(episode / '声音记录.json') if s['id'] == position.get('句')), ''),
                        '类型': '声音与画面归属' if '句' in position else '角色卡资料', '级别': level,
                        '相关人物': related, '依据': problem, '修正要求': need, '处理职责': '角色卡',
                        '状态': '待核对', '复核结果': None})

    for row in result['待核对']:
        add_pending(row['位置'], row['问题'], [], level=row['级别'], need=row.get('需要') or '主控核对原始证据')
    for person in table['人物']:
        if person['审核C'] is None:
            member = next(t for t in temporary if t['id'] == person['成员'][0])
            add_pending({'镜头': member['出现镜头'][0]}, '本集人物未审核；' + '；'.join(person['待核对']), [person['id']])
    for row in attribution:
        if row['说话人'] is None or row['说话人'] in people and row['说话人'] not in approved:
            if not any(p['位置'].get('句') == row['句'] and p['级别'] == '阻断'
                       and p['状态'] not in ('已修正', '已裁定') for p in pending):
                add_pending({'句': row['句']}, '说话人身份未审核或未知', [row['说话人']] if row['说话人'] else [])
    for action in load(episode / '动作记录.json'):
        actor = next((p['映射'] for p in temporary if p['id'] == action['执行者']), action['执行者'])
        if action['关键'] and actor in people and actor not in approved:
            add_pending({'镜头': action['镜头']}, '关键动作事实保留，但执行者身份尚未审核', [actor],
                        need='核对该动作人物身份；保留原动作及依据，不作为正式正文事实')
    state = load(episode / '状态.json')
    state['状态'] = '进行中'
    state['步骤']['3'] = True
    for step in range(4, 10):
        state['步骤'][str(step)] = False
    state['本集新增人物'] = list(dict.fromkeys([*state['本集新增人物'], *new_ids]))
    planned = {'本集人物': table, '临时人物': temporary, '归属表': attribution, '待核对': pending, '状态': state}
    # 校验拟写文件及其相互引用；临时目录不是可消费的黑板结果。
    with tempfile.TemporaryDirectory(prefix='s2s-rolecard-') as folder:
        context = Path(folder) / episode.name
        context.mkdir()
        for name in ('镜头表.json', '片段表.json', '声音记录.json', '字幕记录.json', '动作记录.json',
                     '问题清单.json', '对照表.json', '工作稿.md'):
            path = episode / name
            if path.is_file():
                (context / name).symlink_to(path)
        # 全剧角色卡留在集目录外，保留跨集首次出场/声线引用的真实语义。
        (Path(folder) / '角色卡.json').write_text(dumps(roles), encoding='utf-8')
        for other in episode.parent.glob('E[0-9]*'):
            if other.is_dir() and other.name != episode.name:
                (Path(folder) / other.name).symlink_to(other)
        for kind, value in planned.items():
            (context / f'{kind}.json').write_text(dumps(value), encoding='utf-8')
        for kind in ('角色卡', *planned):
            target = (Path(folder) if kind == '角色卡' else context) / f'{kind}.json'
            checked = validate_file(target, kind)
            _require(checked['通过'], '拟写黑板校验失败：' + dumps(checked['错误']))
    validate_two_stage_result(episode, result)
    writes = [(episode.parent / '角色卡.json', roles),
              *[(episode / f'{kind}.json', value) for kind, value in planned.items()]]
    receipt = {'通过': True, '本集人物': len(people), '获批映射': approved, '本集新增人物': new_ids}
    if deferred:
        receipt['审核状态'] = '未提供审核；没有批准任何本集人物'
    if publish is not None:
        publish(writes, receipt)
        return receipt
    for path, value in writes:
        temporary_path = path.with_name(path.name + '.new')
        temporary_path.write_text(dumps(value) + '\n', encoding='utf-8')
        temporary_path.replace(path)
    return receipt
