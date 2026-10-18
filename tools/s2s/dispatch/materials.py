"""按提示词的资料隔离表装配数据，不把原始黑板路径开放给子 agent。"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

from s2s.config import PROJECT_ROOT
from s2s.evidence import evidence_problem
from s2s.segments import per_second
from s2s.validate import time_seconds, validate_file, episode_people_errors

PROMPTS = {'全剧主控': '01-全剧主控.md', '单集主控': '02-单集主控.md',
           '角色卡': '03-角色卡.md', '动作识别': '04-动作识别.md',
           '成稿': '05-成稿.md', '验收': '06-验收.md'}
# 文件名后缀省略；“投影”表示只传下列代码提取后的字段，绝不附原文件。
MATERIALS = {
    '动作识别': ('镜头表:片段及衔接投影', '片段表:本段投影', '声音记录:仅起止',
               '角色卡:已确认外观', '字幕记录:非台词画面文字', '帧:白名单'),
    '角色卡': ('角色卡:已确认', '临时人物:候选/冻结投影', '本集人物',
              '字幕记录:仅人物字卡/地点', '声音及口型声纹:仅归属核对', '帧:阶段授权'),
    '成稿': ('镜头表:不含帧路径', '动作记录:已确认', '归属表:已确认', '声音记录',
           '字幕记录', '角色卡', '待核对', '问题清单:定向改写', '工作稿:定向改写', '对照表:定向改写'),
    '验收': ('工作稿', '对照表', '角色卡', '镜头表', '声音记录', '字幕记录',
           '问题清单:仅复核', '跨集:人物行与首尾场景及结尾衔接', '帧:白名单'),
}
# 子角色只附本角色用到的数据约定章节；主控仍读全文。
CONVENTION = {
    '动作识别': {'1', '4.1', '4.3', '4.4', '4.6', '4.9', '4.10', '4.11', '4.12'},
    '角色卡': {'1', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.9', '4.12', '4.13', '4.15', '4.16'},
    '成稿': {'1', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.9'},
    '验收': {'1', '4.1', '4.2', '4.3', '4.6', '4.7', '4.8', '4.9', '4.10', '4.11'},
}
MODES = {'动作识别': ('片段识别', '定向复看'), '角色卡': ('建立', '定向核对', '总结'),
         '成稿': ('全集成稿', '定向改写'), '验收': ('全文审查', '复核', '跨集')}
PLACEHOLDER = re.compile(r'{{\s*(.*?)\s*}}')


def render(template: str, variables: dict) -> str:
    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(template)} - variables.keys())
    if missing:
        raise ValueError('未替换占位：' + '、'.join(missing))
    result = PLACEHOLDER.sub(lambda m: str(variables[m.group(1)]), template)
    if PLACEHOLDER.search(result):
        raise ValueError('替换结果仍含占位')
    return result


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def dumps(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def convention_for(role: str, text: str) -> str:
    parts = re.split(r'(?m)^(?=#{2,3} \d)', text)
    kept = [parts[0]]
    for part in parts[1:]:
        number = re.match(r'#{2,3} (\d+(?:\.\d+)*)', part).group(1)
        if number == '4' or number in CONVENTION[role]:
            kept.append(part)
    return ''.join(kept).strip()


def recheck_shots(mapping: list, shots: list, issues: list, scope: str) -> list:
    """复核只开放已修正问题及范围所在场景、前后相邻场景的镜头。"""
    order = list(dict.fromkeys(row['场景'] for row in mapping))
    by_line = {row['行']: row['场景'] for row in mapping}
    scenes = set(re.findall(r'场景\d+-\d+', scope))
    for start, end in re.findall(r'第\s*(\d+)\s*(?:至|到|-)\s*(\d+)\s*行', scope):
        scenes.update(by_line[n] for n in range(int(start), int(end) + 1) if n in by_line)
    scenes.update(by_line[int(n)] for n in re.findall(r'第\s*(\d+)\s*行', scope) if int(n) in by_line)
    for issue in issues:
        if issue.get('状态') != '已修正':
            continue
        position = issue.get('位置', {})
        if position.get('场景'):
            scenes.add(position['场景'])
        if position.get('行') in by_line:
            scenes.add(by_line[position['行']])
        scenes.update(row['场景'] for row in mapping if position.get('镜头') in row['镜头'])
    wanted = set()
    for index, scene in enumerate(order):
        if scene in scenes:
            wanted.update(order[max(index - 1, 0):index + 2])
    ids = {shot for row in mapping if row['场景'] in wanted for shot in row['镜头']}
    if not ids:
        raise ValueError('复核未能从问题清单和范围定位到场景，无法确定需要核对的帧')
    return [shot for shot in shots if shot['id'] in ids]


def frame_path(folder: Path, filename: str) -> Path:
    path = (folder / filename).resolve()
    if path.suffix != '.jpg' or not path.is_relative_to(folder.resolve()):
        raise ValueError('帧图片路径越界或扩展名错误：' + filename)
    if not path.is_file():
        raise ValueError(f'图片不存在：{path}')
    return path


@dataclass
class Package:
    prompt: str
    files: list[Path] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)
    # 与 images 一一对应的“文件名｜镜头｜时间”标注；view_all 时帧随请求一次送入并须全部查看。
    labels: list[str] = field(default_factory=list)
    view_all: bool = False
    # 同一角色各次调用不变的部分（数据约定与返回格式），放在系统提示中以便命中输入缓存。
    system: str = ''
    segment: dict | None = None
    # 动作识别实际送出的镜头及帧，校验依据帧时以此为准。
    shots: list[dict] = field(default_factory=list)
    mode: str = ''
    scope: str = ''
    dense: list[dict] = field(default_factory=list)
    require_placement: bool = False

    def signature(self):
        return {'模式': self.mode, '范围': self.scope, '提示词': self.prompt, '系统': self.system,
                '文件': [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns]
                       for p in [*self.files, *self.images]]}


def local_people_projection(episode):
    path = episode / '本集人物.json'
    if not path.is_file():
        return []  # 旧无N黑板兼容；N引用仍由validate硬核验。
    table = load(path)
    errors = episode_people_errors(episode, table)
    if errors:
        raise ValueError('本集人物登记无效：' + '; '.join(errors))
    return [{k: row[k] for k in ('id', '名称', '名称类型', '状态', '审核C')} for row in table['人物']]


def assemble(role: str, episode: Path, variables: dict, mode=None, scope=None, *, prompt_file=None, review_stage=None) -> Package:
    if role not in MODES:
        raise ValueError('只能派发子角色：' + '、'.join(MODES))
    if not re.fullmatch(r'E[0-9]{2,}', episode.name):
        raise ValueError('本集目录名须为 E01 这样的集号')
    mode = mode or variables.get('模式') or MODES[role][0]
    if mode not in MODES[role]:
        raise ValueError(f'{role} 模式无效：{mode}')
    scope = scope if scope is not None else variables.get('范围', '')
    if role == '角色卡' and mode == '建立':
        from .rolecard import _materials, _package, _confirmed
        materials = _materials(episode)
        return _package(episode, '人物合并', {'候选人物': materials['临时人物'], '已确认角色': _confirmed(episode),
                        '身份证据字幕': materials['身份证据字幕']}, materials['images'], materials['labels'], prompt_file=prompt_file)
    packet = Package('', mode=mode, scope=str(scope))
    packet.require_placement = role == '成稿' and variables.get('固定动作排布') is True
    data = {}

    def read(name, *, series=False, optional=False):
        path = (episode.parent if series else episode) / name
        if optional and not path.exists():
            return []
        packet.files.append(path)
        return path.read_text(encoding='utf-8') if path.suffix == '.md' else load(path)

    roles = read('角色卡.json', series=True, optional=True)
    shots = read('镜头表.json')
    values = {**variables, '剧名': episode.parent.name, '集号': episode.name[1:],
              '上集结尾衔接': variables.get('上集结尾衔接', ''),
              '建立 | 定向核对 | 总结': mode, '片段识别 | 定向复看': mode,
              '全集成稿 | 定向改写': mode, '全文审查 | 复核 | 跨集': mode,
              '定向核对时填写问题编号、句号和镜头范围': scope,
              '问题编号、镜头和需要确认的内容，片段识别时为空': scope,
              '问题编号、行号和修正要求，全集成稿时为空': scope,
              '全文，或第 N 至 M 行 / 场景 X 至 Y': scope or '全文',
              '前一段的最后一个场景，分段审查时提供': variables.get('衔接参考', '')}
    if mode in ('定向复看', '定向核对', '定向改写', '复核') and not scope:
        raise ValueError('定向任务或复核必须填写 --范围 或变量“范围”')
    selected_shots = shots['镜头']
    packet.view_all = role == '动作识别' or (role == '验收' and (review_stage == '画面' or mode in ('复核', '跨集')))
    if role == '动作识别' or (role == '验收' and review_stage == '画面'):
        segments = read('片段表.json')['片段']
        segment_id = variables.get('片段号') or (scope if any(s['id'] == scope for s in segments) else None)
        selected = [s for s in segments if s['id'] == segment_id]
        if len(selected) != 1:
            raise ValueError('动作识别需要有效变量“片段号”')
        segment = packet.segment = selected[0]
        wanted = {*segment['镜头'], segment['衔接前'], segment['衔接后']}
        selected_shots = [k for k in shots['镜头'] if k['id'] in wanted]
        if role == '动作识别':
            selected_shots = packet.shots = [{**k, '帧': per_second(k)} for k in selected_shots]
        values.update({'片段号': segment['id'], '起始镜头': segment['镜头'][0],
                       '结束镜头': segment['镜头'][-1], '起': segment['起'], '止': segment['止']})
        data['片段'] = segment
        data['镜头表'] = selected_shots
    if role == '动作识别':
        segment = packet.segment
        data['人声时间段'] = [{k: s[k] for k in ('起', '止')}
                            for s in read('声音记录.json', optional=True)
                            if time_seconds(s['起']) < time_seconds(segment['止'])
                            and time_seconds(s['止']) > time_seconds(segment['起'])]
        data['已确认人物外观'] = [{k: r[k] for k in ('id', '名称', '外观')}
                               for r in roles if r['状态'] == '已确认']
        data['画面文字'] = [s for s in read('字幕记录.json', optional=True)
                         if s['类型'] != '台词字幕' and time_seconds(s['起']) < time_seconds(selected_shots[-1]['止'])
                         and time_seconds(s['止']) > time_seconds(selected_shots[0]['起'])]
    else:
        if role == '角色卡' and mode == '总结':
            data = {'角色卡': [r for r in roles if r['状态'] == '已确认'],
                    '确认事件': [r for r in read('动作记录.json') if r['确定度'] == '确认']}
            selected_shots = []
        else:
            data.update({'角色卡': roles, '镜头表': shots,
                         '声音记录': read('声音记录.json'), '字幕记录': read('字幕记录.json')})
        if role in ('角色卡', '成稿', '验收'):
            data['本集人物'] = local_people_projection(episode)
            if (episode / '本集人物.json').is_file():
                packet.files.append(episode / '本集人物.json')
        if role == '角色卡':
            data['角色卡'] = [r for r in roles if r['状态'] == '已确认']
            if '字幕记录' in data:
                data['字幕记录'] = [r for r in data['字幕记录'] if r['类型'] in ('人物字卡', '地点字幕')]
            data['临时人物'] = read('临时人物.json', optional=True)
            data['冻结人物表约束'] = '绑定当前登记和映射；只核对归属或提出建议，禁止重分组、造N/C或改变映射。'
            data['证据状态'] = {}
            for name in (() if mode == '总结' else ('口型记录', '声纹聚类', '声纹候选')):
                path = episode / (name + '.json')
                if not path.is_file():
                    data['证据状态'][name] = {'状态': '缺失', '说明': '本次没有该项证据，不能据此作否定判断'}
                    continue
                value = read(name + '.json')
                problem = evidence_problem(name, value, data['声音记录'])
                if problem:
                    data['证据状态'][name] = {'状态': '过期', '说明': problem}
                    continue
                data['证据状态'][name] = {'状态': '有效'}
                if name == '口型记录':
                    current = {row['id']: row['声线'] for row in data['声音记录']}
                    if any(row.get('声线') != current[row['句']] for row in value['句']):
                        from s2s.lipsync.core import pair_voices
                        rows = [{**row, '声线': current[row['句']]} for row in value['句']]
                        value = {**value, '句': rows, '声线配对': pair_voices(rows)}
                    value = {
                        '句': [{**{k: v for k, v in row.items() if k != '轨迹'},
                                '轨迹': [{k: v for k, v in track.items() if k not in ('曲线', '模型曲线')}
                                         for track in row.get('轨迹', [])]}
                               for row in value['句']],
                        '声线配对': value.get('声线配对', [])}
                data[name] = value
            # 片段原始响应保留口型；合并动作记录本身没有口型字段。
            people = []
            for path in ([] if mode == '总结' else sorted((episode / '派发结果' / '动作识别').glob('*.json'))):
                result = load(path)
                packet.files.append(path)
                for shot in result.get('镜头', []):
                    people.append({k: shot[k] for k in ('镜头', '画面人物', '口型')})
            if not people and mode != '总结':
                raise ValueError('缺少派发结果/动作识别/*.json，无法提供人物与口型信息')
            data['人物与口型'] = people
        elif role == '成稿':
            data['镜头表'] = {**shots, '镜头': [{k: v for k, v in shot.items() if k != '帧'} for shot in shots['镜头']]}
            data['动作记录'] = [r for r in read('动作记录.json') if r['确定度'] == '确认']
            data['归属表'] = [r for r in read('归属表.json') if r['确定度'] == '确认']
            data['待核对'] = read('待核对.json')
            if mode == '定向改写':
                for name in ('工作稿.md', '对照表.json', '问题清单.json'):
                    data[Path(name).stem] = read(name)
        elif role == '验收':
            data['工作稿'] = read('工作稿.md')
            data['对照表'] = read('对照表.json')
            if review_stage == '文本':
                selected_shots = []
                data['镜头表'] = {**shots, '镜头': [{k: v for k, v in shot.items() if k != '帧'} for shot in shots['镜头']]}
            elif review_stage == '画面':
                segment = packet.segment
                ids = {shot['id'] for shot in selected_shots}
                mapping = data['对照表']
                lines = data['工作稿'].splitlines()
                wanted_lines = {row['行'] for row in mapping if ids.intersection(row['镜头'])}
                scenes = {row['场景'] for row in mapping if row['行'] in wanted_lines}
                wanted_lines.update(row['行'] for row in mapping if row['场景'] in scenes and row['类型'] in ('场景行', '人物行'))
                # 保留相邻行以核对跨段动作、对白，不改变原稿行号。
                wanted_lines = {n + delta for n in wanted_lines for delta in (-1, 0, 1) if 1 <= n + delta <= len(lines)}
                data['工作稿'] = [{'行': n, '正文': lines[n - 1]} for n in sorted(wanted_lines)]
                data['对照表'] = [row for row in mapping if row['行'] in wanted_lines]
                data['镜头表'] = {**shots, '镜头': selected_shots}
                data['声音记录'] = [row for row in data['声音记录'] if time_seconds(row['起']) < time_seconds(selected_shots[-1]['止']) and time_seconds(row['止']) > time_seconds(selected_shots[0]['起'])]
                data['字幕记录'] = [row for row in data['字幕记录'] if time_seconds(row['起']) < time_seconds(selected_shots[-1]['止']) and time_seconds(row['止']) > time_seconds(selected_shots[0]['起'])]
            if mode == '复核':
                data['问题清单'] = read('问题清单.json')
                data['改动对照'] = variables.get('改动对照', [])
                selected_shots = recheck_shots(data['对照表'], shots['镜头'], data['问题清单'], scope)
            if mode == '跨集':
                cross = []
                for number in variables.get('跨集集号', []):
                    other = episode.parent / f'E{int(number):02d}'
                    script = other / '正式稿.md'
                    state = other / '状态.json'
                    packet.files.extend((script, state))
                    text = script.read_text(encoding='utf-8')
                    scenes = re.split(r'(?=^场景\d+-\d+：)', text, flags=re.M)[1:]
                    cross.append({'集': int(number), '人物行': re.findall(r'^人物：.*$', text, re.M),
                                  '首尾场景': list(dict.fromkeys(scenes[:2] + scenes[-2:])),
                                  '结尾衔接': load(state)['结尾衔接']})
                    # 仅开放首尾场景对照表引用的帧。
                    mapping_path = other / '对照表.json'
                    shot_path = other / '镜头表.json'
                    packet.files.extend((mapping_path, shot_path))
                    scenes_used = set(re.findall(r'^场景\d+-\d+', '\n'.join(cross[-1]['首尾场景']), re.M))
                    ids = {ident for row in load(mapping_path) if row['场景'] in scenes_used for ident in row['镜头']}
                    for shot in load(shot_path)['镜头']:
                        if shot['id'] in ids:
                            for f in shot['帧']:
                                packet.images.append(frame_path(other / '素材/frames', f['文件']))
                                packet.labels.append(f"{f['文件']}｜镜头 {shot['id']}｜{f['时间']}")
                if not cross:
                    raise ValueError('跨集模式需要变量“跨集集号”数组')
                data = {'角色卡': roles, '跨集资料': cross}
                selected_shots = []
    if role != '成稿':
        bridges = {packet.segment['衔接前'], packet.segment['衔接后']} if packet.segment else set()
        for shot in selected_shots:
            for f in shot['帧']:
                packet.images.append(frame_path(episode / '素材/frames', f['文件']))
                packet.labels.append(f"{f['文件']}｜镜头 {shot['id']}｜{f['时间']}" + ('｜衔接参考' if shot['id'] in bridges else ''))
        for filename in variables.get('加密帧索引', []):
            index = (episode / filename).resolve()
            if not index.is_relative_to((episode / '素材/frames_dense').resolve()):
                raise ValueError('加密帧索引必须位于本集素材/frames_dense 内')
            check = validate_file(index, '加密帧索引')
            if not check['通过']:
                raise ValueError('加密帧索引不合格：' + dumps(check['错误']))
            dense = load(index)
            packet.files.append(index)
            for frame in dense['帧']:
                belongs = next((shot['id'] for shot in selected_shots
                                if time_seconds(shot['起']) <= time_seconds(frame['时间']) < time_seconds(shot['止'])), None)
                if belongs is None:
                    raise ValueError('加密帧时间不在本次授权镜头内')
                path = frame_path(index.parent, frame['文件'])
                packet.images.append(path)
                packet.labels.append(f"{frame['文件']}｜镜头 {belongs}｜{frame['时间']}｜加密帧")
                packet.dense.append({'镜头': belongs, '文件': frame['文件'], '时间': frame['时间'], '路径': str(path)})
        if packet.dense:
            data['加密帧'] = packet.dense
    template = Path(prompt_file).resolve() if prompt_file else PROJECT_ROOT / 'prompts' / PROMPTS[role]
    convention = PROJECT_ROOT / 'prompts/00-数据约定.md'
    packet.files.extend((template, convention))
    packet.prompt = render(template.read_text(encoding='utf-8'), values)
    packet.prompt += '\n\n## 本次授权资料\n' + dumps(data)
    if role == '验收' and review_stage:
        if review_stage == '文本':
            packet.prompt += '\n\n## 本次阶段：整集文本检查\n逐行检查全文规范、逐句反查全部声音记录及对白顺序；没有授权帧，不作画面已核对的声明。画面问题由后续逐段独立核对。'
        else:
            packet.prompt += '\n\n## 本次阶段：分段画面核对\n本段：' + dumps(packet.segment)
            packet.prompt += '\n查看全部附图，独立扫描本段人物与物件变化，再对照正文。只报告本段问题，衔接镜头只作参考。正文保留原行号；未写出的关键动作也必须报告。只对本段作结论，不代表整集通过。'
            packet.prompt += ('\n本阶段只报告需要看画面才能判断的问题：具体动作（无据、漏写、执行者或对象错误）、动作与对白关联、'
                              '声音与画面归属、画面人物与正文不符。人物名称与代词规范、格式、文学化与镜头语言、对白逐字等纯文本问题'
                              '已由整集文本检查覆盖，本阶段不重复报告。')
    if packet.view_all:
        packet.prompt += ('\n\n## 图片\n本次授权的全部帧附在本条消息末尾，每张图前一行标注文件名、所属镜头和时间；衔接参考帧和加密帧另有标注。'
                          '\n必须查看全部附图，再输出 JSON。' + ('衔接镜头不输出条目。' if packet.segment else ''))
    elif packet.images:
        packet.prompt += '\n\n## 图片白名单\n只能用 read 读取这些图片，不允许读取其他文件。\n'
        packet.prompt += '\n'.join(map(str, packet.images))
    if role == '成稿':
        packet.prompt += '\n返回一个 JSON 对象：工作稿为完整正文字符串，对照表为数组；定向改写另含改动对照数组。'
        if variables.get('固定动作排布'):
            packet.prompt += ('\n固定单集合同：同次返回动作排布数组，每个正文实际呈现动作恰好一项：'
                              '{动作:动作ID, 排布:{位置:对白括号或独立△, 句:实际关联句或null, 依据:原记录与时序依据, 表述:正文采用的完整表述}}。'
                              '不改变执行者、对象、事实和时间；未确认身份的动作留在待核对，不能写成正文事实。'
                              '已确认说话人但形态待核对时保留该姓名及身份证据，在行尾引用对应待核对Q；不要伪造词时间拆句。')
    else:
        packet.prompt += '\n只返回符合系统提示中返回包 Schema 的一个 JSON 对象，不附推理过程。'
    schema = load(Path(__file__).with_name('role_schemas.json'))[role]
    packet.files.append(Path(__file__).with_name('role_schemas.json'))
    packet.system = ('## 数据约定\n' + convention_for(role, convention.read_text(encoding='utf-8'))
                     + '\n\n## 返回包精确 Schema\n' + json.dumps(schema, ensure_ascii=False, separators=(',', ':')))
    if role == '动作识别':
        packet.system += '\n顶层仅允许：片段、模式、镜头、衔接、加密取帧请求、待核对。临时人物在镜头的画面人物中声明，不另设顶层临时人物字段。'
    if role == '角色卡':
        packet.system += '\n本次绑定当前冻结人物表；临时人物映射、新增人物必须为空。只引用已确认C或登记N，禁止重分组或分配编号。总结的对白归属也必须为空。'
    return packet
