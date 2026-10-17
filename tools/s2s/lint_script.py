"""逐行检查剧本描述，不检查对白冒号后的台词用词。"""

import argparse
import json
from pathlib import Path
import re

from s2s.validate import read_json
from s2s.names import invalid_name, name_errors

DEFAULT_WORDS = Path(__file__).with_name('schemas') / 'lint_words.json'
TITLE_RE = re.compile(r'^(?:#{1,6}\s*)?第([1-9][0-9]*)集$')
SCENE_RE = re.compile(r'^场景([1-9][0-9]*)-([1-9][0-9]*)：[^，,：:]+，[^，,：:]+，(内|外)(?:（闪回）)?$')
DIALOGUE_RE = re.compile(r'^([^：:（）()]+?)(?:（([^（）]*)）|\(([^()]*)\))?[：:](.*)$')
PRONOUN_RE = re.compile(r'他们|她们|(?<![其吉])他|她')
VAGUE_RE = re.compile(r'另一个人|其中一人|另一人|对方|前者|后者')
OFFSCREEN_RE = re.compile(r'^(画外|心声|电话|视频|广播)')
FORBIDDEN_STYLE = {
    '文学化表达': ('眼中闪过一丝', '嘴角勾起一抹', '空气仿佛凝固', '意味深长', '下意识地',
                '若有所思', '缓缓地', '不禁', '无声地', '显然', '看得出来'),
    '镜头语言': ('特写', '近景', '远景', '俯拍', '镜头推近', '镜头拉远', '镜头跟随', '镜头扫过',
               '慢镜头', '画面切到', '画面切向', '镜头一转', '画面一黑', '白光一闪',
               '画面定格', '淡入', '淡出', '画面中', '画面里', '镜头前'),
}
PENDING_SPEAKER_RE = re.compile(r'^【说话人待核对 E[0-9]{2,}-Q[0-9]{3,}】$')
PENDING_NOTE_RE = re.compile(r'【待核对 (E[0-9]{2,}-Q[0-9]{3,})】')
FINAL_PATTERNS = (re.compile(r'【(?:说话人)?待核对[^】]*'), re.compile(r'[0-9]{2}:[0-9]{2}'),
                  re.compile(r'E[0-9]{2,}-[SKAQP][0-9]*'))


def parse_dialogue(line: str) -> tuple[str, str, str] | None:
    line = line.strip()
    if line.startswith(('△', '人物', '场景')) or TITLE_RE.fullmatch(line):
        return None
    match = DIALOGUE_RE.fullmatch(line)
    if not match:
        return None
    return match[1].strip(), match[2] or match[3] or '', match[4]


def line_type(line: str) -> str:
    line = line.strip()
    if not line:
        return '空行'
    if TITLE_RE.fullmatch(line):
        return '集标题'
    if line.startswith('场景'):
        return '场景行'
    if line.startswith('人物：'):
        return '人物行'
    if line.startswith('△'):
        return '△'
    return '对白' if parse_dialogue(line) else '未知'


def _names(line: str) -> list[str]:
    return [name.strip() for name in re.split(r'[、，,]', line.split('：', 1)[1]) if name.strip()]


def _mask_names(text: str, names: set[str]) -> str:
    for name in sorted(names, key=len, reverse=True):
        text = text.replace(name, '\0' * len(name))
    # 书名号内是照录的作品名称，不属于人物指代。
    text = re.sub(r'《[^》]*》', lambda m: '\0' * len(m[0]), text)
    return text


def lint_text(text: str, roles: list[dict] | None = None, *, formal: bool = False,
              words: dict | None = None) -> dict:
    words = read_json(DEFAULT_WORDS) if words is None else words
    lines = text.splitlines()
    known = {role['名称'] for role in roles} if roles is not None else None
    # 未提供角色卡时，仅用人物行中的具体名称保护描述名。
    protected = {name for name in known if not invalid_name(name)} if known is not None else {
        name for line in lines if line.strip().startswith('人物：') for name in _names(line.strip())
        if not invalid_name(name) and name not in words['泛称']}
    aliases = {alias for role in roles or [] for alias in role.get('别名称呼', [])
               if alias and alias not in (known or set()) and not invalid_name(alias)}
    errors = [{'行': 0, '原文': roles[index]['名称'], '检查项': '角色卡名称', '命中词': message}
              for index, message in name_errors(roles or [])]

    def fail(i, rule, token):
        errors.append({'行': i + 1, '原文': lines[i], '检查项': rule, '命中词': token})

    episode = None
    expected_scene = 1
    for i, raw in enumerate(lines):
        line = raw.strip()
        kind = line_type(line)
        if formal:
            for pattern in FINAL_PATTERNS:
                for match in pattern.finditer(line):
                    fail(i, '正式稿残留', match[0])
        if kind == '集标题':
            episode = int(TITLE_RE.fullmatch(line)[1])
            expected_scene = 1
        elif kind == '场景行':
            match = SCENE_RE.fullmatch(line)
            if match is None:
                fail(i, '场景行格式', line)
            else:
                scene_ep, serial = int(match[1]), int(match[2])
                if episode is None:
                    episode = scene_ep
                if scene_ep != episode:
                    fail(i, '场景集号', match[1])
                if serial != expected_scene:
                    fail(i, '场景编号连续', match[2])
                expected_scene = serial + 1
            if i + 1 >= len(lines) or not lines[i + 1].strip().startswith('人物：'):
                fail(i, '场景后人物行', line)
        elif kind == '人物行':
            for name in _names(line):
                if invalid_name(name):
                    fail(i, '非法人物名称', name)
                elif known is not None and name not in known:
                    fail(i, '角色卡名称', name)
        elif kind == '未知':
            fail(i, '行类型', line)

        description = None
        if kind == '△':
            description = line[1:].strip()
            if description.startswith('画面字幕：'):
                description = None
        elif kind == '对白':
            speaker, parenthesis, _ = parse_dialogue(line)
            if invalid_name(speaker):
                fail(i, '非法人物名称', speaker)
            elif known is not None and speaker not in known and not PENDING_SPEAKER_RE.fullmatch(speaker):
                fail(i, '角色卡名称', speaker)
            description = parenthesis
            if OFFSCREEN_RE.match(parenthesis):
                masked = _mask_names(parenthesis, protected)
                for word in words['动作词']:
                    if word in masked:
                        fail(i, '画外形态动作', word)
        if description:
            description = PENDING_NOTE_RE.sub('', description)
            masked = _mask_names(description, protected)
            for alias in sorted(aliases, key=len, reverse=True):
                if alias in masked:
                    fail(i, '角色别称漂移', alias)
            for rule, expressions in FORBIDDEN_STYLE.items():
                for expression in expressions:
                    if expression in masked:
                        fail(i, rule, expression)
            for match in PRONOUN_RE.finditer(masked):
                fail(i, '人物代词', match[0])
            # “它们”还可指物，只有明确的人物先行词时才报出。
            for match in re.finditer('它们', masked):
                prefix = description[:match.start()]
                people = [prefix.rfind(name) for name in protected]
                people += [prefix.rfind(word) for word in ('众人', '人群', '人物', '演员')]
                objects = [prefix.rfind(word) for word in ('物品', '杯子', '文件', '动物', '猫', '狗', '桌椅', '椅子')]
                if max(people, default=-1) > max(objects, default=-1):
                    fail(i, '人物代词', '它们')
            # 长词先匹配，避免“小男孩”同时报成“男孩”。
            generic = re.compile('|'.join(re.escape(w) for w in sorted(words['泛称'], key=len, reverse=True))) if words['泛称'] else None
            if generic:
                for match in generic.finditer(masked):
                    fail(i, '人物泛称', match[0])
            for match in VAGUE_RE.finditer(masked):
                fail(i, '含糊指代', match[0])
    return {'通过': not errors, '错误': errors}


def lint_file(file: str | Path, role_card: str | Path | None = None, *, formal: bool = False,
              word_file: str | Path | None = None) -> dict:
    result = {'文件': str(file), '通过': False, '错误': []}
    try:
        roles = read_json(role_card) if role_card is not None else None
        if roles is not None and (not isinstance(roles, list) or any(
                not isinstance(r, dict) or not isinstance(r.get('名称'), str) for r in roles)):
            raise ValueError('角色卡须为含名称字段的条目数组')
        words = read_json(word_file or DEFAULT_WORDS)
        if not isinstance(words, dict) or any(
                not isinstance(words.get(k), list) or any(not isinstance(w, str) or not w for w in words[k])
                for k in ('泛称', '动作词')):
            raise ValueError('词表须包含泛称、动作词两个字符串数组')
        result.update(lint_text(Path(file).read_text(encoding='utf-8'), roles, formal=formal, words=words))
    except (OSError, ValueError) as exc:
        result['错误'].append({'行': 0, '原文': '', '检查项': '输入文件', '命中词': str(exc)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='检查剧本格式和非对白描述')
    parser.add_argument('文件')
    parser.add_argument('--角色卡')
    parser.add_argument('--正式稿', action='store_true')
    parser.add_argument('--词表', help='自定义泛称与动作词 JSON 文件')
    args = parser.parse_args()
    result = lint_file(args.文件, args.角色卡, formal=args.正式稿, word_file=args.词表)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['通过'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
