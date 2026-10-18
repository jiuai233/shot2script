"""只读生成临时人物候选代表帧及送图容量清单，不派发、不改黑板。"""

from pathlib import Path
import re

from s2s.config import get_vision
from s2s.validate import time_seconds, validate_file
from .materials import frame_path, load


SELECTION_RULE = (
    '每人选一帧：优先使用同片段、本人出现镜头中显式画面人物依据帧；'
    '仅旧声明缺字段时回退本人动作/神态依据，坏字段不回退。'
    '池内按镜头起点、镜头表顺序、帧时间、帧文件名、动作先于神态、来源文件名及条目序号排序取首项。'
    '校验全部此类引用，不以整镜头首帧或其他人的依据补齐；不按清晰度择优。'
    '加密帧仅核验本集索引，不代表重证历史实际送图；实际送出由生产返回校验负责。'
)
VISUAL_STATUS = '未确认：未读图，候选关联不等于清晰可认人或视觉覆盖'
FRAME_FILENAME = re.compile(r'(K[0-9]{3,}_[0-9]{2}|t[0-9]{2}-[0-5][0-9]\.[0-9])\.jpg')


def _dense_frame(episode: Path, filename: str, video: str) -> tuple[dict, Path, list[Path]]:
    """只校验含所需文件名的本集索引，不因无关索引损坏而失败。"""
    root = episode / '素材/frames_dense'
    matches, indexes = {}, []
    for index in sorted(root.rglob('帧.json')):
        # 越界索引不读取，也不因无关链接使合法引用失败。
        if not index.resolve().is_relative_to(root.resolve()):
            continue
        try:
            dense = load(index)
        except (OSError, ValueError):
            continue
        if not isinstance(dense, dict) or not isinstance(dense.get('帧'), list):
            continue
        frames = [f for f in dense['帧'] if isinstance(f, dict) and f.get('文件') == filename]
        if not frames:
            continue
        if not validate_file(index, '加密帧索引')['通过']:
            raise ValueError('加密帧索引不合格：' + str(index))
        if dense['视频'] != video:
            raise ValueError('加密帧索引视频不一致：' + str(index))
        path = frame_path(index.parent, filename)
        if not path.is_relative_to(root.resolve()):
            raise ValueError('加密帧图片路径越界：' + filename)
        matches[path] = frames[0]
        indexes.append(index)
    if not matches:
        raise ValueError('非本镜头授权帧或缺少有效加密帧索引：' + filename)
    if len(matches) != 1:
        raise ValueError('加密帧同名多路径歧义：' + filename)
    path, frame = next(iter(matches.items()))
    return frame, path, indexes


def build_rolecard_materials(episode: Path, *, max_images: int | None = None) -> dict:
    """返回可复用的 JSON 材料；images 按真实路径去重，labels 与其一一对应。"""
    episode = Path(episode).resolve()
    limit = get_vision()['max_images_per_call'] if max_images is None else max_images
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError('图片上限须为正整数')
    people_path = episode / '临时人物.json'
    shots_path = episode / '镜头表.json'
    people = load(people_path)
    shot_table = load(shots_path)
    shots = shot_table['镜头']
    result_paths = sorted((episode / '派发结果/动作识别').glob('*.json'))
    if not result_paths:
        raise ValueError('缺少派发结果/动作识别/*.json；不自动生成素材')
    if not people:
        raise ValueError('临时人物清单为空')
    by_person = {p['id']: p for p in people}
    by_shot = {s['id']: (index, s) for index, s in enumerate(shots)}
    if len(by_person) != len(people):
        raise ValueError('临时人物 ID 重复')
    if len(by_shot) != len(shots):
        raise ValueError('镜头 ID 重复')
    for person in people:
        if any(shot not in by_shot for shot in person['出现镜头']):
            raise ValueError('人物出现镜头不在镜头表中：' + person['id'])
    candidates = {p['id']: [] for p in people}
    frames_root = episode / '素材/frames'
    dense_cache, used_indexes = {}, set()

    for source in result_paths:
        result = load(source)
        for shot_index, row in enumerate(result['镜头']):
            shot_id = row['镜头']
            if shot_id not in by_shot:
                raise ValueError('动作结果镜头不在镜头表中：' + shot_id)
            order, shot = by_shot[shot_id]
            local_people = {p['人物'] for p in row['画面人物']}
            authorized = {f['文件']: f for f in shot['帧']}
            explicit = {p['人物'] for p in row['画面人物'] if '依据帧' in p}
            for kind, actor_key in (('画面人物', '人物'), ('动作', '执行者'), ('神态', '人物')):
                for entry_index, entry in enumerate(row.get(kind, [])):
                    actor = entry[actor_key]
                    if kind == '画面人物':
                        if '依据帧' not in entry:
                            continue
                        evidence = entry['依据帧']
                        if (not isinstance(evidence, list) or not evidence
                                or any(not isinstance(f, str) or not FRAME_FILENAME.fullmatch(f)
                                       for f in evidence)):
                            raise ValueError(f'画面人物依据帧须为非空合法帧文件名数组：{actor}｜{shot_id}')
                    elif actor not in by_person:
                        continue
                    person = by_person.get(actor)
                    if person and (actor not in local_people or shot_id not in person['出现镜头']
                                   or person['片段'] != result['片段']):
                        raise ValueError(f'人物依据与片段/本镜头声明不符：{actor}｜{shot_id}')
                    for filename in entry.get('依据帧', []):
                        dense = filename.startswith('t')
                        if dense:
                            if filename not in dense_cache:
                                dense_cache[filename] = _dense_frame(episode, filename, shot_table['视频'])
                            frame, path, indexes = dense_cache[filename]
                            used_indexes.update(indexes)
                        else:
                            # 允许 frames 根本身是合法素材链接；子路径不得逃出该根。
                            path = frame_path(frames_root, filename)
                            if filename not in authorized:
                                raise ValueError(f'非本镜头授权帧：{actor}｜{shot_id}｜{filename}')
                            frame = authorized[filename]
                        seconds = time_seconds(frame['时间'])
                        start, end = time_seconds(shot['起']), time_seconds(shot['止'])
                        if not (start <= seconds < end if dense else start <= seconds <= end):
                            raise ValueError(f'帧时间越出本镜头：{shot_id}｜{filename}')
                        # 显式依据不被本声明的动作/神态抢占，旧声明仍可提供回退候选。
                        if person is None or (kind != '画面人物' and actor in explicit):
                            continue
                        basis = {
                            '类型': kind + '依据帧', '人物': actor,
                            '来源': str(source.relative_to(episode)),
                            '镜头序号': shot_index, '条目序号': entry_index,
                        }
                        if kind == '动作':
                            basis['原确定度'] = entry.get('确定度')
                        candidate = {'文件': filename, '路径': str(path), '镜头': shot_id,
                                     '时间': frame['时间'], '选择依据': basis}
                        key = (kind != '画面人物', time_seconds(shot['起']), order, seconds, filename,
                               kind != '动作', source.name, shot_index, entry_index)
                        candidates[actor].append((key, candidate))

    rows, images, relations, missing = [], [], {}, []
    for person in people:
        identifier = person['id']
        selected = min(candidates[identifier], key=lambda item: item[0])[1] if candidates[identifier] else None
        row = {key: person[key] for key in ('id', '片段', '外观', '出现镜头')}
        row.update({'候选帧': [selected] if selected else [], '视觉适用性': VISUAL_STATUS})
        if selected:
            row['候选状态'] = '已关联，视觉待确认'
            path = selected['路径']
            if path not in relations:
                images.append(path)
                relations[path] = []
            relations[path].append(f"{identifier}｜{selected['镜头']}｜{selected['时间']}")
        else:
            row['候选状态'] = '未找到本人画面人物依据帧或旧声明动作/神态依据帧'
            missing.append({'人物': identifier, '原因': row['候选状态']})
        rows.append(row)

    subtitles_path = episode / '字幕记录.json'
    subtitles = load(subtitles_path) if subtitles_path.is_file() else []
    # 不输出动作/神态叙述、口型、声音全文、映射结论或其他画面文字。
    identity_subtitles = [{key: row[key] for key in ('起', '止', '文字', '类型')}
                          for row in subtitles if row['类型'] in ('人物字卡', '地点字幕')]
    over_limit = len(images) > limit
    capacity = '候选超限：需后续分批' if over_limit else (
        '未满足：候选材料不完整' if missing else '候选容量内')
    return {
        '用途': '离线人物合并候选材料与容量核验；不是模型请求或身份确认结果',
        '来源快照': str(episode),
        '来源文件': [str(path.relative_to(episode)) for path in
                     [people_path, shots_path, *result_paths, *sorted(used_indexes),
                      *([subtitles_path] if subtitles_path.is_file() else [])]],
        '帧根': str(frames_root.resolve()), '选择机制': SELECTION_RULE,
        '视觉适用性': VISUAL_STATUS,
        '临时人物': rows, '身份证据字幕': identity_subtitles,
        'images': images,
        'labels': [Path(path).name + '｜' + '；'.join(relations[path]) for path in images],
        '统计': {
            '临时声明数': len(people), '候选关联人数': len(people) - len(missing),
            '缺失人数': len(missing), '实际唯一图数U': len(images), '图片上限': limit,
            '上限来源': 'vision.max_images_per_call' if max_images is None else '显式参数',
            '候选关联覆盖': '不完整' if missing else '完整',
            '材料状态': '候选材料不完整' if missing else ('候选超限' if over_limit else '候选容量内'),
            '容量状态': capacity, '超过图片上限': over_limit,
            '视觉覆盖': '未确认',
        },
        '缺失项': missing,
    }
