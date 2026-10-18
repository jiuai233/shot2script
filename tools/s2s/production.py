"""视频输入、固定素材处理、主控调度与经验收剧本交付。"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from . import config
from .board import initialize_board
from .docx_build import build_docx
from .docx_check import check_docx
from .quality import check_script
from .validate import validate_file

VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    if path.exists():
        path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def stamp(path):
    path = Path(path).resolve()
    info = path.stat()
    return {'路径': str(path), '字节': info.st_size, '修改时间ns': info.st_mtime_ns}


def require(result, name):
    if not result.get('通过'):
        raise ValueError(name + '未通过：' + json.dumps(result.get('错误', result), ensure_ascii=False))
    return result


def discover_videos(directory, count):
    directory = Path(directory).expanduser().resolve()
    if count < 1 or not directory.is_dir():
        raise ValueError('集数须为正整数，素材目录须存在')
    found = {}
    for path in sorted(directory.iterdir()):
        match = re.fullmatch(r'E?(\d+)', path.stem, re.IGNORECASE)
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES or not match:
            continue
        number = int(match[1])
        if not 1 <= number <= count:
            continue
        if number in found:
            raise ValueError(f'E{number:02d} 有多个视频，不能自动选择')
        found[number] = path
    missing = [f'E{n:02d}' for n in range(1, count + 1) if n not in found]
    if missing:
        raise ValueError('缺失视频：' + '、'.join(missing))
    return found


def resolve_video(series, number, video=None):
    if video:
        path = Path(video).expanduser().resolve()
    elif (series / '生产任务.json').is_file():
        mapping = read(series / '生产任务.json').get('视频', {})
        if str(number) not in mapping:
            raise ValueError(f'生产任务没有 E{number:02d} 的视频')
        path = Path(mapping[str(number)])
    else:
        table = series / f'E{number:02d}' / '镜头表.json'
        source = read(table).get('视频') if table.is_file() else None
        if source:
            path = Path(source)
            if not path.is_absolute():
                path = config.PROJECT_ROOT / path
        else:
            directory = config.PROJECT_ROOT / 'videos' / series.name
            candidates = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
                          and re.fullmatch(r'E?0*' + str(number), p.stem, re.IGNORECASE)] if directory.is_dir() else []
            if len(candidates) != 1:
                raise ValueError(f'请用 --视频 指定 E{number:02d} 的唯一输入视频')
            path = candidates[0]
    if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f'视频不存在或格式不支持：{path}')
    return path.resolve()


def prepare_shots(video, episode):
    from .shots import build_shot_table
    build_shot_table(video, episode)


def prepare_segments(video, episode):
    from .segments import write_segments
    write_segments(episode / '镜头表.json')


def prepare_audio(video, episode):
    from .asr.single import run
    run(video, episode)


def prepare_subtitles(video, episode):
    from .subtitles import load_frames, recognize, merge_records
    frames = load_frames(episode, None, 2)
    raw = recognize([path for path, _ in frames])
    for row, (path, seconds) in zip(raw, frames):
        row.update(path=str(path), time=seconds)
    rows, _ = merge_records(raw)
    write(episode / '字幕记录.json', rows)


def prepare_lipsync(video, episode):
    # 口型工具保留独立证据目录，根目录只保存本轮有效记录。
    folder = episode / '素材' / '口型'
    folder.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=folder)) / '口型记录.json'
    python = config.get_environment().get('S2S_LIPSYNC_PYTHON') or sys.executable
    command = [python, str(config.PROJECT_ROOT / 'tools/lipsync.py'),
               str(video), str(episode), str(output), '--model',
               str(config.PROJECT_ROOT / 'models/lipsync/face_landmarker.task')]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1200,
                               cwd=config.PROJECT_ROOT)
    if completed.returncode:
        raise RuntimeError('口型处理失败：' + (completed.stderr or completed.stdout)[-3000:])
    write(episode / '口型记录.json', read(output))


def prepare_voices(video, episode):
    from .asr.single import flag, update_pending
    from .diarize.integration import analyze_board
    from .listen.storage import save_board
    from .voiceprint.library import compare, model_info
    settings = config.load_config().get('diarize', {})
    if settings.get('enabled') is not True:
        raise ValueError('完整生产入口要求启用本地声纹组件 diarize.enabled')
    sounds = read(episode / '声音记录.json')
    vectors = []
    sounds, clustering = analyze_board(video, sounds, episode, settings, vectors_out=vectors)
    for sound, row in zip(sounds, clustering['句列表']):
        if row['待定']:
            flag(sound, '声纹待定')
        if row['疑似重叠']:
            flag(sound, '句内换人或重叠', 1)
        for conflict in clustering.get('约束', {}).get('冲突', []):
            if sound['id'] in conflict.get('句', []):
                flag(sound, conflict['类型'], 1)
    candidates = compare(episode.parent, episode.name,
                         extracted=(sounds, vectors, model_info(settings.get('model', 'campp'))),
                         clustering=clustering, persist=False)
    pending = update_pending(sounds, read(episode / '待核对.json'), episode.name,
                             read(episode / '素材/回听/疑似漏段.json'))
    save_board(episode, sounds, pending, '口型声纹刷新')
    write(episode / '声纹聚类.json', clustering)
    write(episode / '声纹候选.json', candidates)


def prepare_listen(video, episode):
    from .listen.sheet import build_sheet
    build_sheet(episode, episode / '回听.html')


# 输入仅列上游资料；回听改字不触发重新识别，避免覆盖人工决定。
PHASES = (
    ('镜头', prepare_shots, (), ('镜头表.json',)),
    ('分段', prepare_segments, ('镜头表.json',), ('片段表.json',)),
    ('语音', prepare_audio, (), ('声音记录.json', '素材/人声段.json', '素材/回听/疑似漏段.json')),
    ('字幕', prepare_subtitles, ('镜头表.json',), ('字幕记录.json',)),
    ('口型', prepare_lipsync, ('镜头表.json', '声音记录.json'), ('口型记录.json',)),
    ('声纹', prepare_voices, ('声音记录.json', '口型记录.json'), ('声纹聚类.json', '声纹候选.json')),
    ('回听', prepare_listen, ('声音记录.json', '字幕记录.json', '待核对.json'), ('回听.html',)),
)


def input_version(episode, names, *, optional_lipsync=False):
    result = {}
    for name in names:
        if optional_lipsync and name == '口型记录.json' and not (episode / name).is_file():
            result[name] = None
            continue
        if name == '声音记录.json':
            result[name] = read(episode / name)
        else:
            result[name] = stamp(episode / name)
    return result


def prepare_episode(video, episode, *, optional_lipsync=False):
    path = episode / '生产任务.json'
    job = read(path) if path.is_file() else {}
    job.setdefault('阶段', {})
    source = stamp(video)
    if job.get('视频') and job['视频'] != source:
        raise ValueError('本集已绑定其他视频版本；请使用新的剧名或集号，避免混用旧证据')
    table = read(episode / '镜头表.json')
    if table.get('镜头') and table.get('视频'):
        old_video = Path(table['视频'])
        if not old_video.is_absolute():
            old_video = config.PROJECT_ROOT / old_video
        if old_video.resolve() != video.resolve():
            raise ValueError('镜头表已绑定其他视频；请使用新的剧名或集号')
    job['视频'] = source
    for name, operation, inputs, outputs in PHASES:
        version = input_version(episode, inputs, optional_lipsync=optional_lipsync)
        previous = job['阶段'].get(name, {})
        if (optional_lipsync and name == '口型' and previous.get('状态') == '降级'
                and previous.get('输入') == version and not (episode / '口型记录.json').exists()):
            continue
        if (previous.get('状态') == '完成' and previous.get('输入') == version
                and all((episode / p).is_file() for p in outputs)):
            if name != '镜头' or all((episode / '素材/frames' / f['文件']).is_file()
                                      for s in read(episode / '镜头表.json')['镜头'] for f in s['帧']):
                continue
        job.update(状态='运行中', 当前阶段=name, 错误=None)
        job['阶段'][name] = {'状态': '运行中', '输入': version}
        write(path, job)
        try:
            operation(video, episode)
            for output in outputs:
                if not (episode / output).is_file():
                    raise ValueError(f'{name}未产出 {output}')
            job['阶段'][name] = {'状态': '完成', '输入': input_version(episode, inputs, optional_lipsync=optional_lipsync)}
            write(path, job)
        except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
            if optional_lipsync and name == '口型':
                output = episode / '口型记录.json'
                if output.exists():
                    backup = episode / '素材/口型' / ('失败-' + str(output.stat().st_mtime_ns) + '.json')
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    output.replace(backup)
                job['阶段'][name] = {'状态': '降级', '输入': version, '错误': str(exc),
                                     '证据状态': '缺失；不能作为否定证据'}
                write(path, job)
                continue
            job.update(状态='失败', 错误=str(exc))
            write(path, job)
            raise
    # 声纹步骤可能补充异常字段；记录最终口型输入，避免下次因自身输出重复运行。
    job['阶段']['口型']['输入'] = input_version(episode, ('镜头表.json', '声音记录.json'), optional_lipsync=optional_lipsync)
    for name in ('镜头表', '片段表', '声音记录', '字幕记录', '口型记录', '声纹聚类', '声纹候选'):
        if name == '口型记录' and optional_lipsync and not (episode / f'{name}.json').is_file():
            continue
        require(validate_file(episode / f'{name}.json'), name)
    job.update(状态='素材就绪', 当前阶段='主控', 错误=None)
    write(path, job)
    return job


def check_episode_delivery(episode):
    state = read(episode / '状态.json')
    require(validate_file(episode / '状态.json'), '单集状态')
    result = {'通过': False, '状态': state['状态'], '目录': str(episode)}
    draft = episode / '工作稿.md'
    if draft.is_file() and draft.read_text(encoding='utf-8').strip():
        result['草稿'] = str(draft)
    if state['状态'] == '待人工裁定':
        result.update(通过=True, 正式交付=False, 待核对=str(episode / '待核对.json'),
                      问题清单=str(episode / '问题清单.json'), 回听页=str(episode / '回听.html'))
        return result
    if state['状态'] != '验收通过':
        raise ValueError(f'主控已退出，但本集状态仍为 {state["状态"]}；重新运行同一命令继续')
    if state['未决问题']:
        raise ValueError('仍有未解决问题，不能正式交付')
    for name in ('正式稿.md', '工作稿.md', '验收报告.md'):
        if not (episode / name).read_text(encoding='utf-8').strip():
            raise ValueError(f'{name}为空，不能正式交付')
    formal = episode / '正式稿.md'
    require(check_script(episode, formal, formal=True), '正式稿检查')
    result.update(通过=True, 正式交付=True, Markdown=str(formal), 验收报告=str(episode / '验收报告.md'))
    return result


def _run_episode_controller(series, number, *, video=None, mode='完整转写', review='', timeout=900):
    from .dispatch.controllers import ensure_state, episode_number, episode_prompt, start_controller
    number = episode_number(number)
    if mode not in ('完整转写', '定向复核') or mode == '定向复核' and not review.strip():
        raise ValueError('模式须为完整转写或提供复核要求的定向复核')
    episode, _ = ensure_state(series, number, config.PROJECT_ROOT / '剧目')
    source = resolve_video(episode.parent, number, video)
    job_path = episode / '生产任务.json'
    try:
        job = prepare_episode(source, episode)
        version = input_version(episode, ('镜头表.json', '声音记录.json', '字幕记录.json', '口型记录.json', '声纹候选.json', '待核对.json'))
        if job.get('主控输入') and job['主控输入'] != version:
            state = read(episode / '状态.json')
            state.update(状态='进行中', 步骤={str(i): False for i in range(1, 10)})
            write(episode / '状态.json', state)
        state = read(episode / '状态.json')
        if mode != '完整转写' or state['状态'] not in ('验收通过', '待人工裁定'):
            job.update(状态='运行中', 当前阶段='主控', 主控输入=version)
            write(job_path, job)
            episode, prompt = episode_prompt(series, number, mode=mode, review=review)
            start_controller('单集主控', prompt, episode / '会话', timeout=timeout)
        result = check_episode_delivery(episode)
        job.update(状态=result['状态'], 当前阶段='完成', 错误=None,
                   主控输入=input_version(episode, tuple(version)), 交付=result)
        write(job_path, job)
        return result
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        job = read(job_path) if job_path.is_file() else {}
        job.update(状态='失败', 错误=str(exc))
        write(job_path, job)
        raise


def run_episode(series, number, *, video=None, mode='完整转写', review='', timeout=900,
                scheduler='主控', run_root=None, fixture=None, approvals=None, max_requests=100, preflight=False):
    """生产默认保留原Pi主控；固定单集E0须显式选择，当前仅离线可运行。"""
    if scheduler == '主控':
        if any((run_root, fixture, approvals, preflight)):
            raise ValueError('旧主控不接受固定调度的运行目录、夹具、审核或预检参数')
        return _run_episode_controller(series, number, video=video, mode=mode, review=review, timeout=timeout)
    if scheduler != '固定' or mode != '完整转写' or review:
        raise ValueError('固定首阶段只支持完整单集；定向复核请显式使用 --调度 主控')
    from .fixed_pipeline import run_fixed_episode
    return run_fixed_episode(series, number, video=video, run_root=run_root, fixture=fixture,
                             approvals=approvals, timeout=timeout, max_requests=max_requests, preflight=preflight)


def export_series(series, count):
    deliveries = [check_episode_delivery(series / f'E{n:02d}') for n in range(1, count + 1)]
    accepted = [d for d in deliveries if d['正式交付']]
    missing = {n for n, d in enumerate(deliveries, 1) if not d['正式交付']}
    result = {'通过': True, '正式交付': not missing, '状态': '待人工裁定' if missing else '验收通过',
              '单集': deliveries, '未收录集': sorted(missing)}
    if len(accepted) > 1:
        cross_path = series / '派发结果/跨集验收.json'
        if not cross_path.is_file():
            raise ValueError('缺少跨集验收，不能导出全剧正式合集')
        cross = read(cross_path)
        if cross.get('模式') != '跨集' or cross.get('结论') != '通过' or cross.get('问题'):
            raise ValueError('跨集验收尚未通过，不能导出全剧正式合集')
        if any(Path(d['Markdown']).stat().st_mtime_ns > cross_path.stat().st_mtime_ns for d in accepted):
            raise ValueError('跨集验收后正式稿发生变化，须重新跨集验收')
    if accepted:
        text = '\n\n'.join(Path(d['Markdown']).read_text(encoding='utf-8').strip() for d in accepted) + '\n'
        folder = series / '合并'
        folder.mkdir(exist_ok=True)
        markdown = folder / ('合并稿.md' if not missing else '部分合并稿.md')
        if missing:
            text = '未收录集：' + '、'.join(f'E{n:02d}' for n in sorted(missing)) + '\n\n' + text
        if markdown.exists():
            markdown.read_text(encoding='utf-8')
        markdown.write_text(text, encoding='utf-8')
        word = markdown.with_suffix('.docx')
        if word.exists():
            word.read_bytes()
        require(build_docx(markdown, word, series.name + ' 剧本合集'), 'Word 生成')
        require(check_docx(word, count, missing=missing), 'Word 检查')
        result.update(Markdown=str(markdown), Word=str(word))
    write(series / '全剧状态.json', {'状态': result['状态'], '集': [
        {'集': n, '状态': d['状态']} for n, d in enumerate(deliveries, 1)]})
    return result


def run_series(name, count, *, directory=None, timeout=900):
    from .dispatch.controllers import series_prompt, start_controller
    if not name.strip() or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('剧名须为单个目录名')
    directory = directory or config.PROJECT_ROOT / 'videos' / name
    videos = discover_videos(directory, count)
    source_versions = {str(n): stamp(p) for n, p in videos.items()}
    series = config.PROJECT_ROOT / '剧目' / name
    path = series / '生产任务.json'
    job = read(path) if path.is_file() else {}
    if job.get('输入版本') and job['输入版本'] != source_versions:
        raise ValueError('全剧输入或集数已变化；请使用新的剧名，避免混用旧证据')
    initialize_board(name, count, root=config.PROJECT_ROOT / '剧目')
    job.update(状态='运行中', 视频={str(n): str(p) for n, p in videos.items()},
               输入版本=source_versions, 错误=None,
               更新时间=datetime.now(timezone(timedelta(hours=8))).isoformat())
    write(path, job)
    try:
        prompt = series_prompt(name, count, videos=videos)
        start_controller('全剧主控', prompt, series / '会话', timeout=timeout)
        state = read(series / '全剧状态.json')
        require(validate_file(series / '全剧状态.json'), '全剧状态')
        if state['状态'] not in ('验收通过', '待人工裁定'):
            raise ValueError('全剧主控尚未完成跨集审核，重新运行同一命令继续')
        for number, source in videos.items():
            episode_job = read(series / f'E{number:02d}' / '生产任务.json')
            if episode_job.get('视频') != stamp(source) or episode_job.get('状态') not in ('验收通过', '待人工裁定'):
                raise ValueError(f'E{number:02d} 尚未完成当前视频的生产流程')
        result = export_series(series, count)
        job.update(状态=result['状态'], 交付=result)
        write(path, job)
        return result
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        job.update(状态='失败', 错误=str(exc))
        write(path, job)
        raise
