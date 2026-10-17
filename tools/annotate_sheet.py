"""从字幕预填人工标注，生成无需服务器的逐镜头校对页。"""

import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from s2s.evaluate import validate_annotation

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'eval'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prefill(episode: Path) -> dict:
    shots = read_json(episode / '镜头表.json')
    subtitles = read_json(episode / '字幕记录.json')
    lines = sorted((row for row in subtitles if row['类型'] == '台词字幕'), key=lambda row: (row['起'], row['止']))
    return {'集': shots['集'], '视频': shots['视频'], '校对完成': False, '人物': [],
            '台词': [{'id': f'L{i:03d}', '起': row['起'], '止': row['止'], '文本': row['文字'],
                    '说话人': None, '形态': None} for i, row in enumerate(lines, 1)], '关键动作': []}


def relative_url(path: Path, output: Path) -> str:
    return quote(Path(os.path.relpath(path.resolve(), output.parent.resolve())).as_posix(), safe='/')


def render_sheet(episode: Path, annotation: dict, output: Path) -> str:
    validate_annotation(annotation)
    source = read_json(episode / '镜头表.json')
    if source['集'] != annotation['集'] or source['视频'] != annotation['视频']:
        raise ValueError('标注与镜头表的集号或视频不一致')
    shots = []
    for row in source['镜头']:
        frames = row['帧']
        selected = sorted({0, len(frames) // 2, len(frames) - 1}) if frames else []
        shots.append({'id': row['id'], '起': row['起'], '止': row['止'], 'frames': [
            {'时间': frames[i]['时间'], 'src': relative_url(episode / '素材' / 'frames' / frames[i]['文件'], output)}
            for i in selected]})
    payload = {'annotation': annotation, 'shots': shots, 'schema': read_json(ASSETS / '标注.schema.json')}
    # JSON 位于 script 文本节点，转义 < 防止字幕内容提前结束节点。
    data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    video = Path(source['视频'])
    if not video.is_absolute():
        video = ROOT / video
    video_url = html.escape(relative_url(video, output), quote=True)
    css = (ASSETS / 'sheet.css').read_text(encoding='utf-8')
    js = (ASSETS / 'sheet.js').read_text(encoding='utf-8')
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>第{source['集']}集 标注工作表</title><style>{css}</style></head><body>
<header><a href="#shots">跳到镜头</a><h1>第{source['集']}集 标注工作表</h1>
<p>先回听校对台词，再登记人物与关键动作。字幕时间仅供参考；修改不会自动保存，离开前请导出。</p>
<p id="counts"></p><nav aria-label="标注区域"><a href="#people-title">人物</a><a href="#actions-title">关键动作</a><a href="#shots-title">逐镜头台词</a></nav>
<div class="toolbar"><button id="export" type="button">导出标注.json</button><label>导入已保存标注<input id="import" type="file" accept=".json,application/json"></label>
<label><input id="complete" type="checkbox">已完整回看、回听并校对，无待核对项</label></div>
<p id="status" role="status" tabindex="-1"></p></header><main>
<section aria-labelledby="video-title"><h2 id="video-title">原视频</h2><video id="video" src="{video_url}" controls preload="metadata"></video></section>
<section aria-labelledby="people-title"><h2 id="people-title">人物</h2><p>姓名没有依据时填具体描述名；出现时间用 00:05.0-00:30.0，多个时间段用逗号隔开。</p><button id="add-person" type="button">添加人物</button><div id="people"></div></section>
<section aria-labelledby="actions-title"><h2 id="actions-title">关键动作</h2><p>执行者选人物编号；对象是人物时填 H 编号，是物品时写物品名称；没有则留空。</p><button id="add-action" type="button">添加关键动作</button><div id="actions"></div></section>
<section aria-labelledby="shots-title"><h2 id="shots-title">逐镜头台词</h2><div class="toolbar"><button id="add-line" type="button">添加台词</button><button id="refresh" type="button">按修改后的时间重新排列</button></div><div id="shots"></div><h3>未落在镜头内的台词</h3><div id="unplaced"></div></section>
</main><script id="sheet-data" type="application/json">{data}</script><script>{js}</script></body></html>'''


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('annotation', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == args.annotation.resolve():
            raise ValueError('HTML 输出不能覆盖标注 JSON')
        annotation = read_json(args.annotation) if args.annotation.exists() else prefill(args.episode)
        page = render_sheet(args.episode, annotation, args.output)
        if not args.annotation.exists():
            args.annotation.parent.mkdir(parents=True, exist_ok=True)
            args.annotation.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if args.output.exists():
            args.output.read_text(encoding='utf-8')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(page, encoding='utf-8')
        print(json.dumps({'输出': str(args.output), '集': annotation['集'], '台词': len(annotation['台词']),
                          '校对完成': annotation.get('校对完成', False)}, ensure_ascii=False))
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({'错误': str(exc)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
