"""导出带压缩音频片段的离线回听页。"""

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote

from .. import audio
from ..asr.postprocess import seconds
from ..asr.crosscheck import needs_listen as legacy_needs_listen
from ..asr.single import needs_listen as single_needs_listen
from ..config import PROJECT_ROOT
from ..sub_align import differences
from .storage import read
from .snapshot import summary

ASSETS = Path(__file__).parent


def relative_url(path, output):
    return quote(Path(os.path.relpath(Path(path).resolve(), Path(output).parent.resolve())).as_posix(), safe="/")


def clip(source, target, start, end):
    if target.exists():
        audio.duration(target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run([audio.FFMPEG, "-nostdin", "-v", "error", "-n", "-ss", str(start), "-i", str(source),
                                 "-t", str(end - start), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "24000",
                                 "-c:a", "aac", "-b:a", "64k", str(target)], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        target.unlink(missing_ok=True)
        raise ValueError("ffmpeg 导出回听片段超过 60 秒，已停止") from None
    if result.returncode:
        target.unlink(missing_ok=True)
        raise ValueError(f"ffmpeg 导出回听片段失败：{target}")


def marked(text, others):
    ranges = []
    for other in others:
        for change in differences(text, other):
            start, end = change["字幕原文位置"]
            ranges.append((start, end))
    return "".join(("<mark>" + html.escape(char) + "</mark>") if any(a <= i < b for a, b in ranges)
                   else html.escape(char) for i, char in enumerate(text)) or "<span class=empty>没有识别到</span>"


def needs_listen(row):
    return single_needs_listen(row) if "原始文本" in row.get("核对", {}) else legacy_needs_listen(row)


def build_sheet(episode, output, all_rows=False):
    episode, output = Path(episode), Path(output)
    sounds = read(episode / "声音记录.json")
    shots = read(episode / "镜头表.json")
    source = Path(shots["视频"])
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    total = audio.duration(source)
    migration_path = episode / "素材/回听/裁定迁移.json"
    migrations = {r["新句号"]: r for r in read(migration_path) if r.get("新句号")} if migration_path.exists() else {}
    subtitles = read(episode / "字幕记录.json") if (episode / "字幕记录.json").exists() else []
    # 字幕只在此处展示，选句和排序都不引用字幕内容。
    selected = sounds if all_rows else [s for s in sounds if needs_listen(s) or s.get("核对", {}).get("状态") != "人工确认" and migrations.get(s["id"], {}).get("状态") == "待确认"]
    selected = sorted(selected, key=lambda r: (r.get("核对", {}).get("优先级", 2), seconds(r["起"]), seconds(r["止"])))
    cards, items = [], []
    media = episode / "素材" / "回听"
    positions = {r["id"]: i for i, r in enumerate(sounds)}
    for index, row in enumerate(selected):
        start, end = seconds(row["起"]), min(total, seconds(row["止"]))
        if not 0 <= start < end:
            raise ValueError(f'句子时间超出视频：{row["id"]}')
        check = {"主路": row["文本"], "副路": "", "字幕": "", "状态": "未核对", "优先级": 2, **row.get("核对", {})}
        reference = "".join(s["文字"] for s in subtitles if s["类型"] == "台词字幕" and
                            (min(end, seconds(s["止"])) > max(start, seconds(s["起"])) or
                             s["起"] == s["止"] and start <= seconds(s["起"]) < end))
        check["字幕"] = reference or check.get("字幕", "")
        position = positions[row["id"]]
        previous, following = sounds[max(0, position - 1)], sounds[min(len(sounds) - 1, position + 1)]
        clip_start, clip_end = max(0, start - .2), min(total, end + .2)
        context_start, context_end = seconds(previous["起"]), min(total, seconds(following["止"]))
        # 文件名含实际裁剪范围和版本，不能复用旧的未补边或固定时长上下文片段。
        stem = f'{row["id"]}_{clip_start:.1f}-{clip_end:.1f}_v2'
        sentence = media / f"{stem}.m4a"
        context = media / f'{row["id"]}_{context_start:.1f}-{context_end:.1f}_上下文_v2.m4a'
        clip(source, sentence, clip_start, clip_end)
        clip(source, context, context_start, context_end)
        shot = max((s for s in shots["镜头"] if min(end, seconds(s["止"])) > max(start, seconds(s["起"]))),
                   key=lambda s: min(end, seconds(s["止"])) - max(start, seconds(s["起"])), default=None)
        frame_html = "<p>没有代表帧</p>"
        if shot and shot["帧"]:
            frame = min(shot["帧"], key=lambda f: abs(seconds(f["时间"]) - (start + end) / 2))
            path = episode / "素材" / "frames" / frame["文件"]
            if not path.is_file():
                raise ValueError(f"代表帧不存在：{path}")
            frame_html = f'<img loading="lazy" src="{html.escape(relative_url(path, output), quote=True)}" alt="{shot["id"]}，{frame["时间"]}代表帧">'
        single = "原始文本" in check
        item = {"句号": row["id"], "主路": check["主路"], "副路": check["副路"], "字幕": check["字幕"], "听辨": row["听辨"]}
        if single:
            item = {k: v for k, v in item.items() if k not in ("主路", "副路")}
            item["原文"] = row["文本"]
        item["本句片段"] = [clip_start, clip_end]
        item["上下文片段"] = [context_start, context_end]
        items.append(item)
        names = ("原文", "字幕") if single else ("主路", "副路", "字幕")
        texts = [item[k] for k in names]
        comparisons = "".join(f'<div class="reading"><h3>{name}{"（参考，可能不准）" if name == "字幕" else ""}</h3><p>{marked(texts[i], [t for j, t in enumerate(texts) if j != i])}</p><button type="button" data-choice="{name}" {"disabled" if not texts[i] else ""}>采用{name}（{i + 1}）</button></div>' for i, name in enumerate(names))
        prior = migrations.get(row["id"], {})
        anomaly_html = "<p>待核对原因：" + html.escape("；".join(check.get("异常", []))) + "</p>" if single and check.get("异常") else ""
        prior_html = '<p class="prior">原裁定：' + html.escape(prior["原裁定"]) + '</p>' if prior.get("状态") == "待确认" else ""
        cards.append(f'''<article data-index="{index}"><h2 tabindex="-1">{row["id"]} <small>{row["起"]}—{row["止"]}　{check["状态"]}　优先级 {check["优先级"]}</small></h2>
<div class="work"><aside>{frame_html}</aside><div class="content"><div class="playback"><audio controls preload="none" src="{relative_url(sentence, output)}" data-audio="sentence"></audio><audio preload="none" src="{relative_url(context, output)}" data-audio="context"></audio>
<button type="button" data-play="sentence">播放本句（空格）</button><button type="button" data-play="context">播放上下文（C）</button><label><input type="checkbox" data-field="loop">重复播放</label></div>
{prior_html}{anomaly_html}<div class="readings">{comparisons}</div><label>最终文本<textarea data-field="text" rows="2">{html.escape(row["文本"])}</textarea></label>
<div class="decide"><label>听辨<select data-field="clarity">{"".join(f'<option {"selected" if row["听辨"] == c else ""}>{c}</option>' for c in ("清楚", "重叠", "听不清"))}</select></label>
<label><input type="checkbox" data-field="delete">这里没有台词（删除该句）</label><button type="button" data-choice="手改">确认手改文本</button><button type="button" data-reset>撤销本句决定</button><strong data-status>待处理</strong></div>
<label>备注<input type="text" data-field="note"></label></div></div></article>''')
    missing_path = episode / "素材/回听/疑似漏段.json"
    missing_cards = []
    for i, span in enumerate(read(missing_path) if missing_path.exists() else []):
        a, b = seconds(span["起"]), min(total, seconds(span["止"]))
        target = media / f"漏段_{a:.3f}-{b:.3f}.m4a"
        clip(source, target, max(0, a - .2), min(total, b + .2))
        missing_cards.append(f'<section><h2>疑似漏段 {i + 1}：{a:.1f}—{b:.1f} 秒</h2><p>检测到人声但没有识别文本，须回听后定向识别或人工补录，不能按字幕填字。</p><audio controls preload="none" aria-label="疑似漏段 {i + 1}" src="{relative_url(target, output)}"></audio></section>')
    data = json.dumps({"key": "s2s-listen:" + str(episode.resolve()), "snapshot": sounds, "summary": summary(sounds), "items": items}, ensure_ascii=False).replace("<", "\\u003c")
    css = (ASSETS / "sheet.css").read_text(encoding="utf-8")
    js = (ASSETS / "sheet.js").read_text(encoding="utf-8")
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{episode.name} 逐句回听</title><style>{css}</style></head><body>
<header><a href="#sentences">跳到句子</a><h1>{episode.name} 逐句回听</h1><div class="toolbar"><strong id="progress" role="status"></strong><button id="export" type="button">导出回听结果.json</button></div>
<p>听完一句，选正确文本。不确定就留着。字幕仅供参考，可能不准。高亮表示识别文本与参考文本的不同处。</p><details><summary>键盘怎么用</summary><p>空格：播放或暂停本句；C：上下文；数字键：采用本句对应按钮上的文本；J / K：下一句 / 上一句；E：编辑。输入时快捷键不生效，按 Esc 退出输入。</p></details><p id="notice" role="status"></p></header>
<main id="sentences">{"".join(missing_cards)}{"".join(cards) if cards else "<p>没有待回听句子。</p>"}</main><script id="listen-data" type="application/json">{data}</script><script>{js}</script></body></html>'''
    if output.exists():
        output.read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return {"回听页": str(output), "句数": len(items), "音频片段": len(items) * 2, "声音摘要": summary(sounds)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("本集目录", type=Path)
    parser.add_argument("输出", type=Path)
    parser.add_argument("--全部", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.输出.suffix.lower() != ".html":
            raise ValueError("输出须为 .html 文件")
        print(json.dumps(build_sheet(args.本集目录, args.输出, args.全部), ensure_ascii=False))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"生成回听页失败：{exc}\n")
    return 0
