"""独立解析 Word 成品，检查剧集、场景、导航和排版。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from zipfile import BadZipFile, is_zipfile

from docx import Document
from lxml import etree as ET

from .docx_build import EPISODE, TEMPLATE, W, read_parts, template_layout, text_of, xml

SCENE = re.compile(r"场景([1-9][0-9]*)-([1-9][0-9]*)：([^，\s：]+)，([^，\s：]+)，(内|外|内/外)(（闪回）)?")


def check_docx(path: str | Path, episode_count: int, missing: set[int] | None = None) -> dict:
    """episode_count 为全剧总集数；missing 为未收录集号。"""
    missing = set(missing or ())
    errors = []

    def fail(rule: str, detail: str, index: int | None = None, text: str = ""):
        error = {"规则": rule, "说明": detail}
        if index is not None:
            error.update({"段落": index + 1, "原文": text})
        errors.append(error)

    result = {"通过": False, "总集数": episode_count, "缺失集": sorted(missing), "错误": errors}
    if episode_count < 1 or any(n < 1 or n > episode_count for n in missing):
        fail("参数", "集数必须为正整数；缺失集号必须在 1..总集数 内")
        return result
    try:
        parts = read_parts(path)
        required = ("[Content_Types].xml", "_rels/.rels", "word/document.xml",
                    "word/_rels/document.xml.rels", "word/styles.xml", "word/settings.xml",
                    "word/theme/theme1.xml", "word/fontTable.xml")
        for name in required:
            xml(parts[name])
        # 实际打开标准包，核对 Content Types、关系及关系目标是否可读。
        if is_zipfile(path):
            Document(str(path))
        doc = xml(parts["word/document.xml"])
        layout = template_layout(read_parts(TEMPLATE))
        styles = {s.get(W + "styleId"): s for s in xml(parts["word/styles.xml"]).findall(W + "style")}
    except (OSError, ValueError, KeyError, BadZipFile, ET.XMLSyntaxError) as exc:
        fail("文档结构", f"Word 包无法完整解析：{exc}")
        return result

    bodies = list(doc.iter(W + "body"))
    if doc.tag != W + "document" or len(bodies) != 1 or bodies[0].getparent() is not doc:
        fail("文档结构", "document 必须包含唯一的直属 body")
        return result
    body = bodies[0]
    paragraphs = body.findall(W + "p")
    if any(p.getparent() is not body for p in doc.iter(W + "p")):
        fail("文档结构", "正文段落必须直接位于 body 内，不能游离或嵌套")
    sections = list(doc.iter(W + "sectPr"))
    if len(sections) != 1 or not len(body) or body[-1].tag != W + "sectPr":
        fail("文档结构", "body 末尾必须有唯一 sectPr")
    if any(child.tag not in (W + "p", W + "sectPr") for child in body):
        fail("文档结构", "body 包含非剧本段落结构")
    if not paragraphs or not text_of(paragraphs[0]).strip():
        fail("大标题", "首段必须是非空大标题")
    if sections:
        for tag in ("pgSz", "pgMar", "cols"):
            actual = sections[0].find(W + tag)
            expected = layout["section"].find(W + tag)
            if actual is None or dict(actual.attrib) != dict(expected.attrib):
                fail("页面设置", f"{tag} 与范本不一致")

    def style_property(style_id: str, tag: str, seen: set[str] | None = None):
        seen = set() if seen is None else seen
        if style_id in seen or style_id not in styles:
            return None
        seen.add(style_id)
        style = styles[style_id]
        prop = style.find(W + "pPr/" + W + tag)
        if prop is not None:
            return prop.get(W + "val")
        base = style.find(W + "basedOn")
        return style_property(base.get(W + "val"), tag, seen) if base is not None else None

    def property_of(p, tag):
        prop = p.find(W + "pPr/" + W + tag)
        if prop is not None:
            return prop.get(W + "val")
        ref = p.find(W + "pPr/" + W + "pStyle")
        return style_property(ref.get(W + "val") if ref is not None else "1", tag)

    texts = [text_of(p) for p in paragraphs]
    numbers, scene_count, levels = [], 0, []
    current_episode, scene_number = None, 0
    per_episode_scenes = {}
    for i, (p, text) in enumerate(zip(paragraphs, texts)):
        ep = EPISODE.fullmatch(text)
        is_scene = text.startswith("场景")
        kind = "title" if i == 0 else "episode" if ep else "scene" if is_scene else "body"
        if i == 0 and (ep or is_scene):
            fail("大标题", "首段不能是集标题或场景行", i, text)
        level = property_of(p, "outlineLvl")
        levels.append(level)
        expected_level = {"episode": "0", "scene": "1"}.get(kind)
        if (expected_level is not None and level != expected_level) or (expected_level is None and level not in (None, "9")):
            fail("导航层级", f"{kind} 大纲级别应为 {expected_level or '正文'}，实际为 {level}", i, text)
        ref = p.find(W + "pPr/" + W + "pStyle")
        if text and (ref is None or ref.get(W + "val") != layout[kind]["style"]):
            fail("段落样式", f"应引用范本样式 {layout[kind]['style']}", i, text)
        if kind == "title" and property_of(p, "jc") != "center":
            fail("大标题", "大标题必须居中", i, text)
        if ep:
            current_episode, scene_number = int(ep.group(1)), 0
            numbers.append(current_episode)
            per_episode_scenes.setdefault(current_episode, 0)
            if i == 0 or texts[i - 1].strip():
                fail("集前空段", "每个集标题前必须空一段", i, text)
        elif text.startswith("第") and re.match(r"第.*集", text):
            fail("集标题格式", "集标题应为 第{正整数}集", i, text)
        if is_scene:
            scene_count += 1
            match = SCENE.fullmatch(text)
            if match is None:
                fail("场景格式", "应为 场景{集号}-{序号}：地点，时间，内/外，可附（闪回）", i, text)
            else:
                ep_number, serial = int(match.group(1)), int(match.group(2))
                if ep_number != current_episode:
                    fail("场景所属集", f"场景集号 {ep_number} 与当前集 {current_episode} 不一致", i, text)
                if serial != scene_number + 1:
                    fail("场景序号", f"本场序号应为 {scene_number + 1}，实际为 {serial}", i, text)
                scene_number = serial
            if current_episode is not None:
                per_episode_scenes[current_episode] += 1
            if i + 1 >= len(texts) or not texts[i + 1].startswith("人物："):
                fail("人物行", "场景行下一段必须以 人物： 开头", i, text)

        spec = layout[kind]
        for run in p.iter(W + "r"):
            if not text_of(run):
                continue
            rpr = run.find(W + "rPr")
            fonts = rpr.find(W + "rFonts") if rpr is not None else None
            if fonts is None or any(fonts.get(W + slot) != font for slot, font in spec["fonts"].items()):
                fail("字体", f"run 必须显式使用 {spec['fonts']}", i, text)
            if fonts is not None and any("theme" in ET.QName(key).localname.lower() for key in fonts.attrib):
                fail("字体", "run 不应依赖主题字体回填", i, text)
            for tag in ("sz", "szCs"):
                size = rpr.find(W + tag) if rpr is not None else None
                if size is None or size.get(W + "val") != spec["size"]:
                    fail("字号", f"run {tag} 应显式为 {spec['size']} 半磅", i, text)
            if spec["bold"]:
                bold = rpr.find(W + "b") if rpr is not None else None
                if bold is None or bold.get(W + "val", "1") in ("0", "false", "off"):
                    fail("加粗", "标题必须加粗", i, text)

    expected = [n for n in range(1, episode_count + 1) if n not in missing]
    if numbers != expected:
        fail("集号", f"应为 {expected}，实际为 {numbers}")
    if set(numbers) & missing:
        fail("缺失集出现", f"不应收录：{sorted(set(numbers) & missing)}")
    if levels.count("0") != len(expected):
        fail("一级大纲数量", f"应为 {len(expected)}，实际为 {levels.count('0')}")
    if levels.count("1") != scene_count:
        fail("二级大纲数量", f"应为场景行数 {scene_count}，实际为 {levels.count('1')}")
    for number, count in per_episode_scenes.items():
        if count == 0:
            fail("场景缺失", f"第{number}集没有场景")
    result.update({"通过": not errors, "收录集号": numbers, "场景数": scene_count,
                   "一级大纲数": levels.count("0"), "二级大纲数": levels.count("1"), "段落数": len(paragraphs)})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="实际解析并校验 Word 剧本")
    parser.add_argument("docx")
    parser.add_argument("集数", type=int, help="全剧总集数，含缺失集")
    parser.add_argument("--缺失集", default="", help="未收录集号，以逗号分隔，例如 3,5")
    args = parser.parse_args(argv)
    try:
        missing = {int(n.strip()) for n in args.缺失集.split(",") if n.strip()}
    except ValueError:
        result = {"通过": False, "错误": [{"规则": "参数", "说明": "缺失集必须是逗号分隔的整数"}]}
    else:
        result = check_docx(args.docx, args.集数, missing)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["通过"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
