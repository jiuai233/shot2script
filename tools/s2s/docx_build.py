"""以排版范本为底生成标准 OOXML Word 文档。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED, is_zipfile

from lxml import etree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
PKG = "{http://schemas.microsoft.com/office/2006/xmlPackage}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
TEMPLATE = Path(__file__).resolve().parents[2] / "simple/范本_TG26090377假穷真爱_剧本合集_20260916.docx"
EPISODE = re.compile(r"第([1-9][0-9]*)集")


def xml(data: bytes) -> ET._Element:
    return ET.fromstring(data, ET.XMLParser(resolve_entities=False, no_network=True))


def serialize(element: ET._Element) -> bytes:
    return ET.tostring(element, encoding="UTF-8", xml_declaration=True)


def read_parts(path: str | Path) -> dict[str, bytes]:
    """读取标准 docx，或将范本的 Flat OPC 部件映射为 ZIP 条目。"""
    if is_zipfile(path):
        with ZipFile(path) as archive:
            return {name: archive.read(name) for name in archive.namelist()}
    root = xml(Path(path).read_bytes())
    if root.tag != PKG + "package":
        raise ValueError("不是 Word OOXML 或 Flat OPC 文档")
    parts = {}
    types = ET.Element(f"{{{CT}}}Types", nsmap={None: CT})
    for part in root.findall(PKG + "part"):
        name = part.get(PKG + "name")
        data = part.find(PKG + "xmlData")
        if data is None or len(data) != 1:
            raise ValueError(f"范本部件不是单一 XML：{name}")
        parts[name.lstrip("/")] = serialize(data[0])
        ET.SubElement(types, f"{{{CT}}}Override", PartName=name,
                      ContentType=part.get(PKG + "contentType"))
    parts["[Content_Types].xml"] = serialize(types)
    return parts


def text_of(paragraph: ET._Element) -> str:
    pieces = []
    for node in paragraph.iter():
        if node.tag == W + "t":
            pieces.append(node.text or "")
        elif node.tag == W + "tab":
            pieces.append("\t")
        elif node.tag in (W + "br", W + "cr"):
            pieces.append("\n")
    return "".join(pieces)


def template_layout(parts: dict[str, bytes]) -> dict:
    """从当前范本的标题、场景及人物段读取实际排版参数。"""
    doc = xml(parts["word/document.xml"])
    styles = xml(parts["word/styles.xml"])
    theme = xml(parts["word/theme/theme1.xml"])
    paragraphs = doc.findall(W + "body/" + W + "p")
    prototypes = {
        "title": paragraphs[0],
        "episode": next(p for p in paragraphs if EPISODE.fullmatch(text_of(p))),
        "scene": next(p for p in paragraphs if re.match(r"[0-9]+-[0-9]+\s", text_of(p))),
        "body": next(p for p in paragraphs if text_of(p).startswith("人物：")),
    }
    minor = theme.find(".//" + A + "minorFont")
    latin = minor.find(A + "latin").get("typeface")
    east_asia = next(f.get("typeface") for f in minor.findall(A + "font")
                     if f.get("script") == "Hans")
    layout = {}
    for kind, paragraph in prototypes.items():
        ppr = paragraph.find(W + "pPr")
        style_ref = paragraph.find(W + "pPr/" + W + "pStyle")
        style_id = style_ref.get(W + "val") if style_ref is not None else "1"
        style = next(s for s in styles.findall(W + "style") if s.get(W + "styleId") == style_id)
        rpr = ET.Element(W + "rPr")
        # 范本标题继承 Normal，字体的最终落点来自 minor 主题。
        for source in (style.find(W + "rPr"), paragraph.find(W + "r/" + W + "rPr")):
            if source is not None:
                for prop in source:
                    previous = rpr.find(prop.tag)
                    if previous is not None:
                        rpr.remove(previous)
                    rpr.append(deepcopy(prop))
        rf = rpr.find(W + "rFonts")
        fonts = {"ascii": latin, "hAnsi": latin, "eastAsia": east_asia}
        if rf is not None:
            fonts.update({slot: rf.get(W + slot, value) for slot, value in fonts.items()})
        # 范本没有显式复杂文字字体；与对应西文字体统一写出。
        fonts["cs"] = fonts["ascii"]
        layout[kind] = {
            "style": style_id,
            "ppr": deepcopy(ppr) if ppr is not None else ET.Element(W + "pPr"),
            "fonts": fonts,
            "size": rpr.find(W + "sz").get(W + "val"),
            "bold": rpr.find(W + "b") is not None,
        }
    layout["section"] = deepcopy(doc.find(W + "body/" + W + "sectPr"))
    blank = next(p for p in paragraphs if not text_of(p))
    layout["blank_ppr"] = deepcopy(blank.find(W + "pPr"))
    return layout


def paragraph(line: str, kind: str, spec: dict) -> ET._Element:
    p = ET.Element(W + "p")
    ppr = deepcopy(spec["ppr"])
    for tag in ("pStyle", "outlineLvl"):
        for old in ppr.findall(W + tag):
            ppr.remove(old)
    ppr.insert(0, ET.Element(W + "pStyle", {W + "val": spec["style"]}))
    # Word XML 使用零起点：0/1 对应导航窗格的第 1/2 级。
    ET.SubElement(ppr, W + "outlineLvl", {W + "val": {"episode": "0", "scene": "1"}.get(kind, "9")})
    p.append(ppr)
    run = ET.SubElement(p, W + "r")
    rpr = ET.SubElement(run, W + "rPr")
    ET.SubElement(rpr, W + "rFonts", {W + slot: font for slot, font in spec["fonts"].items()})
    if spec["bold"]:
        ET.SubElement(rpr, W + "b")
    for tag in ("sz", "szCs"):
        ET.SubElement(rpr, W + tag, {W + "val": spec["size"]})
    # 不剥离 Markdown、注记或首尾空格，生成层不改写剧本。
    for i, piece in enumerate(line.split("\t")):
        if i:
            ET.SubElement(run, W + "tab")
        ET.SubElement(run, W + "t", {"{http://www.w3.org/XML/1998/namespace}space": "preserve"}).text = piece
    return p


def build_docx(markdown: str | Path, output: str | Path, title: str) -> dict:
    source, output = Path(markdown), Path(output)
    if output.resolve() in (source.resolve(), TEMPLATE.resolve()):
        raise ValueError("输出不能覆盖合并稿或范本")
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    if not title.strip() or not any(line.strip() for line in lines):
        raise ValueError("标题和合并稿不能为空")
    if "\n" in title or "\r" in title:
        raise ValueError("标题必须为单行")
    parts = read_parts(TEMPLATE)
    layout = template_layout(parts)
    doc = xml(parts["word/document.xml"])
    body = doc.find(W + "body")
    body.clear()
    body.append(paragraph(title, "title", layout["title"]))
    for line in lines:
        if EPISODE.fullmatch(line):
            kind = "episode"
            if text_of(body[-1]).strip():
                ET.SubElement(body, W + "p")
            blank = body[-1]
            ppr = blank.find(W + "pPr")
            if ppr is None:
                ppr = ET.SubElement(blank, W + "pPr")
            spacing = ppr.find(W + "spacing")
            if spacing is not None:
                ppr.remove(spacing)
            index = 1 if ppr.find(W + "pStyle") is not None else 0
            ppr.insert(index, deepcopy(layout["blank_ppr"].find(W + "spacing")))
        elif line.startswith("场景"):
            kind = "scene"
        else:
            kind = "body"
        body.append(paragraph(line, kind, layout[kind]))
    body.append(layout["section"])
    parts["word/document.xml"] = serialize(doc)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    return {"通过": True, "输出": str(output), "段落数": len(body) - 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按范本把合并稿生成 Word")
    parser.add_argument("合并稿")
    parser.add_argument("输出")
    parser.add_argument("标题")
    args = parser.parse_args(argv)
    try:
        result = build_docx(args.合并稿, args.输出, args.标题)
    except (OSError, ValueError, KeyError, ET.XMLSyntaxError) as exc:
        result = {"通过": False, "错误": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["通过"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
