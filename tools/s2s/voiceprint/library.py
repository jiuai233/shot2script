"""按确认归属积累角色声纹均值，向新集提供前三候选。"""

from collections import defaultdict
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np

from s2s.config import PROJECT_ROOT
from s2s.diarize.embeddings import Embedder, MODEL_FILES, decode_audio, window_embedding, unit, SAMPLE_RATE
from s2s.diarize.pipeline import validate_records
from s2s.timefmt import to_sec


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value, private=False):
    path = Path(path)
    if path.exists():
        read(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if private:
        path.parent.chmod(0o700)
    # 生物特征文件从创建开始就只允许当前用户读取。
    import os
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600 if private else 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        if private:
            os.fchmod(file.fileno(), 0o600)
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def model_info(model):
    return {"模型": model, "版本": MODEL_FILES[model] + "@sherpa-onnx-" + version("sherpa-onnx"),
            "预处理": "原声16k单声道/3秒等分窗/单位向量句均值"}


def episode_path(drama, episode):
    if isinstance(episode, int) or str(episode).isdigit():
        episode = f"E{int(episode):02d}"
    if not re.fullmatch(r"E\d{2,}", str(episode)):
        raise ValueError("集号须为 E01 或正整数")
    return Path(drama) / str(episode)


def extract(episode_dir, source=None, model="campp"):
    episode_dir = Path(episode_dir)
    records = read(episode_dir / "声音记录.json")
    validate_records(records)
    if source is None:
        table = episode_dir / "镜头表.json"
        if not table.exists():
            raise ValueError("缺少镜头表，请用 --音频 指定本集音视频")
        source = Path(read(table)["视频"])
        if not source.is_absolute():
            source = PROJECT_ROOT / source
    samples = decode_audio(source)[:, 0]
    embedder = Embedder(model)
    vectors = []
    for row in records:
        start, stop = to_sec(row["起"]), to_sec(row["止"])
        if stop > len(samples) / SAMPLE_RATE + 0.05:
            raise ValueError(f"句时间超出音频：{row['id']}")
        clip = samples[round(start * SAMPLE_RATE):round(stop * SAMPLE_RATE)]
        vectors.append(window_embedding(clip, embedder))
    return records, vectors, model_info(model)


def checked_vectors(records, vectors):
    validate_records(records)
    if len(records) != len(vectors):
        raise ValueError("声纹数与声音记录句数不一致")
    dimensions = {len(v) for v in vectors if v is not None}
    if len(dimensions) > 1:
        raise ValueError("声纹向量维度不一致")
    for v in vectors:
        if v is not None and unit(v) is None:
            raise ValueError("无效声纹向量")


def enroll(drama, episode, source=None, model="campp", *, extracted=None):
    directory = episode_path(drama, episode)
    records, vectors, info = extracted if extracted is not None else extract(directory, source, model)
    checked_vectors(records, vectors)
    if records and any(not row["id"].startswith(directory.name + "-") for row in records):
        raise ValueError("声音记录与本集目录不匹配")
    assignments = read(directory / "归属表.json")
    lookup = {}
    for row in assignments:
        if row["句"] in lookup:
            raise ValueError("归属表句号重复")
        lookup[row["句"]] = row
    if set(lookup) - {row["id"] for row in records}:
        raise ValueError("归属表包含当前声音记录之外的句号")
    by_role = defaultdict(list)
    skipped = []
    for row, vector in zip(records, vectors):
        owner = lookup.get(row["id"], {})
        if not re.fullmatch(r"C\d{3,}", owner.get("说话人", "")) or owner.get("确定度") != "确认" or vector is None or row.get("听辨") != "清楚":
            skipped.append(row["id"])
            continue
        by_role[owner["说话人"]].append(vector)
    library = Path(drama) / "声纹"
    # 同一集重入库替换贡献；本集曾入库但现已无确认句的角色也撤去旧贡献。
    old_files = {p.stem: p for p in library.glob("C*.json")} if library.exists() else {}
    planned = []
    for role in sorted(set(old_files) | set(by_role)):
        path = library / (role + ".json")
        old = read(path) if path.exists() else None
        if old and any(old.get(k) != info.get(k) for k in ("模型", "版本", "预处理")):
            raise ValueError("声纹库模型、版本或预处理不一致；须使用独立声纹库")
        sources = [s for s in old.get("来源", []) if s["集"] != directory.name] if old else []
        if by_role[role]:
            mean = np.mean(by_role[role], axis=0)
            if unit(mean) is None:
                raise ValueError(f"角色 {role} 的声纹均值无效")
            sources.append({"集": directory.name, "句数": len(by_role[role]), "均值": mean.tolist()})
        count = sum(s["句数"] for s in sources)
        mean = np.average([s["均值"] for s in sources], axis=0, weights=[s["句数"] for s in sources]) if count else None
        if count and unit(mean) is None:
            raise ValueError(f"角色 {role} 的跨集均值无效")
        data = {"角色": role, **info, "向量": mean.tolist() if count else [], "句数": count,
                "来源集": sorted(s["集"] for s in sources), "来源": sorted(sources, key=lambda s: s["集"])}
        planned.append((path, data))
    for path, data in planned:
        save(path, data, private=True)
    return {"集": directory.name, "角色": [{"角色": role, "本集句数": len(rows)} for role, rows in sorted(by_role.items())],
            "跳过句": skipped, "声纹目录": str(library)}


def compare(drama, episode, source=None, model="campp", *, extracted=None):
    directory = episode_path(drama, episode)
    records, vectors, info = extracted if extracted is not None else extract(directory, source, model)
    checked_vectors(records, vectors)
    if any(not row["id"].startswith(directory.name + "-") for row in records):
        raise ValueError("声音记录与本集目录不匹配")
    library = []
    for path in sorted((Path(drama) / "声纹").glob("C*.json")):
        role = read(path)
        if any(role.get(k) != info.get(k) for k in ("模型", "版本", "预处理")):
            raise ValueError("声纹库模型、版本或预处理不一致")
        if role["句数"]:
            center = unit(role["向量"])
            if center is None:
                raise ValueError("声纹库包含无效中心")
            library.append((role["角色"], center))
    excluded = set()
    result_path = directory / "声纹聚类.json"
    if result_path.exists():
        result = read(result_path)
        if [r["句"] for r in result["句列表"]] != [r["id"] for r in records]:
            raise ValueError("声纹聚类句号已过期，须重新运行聚类")
        for pred, row in zip(result["句列表"], records):
            if any(pred.get(k) != row.get(k) for k in ("起", "止", "文本")) or (not pred["待定"] and pred["声线"] != row["声线"]):
                raise ValueError("声纹聚类输入已变化，须重新运行聚类")
        excluded = {r["句"] for r in result["句列表"] if r["待定"]}
    groups = defaultdict(list)
    for row, vector in zip(records, vectors):
        if vector is not None and row["id"] not in excluded:
            groups[row["声线"]].append(vector)
    output = {"集": directory.name, **info, "仅供参考": True, "声线": []}
    for label in dict.fromkeys(row["声线"] for row in records):
        rows = groups.get(label, [])
        center = unit(np.mean(rows, axis=0)) if rows else None
        candidates = []
        if center is not None:
            for role, vector in library:
                if vector.shape != center.shape:
                    raise ValueError("声纹库与当前模型维度不一致")
                candidates.append({"角色": role, "相似度": round(float(np.clip(center @ vector, -1, 1)), 6)})
            candidates.sort(key=lambda r: (-r["相似度"], r["角色"]))
        output["声线"].append({"声线": label, "句数": len(rows), "候选": candidates[:3]})
    save(directory / "声纹候选.json", output)
    return output
