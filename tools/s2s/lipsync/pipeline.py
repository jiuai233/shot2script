"""固定取帧、音频包络、CPU 检测、核对页与默认识图复核。"""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import resource
import sys
import re
import tempfile
import time
import wave

import cv2
import numpy as np

from ..audio import extract_audio
from ..frames import FrameReader
from ..timefmt import to_sec
from .core import (Parameters, attach_detections, face_measurement, frame_indices,
                   group_tracks, pair_voices, score_samples, sentence_conclusion, combine_scores)



def _clip_box(box):
    """把归一化 [x, y, width, height] 裁到画面内；None 原样返回。"""
    if box is None:
        return None
    x0, y0 = max(0.0, float(box[0])), max(0.0, float(box[1]))
    x1, y1 = min(1.0, float(box[0]) + float(box[2])), min(1.0, float(box[1]) + float(box[3]))
    return [round(x0, 5), round(y0, 5), round(max(0.0, x1 - x0), 5), round(max(0.0, y1 - y0), 5)]

def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def timecode(seconds):
    # T11 的 12 fps 取帧证据保留毫秒；不修改共享的 0.1 秒时间格式。
    ms = round(seconds * 1000)
    minutes, rest = divmod(ms, 60000)
    return f"{minutes:02d}:{rest / 1000:06.3f}"


def load_inputs(episode, folder):
    result, versions = [], []
    for name in ("镜头表.json", "声音记录.json"):
        path = Path(episode) / name
        payload = path.read_bytes()
        dump_path = folder / name
        dump_path.write_bytes(payload)
        result.append(json.loads(payload))
        versions.append({"源": str(path), "快照": str(dump_path), "字节": len(payload),
                         "源修改时间": datetime.fromtimestamp(path.stat().st_mtime, timezone(timedelta(hours=8))).isoformat(),
                         "sha256": hashlib.sha256(payload).hexdigest()})
    shots, lines = result
    previous = 0.0
    ids = set()
    for shot in shots["镜头"]:
        a, b = to_sec(shot["起"]), to_sec(shot["止"])
        if abs(a - previous) > .001 or b <= a or shot["id"] in ids:
            raise ValueError("镜头必须从零开始、连续、不重叠且编号唯一")
        previous = b
        ids.add(shot["id"])
    if not ids or abs(previous - to_sec(shots["时长"])) > .001:
        raise ValueError("镜头未覆盖整集")
    ids.clear()
    for line in lines:
        if not 0 <= to_sec(line["起"]) < to_sec(line["止"]) <= previous + .05 or line["id"] in ids:
            raise ValueError(f"对白时间或编号无效：{line['id']}")
        ids.add(line["id"])
    dump(folder / "输入版本.json", versions)
    return shots, lines, versions


class Envelope:
    def __init__(self, path):
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError("音量包络要求单声道 16-bit PCM")
            rate = audio.getframerate()
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(float) / 32768
        step = max(1, round(rate * .02))
        self.times = np.arange(math.ceil(len(samples) / step)) * step / rate + .01
        self.volume = np.array([np.sqrt(np.mean(samples[i:i + step] ** 2)) for i in range(0, len(samples), step)])

    def __call__(self, time):
        return float(np.interp(time, self.times, self.volume, left=0, right=0))


def reference_paths(samples, count=3):
    paths = list(dict.fromkeys(s["path"] for s in samples))
    if len(paths) <= count:
        return paths
    return [paths[round(i * (len(paths) - 1) / (count - 1))] for i in range(count)]


def parse_review_answer(answer, conclusion):
    candidates = [answer.strip(), *re.findall(r"```(?:json)?\s*(.*?)```", answer, re.S)]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and parsed.get("画面结论") in ("画内", "未开口", "看不到", "存疑"):
                return {"画面结论": parsed["画面结论"], "与代码一致": parsed["画面结论"] == conclusion}
        except (ValueError, TypeError):
            continue
    return {}


def review_sentence(row, paths, ask=None):
    if row["结论"] != "存疑":
        return None
    if ask is None:
        from ..vision import ask
    paths = list(dict.fromkeys(paths))
    if len(paths) < 4:
        return {"状态": "未调用", "原因": "有效时段不足 4 张不同帧；不重复图片凑数"}
    chosen = [paths[round(i * (len(paths) - 1) / (min(6, len(paths)) - 1))] for i in range(min(6, len(paths)))]
    prompt = (f"核对同一句对白的连续画面。句号：{row['句']}；文本：{row['文本']}；时间：{row['起']}-{row['止']}。"
              "图片已按时间顺序排列。只根据画面回答哪个位置、什么外观的人嘴在连续开合；是否能判断正在说话。"
              "不得只看字幕或剧情推断说话人；静止张嘴不等于嘴在动。看不清、侧脸、遮挡须明确说明。"
              "这些静帧没有附带音频，不能证明口型与声音同步，也不能确认心声。"
              '只返回 JSON：{"画面结论":"画内/未开口/看不到/存疑","依据":"具体画面依据"}。'
              "画内仅代表有连续嘴动候选；证据不足选存疑，不要勉强二选一。")
    # 不传 runner/model 覆盖参数，严格使用项目默认识图配置。
    answer = ask(chosen, prompt)
    result = {"状态": "已复核", "依据帧": chosen, "结果": answer}
    result.update(parse_review_answer(answer, row["结论"]))
    return result


def render_html(record, tracks_by_id):
    cards = []
    for row in record["句"]:
        image = cv2.imread(row["代表帧"])
        if image is None:
            raise ValueError(f"无法读取代表帧：{row['代表帧']}")
        h, w = image.shape[:2]
        t = row["代表时间秒"]
        for item in row["轨迹"]:
            samples = tracks_by_id[item["id"]]["samples"]
            sample = min(samples, key=lambda s: abs(s["time"] - t))
            if abs(sample["time"] - t) > .11:
                continue
            x, y, bw, bh = sample["box"]
            color = (0, 180, 0) if item["状态"] == "开口" else (0, 150, 255)
            cv2.rectangle(image, (round(x * w), round(y * h)), (round((x + bw) * w), round((y + bh) * h)), color, 2)
            cv2.putText(image, f"{item['id']} {item['说话分数']:.2f}", (round(x * w), max(16, round(y * h) - 5)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
        # 内嵌图片，HTML 单文件离线打开，无外部脚本/图片请求。
        import base64
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise OSError("核对页图片编码失败")
        uri = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        esc = html.escape
        details = "".join(f"<li>{esc(item['id'])} · {esc(str(item.get('轨迹组')))} · {esc(item['状态'])} · {item['说话分数']:.3f}<pre>{esc(json.dumps(item['评分依据'], ensure_ascii=False))}</pre></li>" for item in row["轨迹"])
        versions = "".join(f"<li>{esc(name)}：<strong>{esc(value)}</strong></li>" for name, value in row.get("版本结论", {}).items())
        if versions:
            details = "<li><ul>" + versions + "</ul></li>" + details
        review = f"<details><summary>识图复核</summary><pre>{esc(json.dumps(row['复核'], ensure_ascii=False, indent=2))}</pre></details>" if row["复核"] else ""
        cards.append(f"<article id='{esc(row['句'])}'><h2>{esc(row['句'])} · {esc(row['结论'])}</h2><p>{esc(row['起'])}–{esc(row['止'])} · {esc(row['声线'] or '声线未知')}</p><p>{esc(row['文本'])}</p><img alt='{esc(row['句'])} 的代表帧与人脸框' src='{uri}'><ul>{details or '<li>未检出可用人脸</li>'}</ul>{review}</article>")
    document = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>口型核对</title><style>body{font-family:system-ui;max-width:1100px;margin:24px auto;padding:0 16px;background:#faf9f5;color:#242424}article{border-top:2px solid #aaa;padding:16px 0;display:flow-root}img{max-width:320px;width:38%;float:left;margin:0 24px 12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:12px 0}h2{font-size:1.2rem}@media(max-width:650px){img{float:none;width:100%;max-width:320px}}</style>"
    document += "<h1>口型核对</h1><p>画内表示至少一个镜头中存在开口证据；未开口只是画外或心声候选。分数是启发式指标，不是概率。人脸几何可能在遮挡时仍被估计，必须对照原片核对。</p>"
    document += "<p><!--S2S_STATS--></p>" + "".join(cards) + "</html>"
    return document


def run(video, episode, output, params=None, review=False, model="models/lipsync/face_landmarker.task",
        variant="baseline", yunet="models/lipsync/face_detection_yunet_2023mar.onnx",
        asd_python="envs/asd/bin/python", asd_weights="models/lipsync/Light-ASD/pretrain_AVA_CVPR.model", asd_timeout=1200):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    if variant not in ("baseline", "side", "model", "fused"):
        raise ValueError("版本须为 baseline、side、model、fused")
    params = params or Parameters()
    supplement = None
    if variant != "baseline":
        from .detector import Supplement
        supplement = Supplement(yunet)
    if variant in ("model", "fused"):
        for required in (asd_python, asd_weights):
            if not Path(required).is_file():
                raise FileNotFoundError(f"缺少 ASD 环境或权重：{required}")
    video, output, model = Path(video), Path(output), Path(model)
    html_path = output.parent / "口型核对.html"
    if output.exists() or html_path.exists():
        raise FileExistsError("输出 JSON 或口型核对.html 已存在，请选新的输出目录，不覆盖历史证据")
    if not model.is_file():
        raise FileNotFoundError(f"缺少本地人脸模型：{model}")
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="lipsync_", dir=output.parent))
    (folder / "frames").mkdir()
    shots, lines, versions = load_inputs(episode, folder)
    windows = [(to_sec(row["起"]), to_sec(row["止"])) for row in lines]
    envelope = Envelope(extract_audio(video, folder / "audio.wav"))
    tracks, frame_rows = [], []
    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model), delegate=BaseOptions.Delegate.CPU),
                                    running_mode=RunningMode.IMAGE, num_faces=params.max_faces,
                                    min_face_detection_confidence=params.confidence,
                                    min_face_presence_confidence=params.confidence,
                                    output_facial_transformation_matrixes=True)
    with FrameReader(video) as reader, FaceLandmarker.create_from_options(options) as detector:
        if abs(to_sec(shots["时长"]) - reader.duration) > .1:
            raise ValueError("镜头表时长与视频不符")
        for shot in shots["镜头"]:
            start, end = to_sec(shot["起"]), min(to_sec(shot["止"]), reader.duration)
            expanded = [(max(start, a - .2), min(end, b + .2)) for a, b in windows if b + .2 > start and a - .2 < end]
            if not expanded:
                continue
            # 对白窗口外只补同镜头无人声基线，不将对白扩展跨到下一镜头。
            indices = set()
            for a, b in expanded:
                indices.update(frame_indices(a, b, params.fps, reader.fps, reader.frame_count))
            indices.update(i for i in frame_indices(start, end, params.fps, reader.fps, reader.frame_count)
                           if not any(a - .1 <= i / reader.fps <= b + .1 for a, b in windows))
            shot_tracks = []
            index_rows = []
            for index in sorted(indices):
                timestamp = index / reader.fps
                image = reader.read_index(index)
                path = folder / "frames" / f"{shot['id']}_{index:06d}.jpg"
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                    raise OSError(f"保存帧失败：{path}")
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                faces = [face_measurement(points, matrix, image, params)
                         for points, matrix in zip(result.face_landmarks, result.facial_transformation_matrixes)]
                if supplement is not None:
                    faces = supplement.detect(image, faces, detector, params)
                attach_detections(shot_tracks, faces, timestamp, str(path), shot["id"], params)
                entry = {"time": timestamp, "path": str(path), "shot": shot["id"], "faces": len(faces)}
                frame_rows.append(entry)
                index_rows.append({"文件": path.name, "时间": timecode(timestamp), "原视频帧": index})
            dump(folder / f"{shot['id']}_帧.json", {"镜头": shot["id"], "帧率": params.fps, "帧": index_rows})
            tracks.extend(shot_tracks)
    if supplement is not None:
        from .detector import add_pixel_motion
        add_pixel_motion(tracks)
    group_tracks(tracks, params)
    asd_result = None
    if variant in ("model", "fused"):
        from .asd import run_asd
        asd_result = run_asd(video, folder / "audio.wav", tracks, shots, params.fps, folder, asd_python, asd_weights, timeout=asd_timeout)
    by_id = {t["id"]: t for t in tracks}
    rows = []
    for line in lines:
        start, end = to_sec(line["起"]), to_sec(line["止"])
        shot_ids = [s["id"] for s in shots["镜头"] if to_sec(s["起"]) < end and to_sec(s["止"]) > start]
        row = {"句": line["id"], "起": line["起"], "止": line["止"], "文本": line["文本"],
               "声线": line.get("声线"), "镜头": shot_ids, "轨迹": [], "复核": None}
        for track in tracks:
            selected = [s for s in track["samples"] if start <= s["time"] < end]
            if not selected:
                continue
            key = "mouth_motion" if selected[0].get("source") == "YuNet嘴部像素" else "mar"
            baseline = [s[key] for s in track["samples"] if s["valid"] and s.get(key) is not None and not any(a - .1 <= s["time"] <= b + .1 for a, b in windows)]
            rule_score, rule_status, evidence = score_samples(selected, baseline, envelope, params)
            model_curve = [] if asd_result is None else [s for s in asd_result["tracks"].get(track["id"], {}).get("frames", []) if start <= s["time"] < end]
            model_score = float(np.mean([s["probability"] for s in model_curve])) if model_curve else None
            score, status, conflict = combine_scores(rule_score, rule_status, model_score, len(model_curve), variant, params)
            evidence.update({"规则分数": rule_score, "规则状态": rule_status, "模型有效帧": len(model_curve), "评分冲突": conflict})
            box = np.median([s["box"] for s in selected], axis=0)
            # 人脸贴近画面边缘时检测框会越出画面；输出前裁到画面内。
            x0, y0 = max(0.0, float(box[0])), max(0.0, float(box[1]))
            x1, y1 = min(1.0, float(box[0] + box[2])), min(1.0, float(box[1] + box[3]))
            box = [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]
            row["轨迹"].append({"id": track["id"], "轨迹组": track["group"],
                                  "框": dict(zip(("x", "y", "width", "height"), [round(float(v), 5) for v in box])),
                                  "说话分数": score, "规则分数": rule_score, "模型分数": None if model_score is None else round(model_score, 5),
                                  "模型曲线": [{"时间": timecode(s["time"]), "说话概率": round(s["probability"], 5)} for s in model_curve],
                                  "检测来源": sorted({s.get("source", "MediaPipe") for s in selected}),
                                  "嘴部区域": _clip_box(next((s["mouth_box"] for s in selected if "mouth_box" in s), None)),
                                  "状态": status, "评分依据": evidence,
                                  "不可用原因": sorted({r for s in selected for r in s["reason"]}),
                                  "依据帧": reference_paths(selected),
                                  "曲线": [{"时间": timecode(s["time"]), "张合比": None if s["mar"] is None else round(s["mar"], 5),
                                            "嘴部像素变化": s.get("mouth_motion"), "有效": s["valid"], "音量": round(envelope(s["time"]), 5)} for s in selected]})
        row["结论"] = sentence_conclusion(row["轨迹"])
        active_frames = [f for f in frame_rows if start <= f["time"] < end]
        if not active_frames:
            # 低 fps/极短对白：保留扩展窗口代表图，但不凭句外图下开口结论。
            active_frames = [f for f in frame_rows if f["shot"] in shot_ids and start - .2 <= f["time"] < end + .2]
        if not active_frames:
            raise ValueError(f"对白没有可回查的帧：{line['id']}")
        opened_ids = {t["id"] for t in row["轨迹"] if t["状态"] == "开口"}
        opened_times = {s["time"] for ident in opened_ids for s in by_id[ident]["samples"]}
        representative = max(active_frames, key=lambda f: (f["time"] in opened_times, f["faces"], -abs(f["time"] - (start + end) / 2)))
        row.update({"代表帧": representative["path"], "代表时间秒": representative["time"]})
        if review and row["结论"] == "存疑":
            row["复核"] = review_sentence(row, [f["path"] for f in active_frames])
        rows.append(row)
    serialized_tracks = [{"id": t["id"], "镜头": t["shot"], "轨迹组": t["group"],
                          "样本": [{"时间": timecode(s["time"]), "框": s["box"], "张合比": s["mar"],
                                    "有效": s["valid"], "原因": s["reason"], "yaw": s["yaw"], "pitch": s["pitch"], "文件": s["path"],
                                    "检测来源": s.get("source", "MediaPipe"), "嘴部区域": _clip_box(s.get("mouth_box"))}
                                   for s in t["samples"]]} for t in tracks]
    dump(folder / "轨迹.json", serialized_tracks)
    counts = {name: sum(r["结论"] == name for r in rows) for name in ("画内", "未开口", "看不到", "存疑")}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 ** 2 if sys.platform == "darwin" else 1024)
    record = {"集": shots["集"], "版本": variant,
              "主动说话人模型": None if asd_result is None else {"名称": "Light-ASD", "权重": str(asd_weights),
                  "设备": "cpu", "推理帧率": 25, "子进程耗时秒": round(asd_result["subprocess_seconds"], 3),
                  "子进程内存峰值MiB": round(asd_result["peak_mib"], 2)}, "参数": {"fps": params.fps, "阈值": params.threshold, **asdict(params)},
              "输入版本": versions, "模型": {"文件": str(model), "字节": model.stat().st_size, "sha256": hashlib.sha256(model.read_bytes()).hexdigest()},
              "句": rows, "声线配对": pair_voices(rows),
              "统计": {"句数": len(rows), "结论数量": counts, "取帧数": len(frame_rows), "轨迹数": len(tracks),
                       "进程内存峰值MiB": round(peak_mb, 2)}, "证据目录": str(folder)}
    document = render_html(record, by_id)
    record["统计"]["耗时秒"] = round(time.perf_counter() - started, 3)
    # 核对页构建包含内嵌图，完成后再次记录进程峰值。
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["统计"]["进程内存峰值MiB"] = round(peak / (1024 ** 2 if sys.platform == "darwin" else 1024), 2)
    html_path.write_text(document.replace("<!--S2S_STATS-->", html.escape(json.dumps(record["统计"], ensure_ascii=False))), encoding="utf-8")
    dump(output, record)
    return record


def rescore_existing(video, source, output, variant, asd_python, asd_weights, asd_timeout=1200, asd_source=None):
    """复用侧脸版规则证据，只补模型与融合评分，不重复检测。"""
    from .asd import run_asd
    output = Path(output)
    html_path = output.parent / "口型核对.html"
    if variant not in ("model", "fused"):
        raise ValueError("复用侧脸证据仅支持 model 或 fused")
    if output.exists() or html_path.exists():
        raise FileExistsError("输出已存在，请选新目录")
    record = json.loads(Path(source).read_text())
    if record.get("版本") != "side":
        raise ValueError("复用源须为 side 侧脸版记录")
    started = time.perf_counter()
    params = Parameters(**{k: record["参数"][k] for k in asdict(Parameters())})
    evidence_folder = Path(record["证据目录"])
    serialized = json.loads((evidence_folder / "轨迹.json").read_text())
    tracks = [{"id": t["id"], "shot": t["镜头"], "samples": [
        {"time": to_sec(s["时间"]), "box": s["框"], "valid": s["有效"]} for s in t["样本"]]} for t in serialized]
    shots = json.loads((evidence_folder / "镜头表.json").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    asd_result = run_asd(video, evidence_folder / "audio.wav", tracks, shots, params.fps,
                         output.parent, asd_python, asd_weights, timeout=asd_timeout, result_source=asd_source)
    for row in record["句"]:
        start, end = to_sec(row["起"]), to_sec(row["止"])
        for item in row["轨迹"]:
            curve = [s for s in asd_result["tracks"].get(item["id"], {}).get("frames", []) if start <= s["time"] < end]
            score = float(np.mean([s["probability"] for s in curve])) if curve else None
            rule_score, rule_status = item["规则分数"], item["评分依据"]["规则状态"]
            combined, status, conflict = combine_scores(rule_score, rule_status, score, len(curve), variant, params)
            item.update({"模型分数": None if score is None else round(score, 5), "说话分数": combined, "状态": status,
                         "模型曲线": [{"时间": timecode(s["time"]), "说话概率": round(s["probability"], 5)} for s in curve]})
            item["评分依据"].update({"模型有效帧": len(curve), "评分冲突": conflict})
        row["结论"] = sentence_conclusion(row["轨迹"])
        row["复核"] = None
    record["版本"] = variant
    record["复用侧脸记录"] = str(source)
    record["声线配对"] = pair_voices(record["句"])
    record["主动说话人模型"] = {"名称": "Light-ASD", "权重": str(asd_weights), "设备": "cpu", "推理帧率": 25,
        "子进程耗时秒": round(asd_result["subprocess_seconds"], 3), "子进程内存峰值MiB": round(asd_result["peak_mib"], 2),
        "复用模型输出": asd_result.get("reused_from"), "分段计时": asd_result.get("decoding")}
    record["统计"]["结论数量"] = {name: sum(r["结论"] == name for r in record["句"]) for name in ("画内", "未开口", "看不到", "存疑")}
    document = render_html(record, {t["id"]: t for t in tracks})
    incremental = time.perf_counter() - started
    record["统计"]["复用前流程耗时秒"] = record["统计"]["耗时秒"]
    record["统计"]["本次增量耗时秒"] = round(incremental, 3)
    reused_seconds = asd_result.get("original_subprocess_seconds", 0.)
    record["统计"]["复用模型生成耗时秒"] = round(reused_seconds, 3)
    record["统计"]["耗时秒"] = round(record["统计"]["耗时秒"] + reused_seconds + incremental, 3)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    record["统计"]["本次主进程峰值MiB"] = round(peak, 2)
    record["统计"]["进程内存峰值MiB"] = max(record["统计"]["进程内存峰值MiB"], round(peak, 2))
    html_path.write_text(document.replace("<!--S2S_STATS-->", html.escape(json.dumps(record["统计"], ensure_ascii=False))))
    dump(output, record)
    return record


def review_existing(source, output, video=None):
    output = Path(output)
    html_path = output.parent / '口型核对.html'
    if output.exists() or html_path.exists():
        raise FileExistsError('复核输出已存在，请选择新目录')
    record = json.loads(Path(source).read_text())
    tracks = json.loads((Path(record['证据目录']) / '轨迹.json').read_text())
    by_id = {t['id']: {'samples': [{'time': to_sec(s['时间']), 'box': s['框']} for s in t['样本']]} for t in tracks}
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reused, called = 0, 0
    for row in record['句']:
        if row['结论'] != '存疑':
            continue
        existing = row.get('复核')
        if existing and existing.get('状态') == '已复核':
            existing.update(parse_review_answer(existing['结果'], row['结论']))
            reused += 1
            continue
        # 句内所有轨迹都漏检时也可用固定流程保留的原帧复核。
        candidates = []
        for index in Path(record['证据目录']).glob('*_帧.json'):
            entries = json.loads(index.read_text())['帧']
            candidates.extend((to_sec(f['时间']), str(Path(record['证据目录']) / 'frames' / f['文件'])) for f in entries
                              if to_sec(row['起']) <= to_sec(f['时间']) < to_sec(row['止']))
        supplemental = []
        if len(candidates) < 4 and video is not None:
            with FrameReader(video) as reader:
                indices = frame_indices(to_sec(row['起']), to_sec(row['止']), reader.fps, reader.fps, reader.frame_count)
                folder = output.parent / '复核补帧' / row['句']
                folder.mkdir(parents=True, exist_ok=True)
                for index in indices:
                    path = folder / f'{index:06d}.jpg'
                    if not cv2.imwrite(str(path), reader.read_index(index)):
                        raise OSError(f'复核补帧保存失败：{path}')
                    supplemental.append({'时间': timecode(index / reader.fps), '文件': str(path), '原视频帧': index})
            dump(folder / '帧.json', {'句': row['句'], '帧': supplemental})
            candidates = [(to_sec(f['时间']), f['文件']) for f in supplemental]
        try:
            row['复核'] = review_sentence(row, [p for _, p in sorted(candidates)])
            if row['复核'].get('状态') == '已复核':
                called += 1
            if supplemental:
                row['复核']['补帧索引'] = str(folder / '帧.json')
        except RuntimeError as exc:
            row['复核'] = {'状态': '失败', '原因': str(exc)}
            dump(output, record)
            raise
        dump(output, record)
        print(f"{row['句']}：{row['复核']['状态']}", flush=True)
    reviews = [r['复核'] for r in record['句'] if r['结论'] == '存疑']
    record['复核统计'] = {'存疑句数': len(reviews), '已复核': sum(r is not None and r.get('状态') == '已复核' for r in reviews),
                            '一致': sum(r is not None and r.get('与代码一致') is True for r in reviews),
                            '不一致': sum(r is not None and r.get('与代码一致') is False for r in reviews),
                            '无法比较': sum(r is None or '与代码一致' not in r for r in reviews),
                            '本次调用': called, '复用已调用': reused,
                            '本次耗时秒': round(time.perf_counter() - started, 3),
                            '耗时秒': round(record.get('复核统计', {}).get('耗时秒', 0) + time.perf_counter() - started, 3)}
    dump(output, record)
    html_path.write_text(render_html(record, by_id).replace('<!--S2S_STATS-->', html.escape(json.dumps(record['复核统计'], ensure_ascii=False))))
    return record
