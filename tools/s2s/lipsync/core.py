"""口型的几何测量、镜头内追踪与可解释评分。"""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass
class Parameters:
    fps: float = 12
    threshold: float = 0.6
    margin: float = 0.08
    motion_weight: float = 0.65
    sync_weight: float = 0.35
    motion_scale: float = 0.10
    noise_floor: float = 0.02
    min_face_px: float = 40
    max_yaw: float = 50
    max_pitch: float = 40
    confidence: float = 0.35
    max_faces: int = 8
    min_samples: int = 4
    min_baseline_samples: int = 4
    max_gap: float = 0.5
    identity_distance: float = 0.10
    appearance_distance: float = 0.45
    model_weight: float = 0.65
    conflict_gap: float = 0.55
    pixel_motion_scale: float = 0.08

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("参数必须是有限数")
        if not 0 < self.fps <= 60 or not 0 < self.threshold < 1:
            raise ValueError("fps 须在 (0, 60] 内，阈值须在 (0, 1) 内")
        if not 0 <= self.margin < min(self.threshold, 1 - self.threshold):
            raise ValueError("存疑带宽须小于阈值到 0/1 的距离")
        if min(self.motion_weight, self.sync_weight) < 0 or self.motion_weight + self.sync_weight <= 0:
            raise ValueError("评分权重非负且不能全部为零")
        if min(self.motion_scale, self.min_face_px, self.max_gap, self.identity_distance, self.appearance_distance) <= 0:
            raise ValueError("尺度与距离须大于零")
        if not 0 < self.confidence < 1 or not 0 < self.max_yaw < 90 or not 0 < self.max_pitch < 90:
            raise ValueError("置信度或姿态阈值无效")
        if not 0 <= self.model_weight <= 1 or not 0 < self.conflict_gap <= 1 or self.pixel_motion_scale <= 0:
            raise ValueError("模型权重须在 [0,1]，冲突差须在 (0,1]，像素变化尺度须为正")
        if self.min_samples < 3 or self.min_baseline_samples < 3 or self.max_faces < 1:
            raise ValueError("样本数至少为 3，最多人脸数至少为 1")


def frame_indices(start, end, fps, source_fps, frame_count):
    """只选镜头半开区间内真实存在的帧，避免舍入越过切点。"""
    lo = math.ceil(start * source_fps - 1e-7)
    hi = min(frame_count, math.ceil(end * source_fps - 1e-7))
    if fps <= 0 or end <= start:
        raise ValueError("采样窗口或帧率无效")
    return sorted({min(hi - 1, max(lo, round((start + i / fps) * source_fps)))
                   for i in range(math.ceil((end - start) * fps)) if hi > lo})


def face_measurement(points, matrix, image, params):
    import cv2

    h, w = image.shape[:2]
    xyz = np.array([[p.x * w, p.y * h, p.z * w] for p in points])
    xy = xyz[:, :2]
    lower, upper = np.clip(xy.min(axis=0), [0, 0], [w, h]), np.clip(xy.max(axis=0), [0, 0], [w, h])
    box = [lower[0] / w, lower[1] / h, (upper[0] - lower[0]) / w, (upper[1] - lower[1]) / h]
    width = np.linalg.norm(xy[78] - xy[308])
    mar = np.mean([np.linalg.norm(xy[a] - xy[b]) for a, b in [(13, 14), (82, 87), (312, 317)]]) / max(width, 1)
    rotation = np.asarray(matrix)[:3, :3]
    normal = rotation[:, 2] / max(np.linalg.norm(rotation[:, 2]), 1e-8)
    yaw = math.degrees(math.atan2(abs(normal[0]), abs(normal[2])))
    pitch = math.degrees(math.atan2(abs(normal[1]), math.hypot(normal[0], normal[2])))
    mouth = xy[[78, 308, 13, 14]]
    reason = []
    if min(upper - lower) < params.min_face_px:
        reason.append("人脸过小")
    if yaw > params.max_yaw or pitch > params.max_pitch:
        reason.append("侧脸或俯仰过大")
    if np.any(mouth < [1, 1]) or np.any(mouth >= [w - 1, h - 1]) or width < 5:
        reason.append("嘴部出框或不可分辨")
    if not 0 <= mar <= 1:
        reason.append("嘴部几何异常")
    # 用眼距与眼线归一化，身份特征不使用会随说话改变的嘴部点。
    left, right = xy[33], xy[263]
    axis = (right - left) / max(np.linalg.norm(right - left), 1)
    axes = np.array([axis, [-axis[1], axis[0]]])
    stable = xy[[33, 263, 1, 168, 10, 234, 454, 127, 356, 6]]
    feature = ((stable - (left + right) / 2) @ axes.T / max(np.linalg.norm(right - left), 1)).ravel()
    x0, y0 = lower.astype(int)
    x1, y1 = upper.astype(int)
    crop = image[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
    hist /= max(hist.sum(), 1)
    return {"box": box, "mar": float(mar), "valid": not reason, "reason": reason,
            "yaw": yaw, "pitch": pitch, "feature": feature, "appearance": hist}


def box_cost(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ac, bc = a[:2] + a[2:] / 2, b[:2] + b[2:] / 2
    scale = max(np.linalg.norm(a[2:]), np.linalg.norm(b[2:]), 1e-4)
    size = abs(math.log(max(np.prod(a[2:]), 1e-8) / max(np.prod(b[2:]), 1e-8)))
    return float(np.linalg.norm(ac - bc) / scale + .25 * size)


def appearance_distance(a, b):
    return float(np.sqrt(max(0, 1 - np.sqrt(np.asarray(a) * np.asarray(b)).sum())))


def attach_detections(tracks, detections, time, path, shot_id, params):
    """全局排序后的一对一位置/大小匹配；镜头切换由调用方重置轨迹。"""
    candidates = []
    for i, track in enumerate(tracks):
        last = track["samples"][-1]
        if time - last["time"] > params.max_gap:
            continue
        for j, face in enumerate(detections):
            cost = box_cost(last["box"], face["box"])
            if last.get("source", "MediaPipe") != face.get("source", "MediaPipe"):
                continue
            if cost < .8 and appearance_distance(last["appearance"], face["appearance"]) < .7:
                candidates.append((cost, i, j))
    used_tracks, used_faces = set(), set()
    for _, i, j in sorted(candidates):
        if i in used_tracks or j in used_faces:
            continue
        tracks[i]["samples"].append({**detections[j], "time": time, "path": path})
        used_tracks.add(i)
        used_faces.add(j)
    for j, face in enumerate(detections):
        if j not in used_faces:
            tracks.append({"id": f"{shot_id}-F{len(tracks) + 1}", "shot": shot_id,
                           "samples": [{**face, "time": time, "path": path}]})


def group_tracks(tracks, params):
    """跨镜头仅作候选合并，不把几何相似等同于身份确认。"""
    groups = []
    for track in tracks:
        samples = [s for s in track["samples"] if s["valid"] and s.get("feature") is not None]
        if not samples:
            track["group"] = None
            continue
        feature = np.median([s["feature"] for s in samples], axis=0)
        hist = np.mean([s["appearance"] for s in samples], axis=0)
        matches = []
        for group in groups:
            if track["shot"] in group["shots"]:
                continue
            distance = float(np.sqrt(np.mean((feature - np.median(group["features"], axis=0)) ** 2)))
            color = appearance_distance(hist, np.mean(group["appearances"], axis=0))
            if distance < params.identity_distance and color < params.appearance_distance:
                matches.append((distance + .1 * color, group))
        if matches:
            group = min(matches, key=lambda x: x[0])[1]
        else:
            group = {"id": f"G{len(groups) + 1}", "shots": set(), "features": [], "appearances": []}
            groups.append(group)
        group["shots"].add(track["shot"])
        group["features"].append(feature)
        group["appearances"].append(hist)
        track["group"] = group["id"]
    return groups


def spread(values):
    return float(np.percentile(values, 90) - np.percentile(values, 10)) if len(values) >= 3 else 0.0


def score_samples(samples, baseline, envelope, params):
    if any(s.get("source") == "YuNet嘴部像素" for s in samples):
        return score_pixel_samples(samples, baseline, envelope, params)
    usable = [s for s in samples if s["valid"]]
    evidence = {"有效帧": len(usable), "检测帧": len(samples), "静音基线帧": len(baseline),
                "基线来源": "同轨迹无人声时段" if len(baseline) >= params.min_baseline_samples else "固定噪声下限（静音样本不足）"}
    if not usable or len(usable) / max(len(samples), 1) < .5:
        return 0.0, "看不到", evidence
    active = spread([s["mar"] for s in usable])
    base = spread(baseline) if len(baseline) >= params.min_baseline_samples else params.noise_floor
    motion = float(np.clip((active - max(base, params.noise_floor)) / params.motion_scale, 0, 1))
    sync = 0.0
    times = np.array([s["time"] for s in usable])
    mouth = np.array([s["mar"] for s in usable])
    # 允许不超过 0.12 秒的音画偏移，不搜索任意延迟来凑相关性。
    if len(usable) >= params.min_samples and np.std(mouth) > 1e-6:
        for lag in [-.12, 0, .12]:
            volume = np.asarray([envelope(float(t + lag)) for t in times])
            if np.std(volume) > 1e-8:
                sync = max(sync, float(np.corrcoef(mouth, volume)[0, 1]))
    score = (params.motion_weight * motion + params.sync_weight * sync) / (params.motion_weight + params.sync_weight)
    evidence.update({"句内张合幅度": round(active, 5), "静音张合幅度": round(base, 5),
                     "运动分": round(motion, 5), "同步分": round(sync, 5)})
    if len(usable) < params.min_samples:
        status = "存疑"
    elif score >= params.threshold + params.margin:
        status = "开口"
    elif score >= params.threshold - params.margin:
        status = "存疑"
    else:
        status = "未开口"
    return round(float(score), 5), status, evidence


def sentence_conclusion(tracks):
    usable = [t for t in tracks if t["状态"] != "看不到"]
    if not usable:
        return "看不到"
    if any(t["状态"] == "开口" for t in usable):
        return "画内"
    if any(t["状态"] == "存疑" for t in usable):
        return "存疑"
    return "未开口"


def pair_voices(sentences):
    counts = {}
    for row in sentences:
        if not row.get("声线"):
            continue
        seen = {t["轨迹组"] for t in row["轨迹"] if t.get("轨迹组")}
        opened = {t["轨迹组"] for t in row["轨迹"] if t.get("轨迹组") and t["状态"] == "开口"}
        for group in seen:
            bucket = counts.setdefault((row["声线"], group), [0, 0])
            if group in opened:
                bucket[0] += 1
            elif row["结论"] == "画内":
                # 只有别的脸明确开口时才算反例；画外不能证明身份不符。
                bucket[1] += 1
    return [{"声线": voice, "轨迹组": group, "支持句数": n, "反例句数": m}
            for (voice, group), (n, m) in sorted(counts.items(), key=lambda x: (-x[1][0], x[0])) if n]


def score_pixel_samples(samples, baseline, envelope, params):
    usable = [s for s in samples if s['valid'] and s.get('mouth_motion') is not None]
    evidence = {'测量方式': '嘴部像素变化扣除上脸变化，非张合比', '有效帧': len(usable),
                '检测帧': len(samples), '静音基线帧': len(baseline),
                '基线来源': '同轨迹无人声时段' if len(baseline) >= params.min_baseline_samples else '固定像素噪声下限（静音样本不足）'}
    if not any(s['valid'] for s in samples):
        return 0., '看不到', evidence
    if len(usable) < params.min_samples:
        return 0., '存疑', evidence
    active = float(np.percentile([s['mouth_motion'] for s in usable], 90))
    base = float(np.percentile(baseline, 90)) if len(baseline) >= params.min_baseline_samples else .015
    motion = float(np.clip((active - max(base, .015)) / params.pixel_motion_scale, 0, 1))
    signal = np.array([s['mouth_motion'] for s in usable])
    sync = 0.
    if np.std(signal) > 1e-6:
        for lag in [-.12, 0, .12]:
            volume = np.array([envelope(s['time'] + lag) for s in usable])
            if np.std(volume) > 1e-8:
                sync = max(sync, float(np.corrcoef(signal, volume)[0, 1]))
    score = (params.motion_weight * motion + params.sync_weight * sync) / (params.motion_weight + params.sync_weight)
    status = score_status(score, params)
    evidence.update({'嘴部变化P90': round(active, 5), '基线变化P90': round(base, 5),
                     '运动分': round(motion, 5), '同步分': round(sync, 5)})
    return round(score, 5), status, evidence


def score_status(score, params):
    if score >= params.threshold + params.margin:
        return '开口'
    return '存疑' if score >= params.threshold - params.margin else '未开口'


def combine_scores(rule_score, rule_status, model_score, model_frames, mode, params):
    if mode in ('baseline', 'side'):
        return rule_score, rule_status, False
    if model_score is None or model_frames < params.min_samples:
        return rule_score, '看不到' if rule_status == '看不到' else '存疑', False
    if rule_status == '看不到':
        return rule_score, rule_status, False
    if mode == 'model':
        return round(model_score, 5), score_status(model_score, params), False
    fused = (1 - params.model_weight) * rule_score + params.model_weight * model_score
    conflict = abs(rule_score - model_score) >= params.conflict_gap
    status = '存疑' if conflict else score_status(fused, params)
    return round(fused, 5), status, conflict
