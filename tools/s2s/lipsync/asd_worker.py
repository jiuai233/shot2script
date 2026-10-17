"""独立 PyTorch 环境入口；官方 Light-ASD 编码器及分类头的 CPU 推理。"""

import argparse
import json
from pathlib import Path
import resource
import sys
import time
import wave

import cv2
import numpy as np
import python_speech_features
from scipy.ndimage import median_filter
import torch

# 允许在只安装推理依赖的独立环境直接执行。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from s2s.lipsync.light_asd.Model import ASD_Model


def crop_face(image, box):
    h, w = image.shape[:2]
    x, y, bw, bh = np.asarray(box) * [w, h, w, h]
    size = max(bw, bh) / 2
    cx, cy = x + bw / 2, y + bh / 2
    pad = int(size * 1.8) + 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), constant_values=110)
    # 与官方 cropScale=.40 的非对称裁剪一致：下方保留更多嘴部。
    face = padded[int(cy + pad - size):int(cy + pad + size * 1.8),
                  int(cx + pad - size * 1.4):int(cx + pad + size * 1.4)]
    gray = cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2GRAY)
    return gray[56:168, 56:168]


def infer(model, head, visual, audio, timings=None):
    started = time.perf_counter()
    features = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=.025, winstep=.010)
    n = len(visual)
    if len(features) < n * 4:
        features = np.pad(features, ((0, n * 4 - len(features)), (0, 0)), mode="edge")
    features = features[:n * 4]
    if timings is not None:
        timings["mfcc_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = []
    # 6 秒分块限制 3D 卷积内存；不混用跨切镜头。
    with torch.inference_mode():
        for start in range(0, n, 150):
            count = min(150, n - start)
            a = torch.tensor(features[4 * start:4 * (start + count)], dtype=torch.float32)[None]
            v = torch.tensor(visual[start:start + count], dtype=torch.float32)[None]
            out = model.forward_audio_visual_backend(model.forward_audio_frontend(a), model.forward_visual_frontend(v))
            probabilities.extend(torch.softmax(head(out), dim=-1)[:, 1].tolist())
    if timings is not None:
        timings["inference_seconds"] = time.perf_counter() - started
    return probabilities


def track_plan(track, fps, detection_fps):
    samples = track["samples"]
    if not samples:
        return np.array([]), np.empty((0, 4)), []
    times = np.array([s["time"] for s in samples])
    start, end = times[0], min(track["shot_end"], times[-1] + 1 / detection_fps)
    if len(samples) < 3 or end - start < .2:
        return np.array([]), np.empty((0, 4)), []
    boxes = np.array([s["box"] for s in samples])
    ts = start + np.arange(int(np.floor((end - start) * 25))) / 25
    interpolated = np.stack([np.interp(ts, times, boxes[:, i]) for i in range(4)], axis=1)
    interpolated = median_filter(interpolated, size=(min(13, len(ts) // 2 * 2 + 1), 1), mode="nearest")
    indices = [min(int(np.ceil(track["shot_end"] * fps - 1e-7)) - 1, int(round(t * fps))) for t in ts]
    return ts, interpolated, indices


def read_shared(cap, plans):
    # 每个原帧只解码一次；同帧多人分别裁剪，不缓存整集高清原图。
    requests = {}
    frames = {ident: [None] * len(plan[0]) for ident, plan in plans.items()}
    for ident, (_, boxes, indices) in plans.items():
        for position, (box, index) in enumerate(zip(boxes, indices)):
            requests.setdefault(index, []).append((ident, position, box))
    timing = {"read_seconds": 0., "crop_seconds": 0., "decoded_frames": 0, "requested_frames": len(requests)}
    if not requests:
        return frames, timing
    started = time.perf_counter()
    current = min(requests)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current)
    timing["read_seconds"] += time.perf_counter() - started
    for index in sorted(requests):
        started = time.perf_counter()
        while current < index:
            if not cap.grab():
                raise ValueError(f"ASD 原视频跳帧失败：{current}")
            current += 1
            timing["decoded_frames"] += 1
        ok, image = cap.read()
        current += 1
        timing["decoded_frames"] += 1
        timing["read_seconds"] += time.perf_counter() - started
        if not ok:
            raise ValueError(f"ASD 原视频读取失败：{index}")
        started = time.perf_counter()
        for ident, position, box in requests[index]:
            frames[ident][position] = crop_face(image, box)
        timing["crop_seconds"] += time.perf_counter() - started
    return frames, timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--decode-mode", choices=["shared", "seek"], default="shared")
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(4)
    model, head = ASD_Model().eval(), torch.nn.Linear(128, 2).eval()
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k[6:]: v for k, v in weights.items() if k.startswith("model.")}, strict=True)
    head.load_state_dict({k[10:]: v for k, v in weights.items() if k.startswith("lossAV.FC.")}, strict=True)
    manifest = json.loads(Path(args.manifest).read_text())
    with wave.open(manifest["audio"], "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError("ASD 要求 16 kHz 单声道 PCM")
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    cap = cv2.VideoCapture(manifest["video"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    result = {}
    timings = []
    plans = {t["id"]: track_plan(t, fps, manifest["fps"]) for t in manifest["tracks"]}
    shared_frames, decoding = read_shared(cap, plans) if args.decode_mode == "shared" else ({}, None)
    for track in manifest["tracks"]:
        ts, interpolated, indices = plans[track["id"]]
        if not len(ts):
            result[track["id"]] = {"frames": [], "reason": "轨迹不足 0.2 秒或 3 个检测样本"}
            continue
        frames = shared_frames.pop(track["id"], [])
        timing = {"id": track["id"], "read_seconds": 0., "crop_seconds": 0.}
        if args.decode_mode == "seek":
            for index, box in zip(indices, interpolated):
                stage = time.perf_counter()
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                timing["read_seconds"] += time.perf_counter() - stage
                if not ok:
                    raise ValueError(f"ASD 原视频读取失败：{index}")
                stage = time.perf_counter()
                frames.append(crop_face(image, box))
                timing["crop_seconds"] += time.perf_counter() - stage
        offset = round(ts[0] * 16000)
        # MFCC 最后一个窗口需额外 15ms；视频时间不延长。
        count = round((len(frames) / 25 + .015) * 16000)
        clip = audio[offset:offset + count]
        if len(clip) < count:
            clip = np.pad(clip, (0, count - len(clip)))
        probs = infer(model, head, np.asarray(frames), clip, timing)
        timing["frames"] = len(frames)
        timings.append(timing)
        result[track["id"]] = {"frames": [{"time": float(t), "probability": float(p)} for t, p in zip(ts, probs)],
                               "reason": None}
    cap.release()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    Path(args.output).write_text(json.dumps({"tracks": result, "seconds": time.perf_counter() - started,
                                           "peak_mib": peak, "torch": torch.__version__, "timings": timings,
                                           "decoding": decoding, "decode_mode": args.decode_mode,
                                           "weights": str(Path(args.weights).resolve())}, ensure_ascii=False))
    print(json.dumps({"tracks": len(result), "seconds": round(time.perf_counter() - started, 3), "peak_mib": round(peak, 2)}))


if __name__ == "__main__":
    main()
