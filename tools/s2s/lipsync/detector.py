"""YuNet 补检、局部唇点重检，以及无唇点侧脸的嘴部像素证据。"""

from types import SimpleNamespace

import cv2
import numpy as np

from .core import face_measurement


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    extent = np.maximum(0, np.minimum(a[:2] + a[2:], b[:2] + b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = np.prod(extent)
    return float(intersection / max(np.prod(a[2:]) + np.prod(b[2:]) - intersection, 1e-8))


def patch(image, box, size=(64, 40)):
    h, w = image.shape[:2]
    x, y, bw, bh = box
    x0, y0 = max(0, round(x * w)), max(0, round(y * h))
    x1, y1 = min(w, round((x + bw) * w)), min(h, round((y + bh) * h))
    if x1 <= x0 or y1 <= y0:
        return None
    gray = cv2.cvtColor(cv2.resize(image[y0:y1, x0:x1], size), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0).astype(float) / 255


class Supplement:
    def __init__(self, model, confidence=.6):
        self.detector = cv2.FaceDetectorYN.create(str(model), "", (320, 320), confidence, .3, 5000)

    def detect(self, image, faces, landmarker, params):
        import mediapipe as mp

        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, found = self.detector.detect(image)
        if found is None:
            return faces
        for detection in found:
            x, y, bw, bh = map(float, detection[:4])
            box = [x / w, y / h, bw / w, bh / h]
            if min(bw, bh) < params.min_face_px or any(iou(box, f["box"]) > .25 for f in faces):
                continue
            x0, y0 = max(0, int(x - .4 * bw)), max(0, int(y - .4 * bh))
            x1, y1 = min(w, int(x + 1.4 * bw)), min(h, int(y + 1.4 * bh))
            crop = cv2.resize(image[y0:y1, x0:x1], (512, 512))
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            candidates = []
            for points, matrix in zip(result.face_landmarks, result.facial_transformation_matrixes):
                mapped = [SimpleNamespace(x=(p.x * (x1 - x0) + x0) / w,
                                          y=(p.y * (y1 - y0) + y0) / h,
                                          z=p.z * (x1 - x0) / w) for p in points]
                face = face_measurement(mapped, matrix, image, params)
                if face["valid"] and iou(box, face["box"]) > .25:
                    face["source"] = "YuNet局部唇点"
                    candidates.append(face)
            if candidates:
                faces.append(max(candidates, key=lambda f: iou(box, f["box"])))
                continue
            # YuNet 只有嘴角，无内唇点；保留像素变化，不伪造张合比。
            corners = detection[10:14].reshape(2, 2)
            cx, cy = corners.mean(axis=0)
            mw = max(np.linalg.norm(corners[0] - corners[1]) * 1.8, bw * .4)
            mh = max(mw * .65, bh * .15)
            mouth_box = [float(v) for v in ((cx - mw / 2) / w, (cy - mh / 2) / h, mw / w, mh / h)]
            mouth = patch(image, mouth_box)
            reference = patch(image, [(x + .2 * bw) / w, (y + .25 * bh) / h, .6 * bw / w, .2 * bh / h])
            reasons = []
            if mouth is None or min(mw, mh) < 6 or min(mouth_box[:2]) < 0 or mouth_box[0] + mouth_box[2] > 1 or mouth_box[1] + mouth_box[3] > 1:
                reasons.append("嘴部区域出框或过小")
            hist_crop = image[max(0, int(y)):min(h, int(y + bh)), max(0, int(x)):min(w, int(x + bw))]
            hsv = cv2.cvtColor(hist_crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
            hist /= max(hist.sum(), 1)
            faces.append({"box": box, "mar": None, "valid": not reasons, "reason": reasons,
                          "yaw": None, "pitch": None, "feature": None, "appearance": hist,
                          "source": "YuNet嘴部像素", "mouth_box": mouth_box,
                          "_mouth": mouth, "_reference": reference})
        return faces


def add_pixel_motion(tracks):
    """嘴部变化扣除上脸变化；头动仍可残留，不能当作真实唇形。"""
    for track in tracks:
        previous = None
        for sample in track["samples"]:
            if sample.get("source") != "YuNet嘴部像素":
                previous = None
                continue
            sample["mouth_motion"] = None
            if previous is not None and sample["valid"] and previous["valid"]:
                mouth = np.mean(np.abs(sample["_mouth"] - previous["_mouth"]))
                reference = np.mean(np.abs(sample["_reference"] - previous["_reference"]))
                sample["mouth_motion"] = float(max(0, mouth - reference))
            previous = sample
        for sample in track["samples"]:
            sample.pop("_mouth", None)
            sample.pop("_reference", None)
