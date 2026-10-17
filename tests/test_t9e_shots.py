import cv2
import numpy as np
from s2s.shots import flash_cuts


def test_gradual_white_flash_adds_one_cut(tmp_path):
    path = tmp_path / 'flash.mp4'
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 96))
    before = np.zeros((96, 64, 3), dtype=np.uint8)
    before[:, :32] = 100
    after = np.fliplr(before).copy()
    for _ in range(30):
        w.write(before)
    for alpha in np.linspace(0, 1, 6):
        w.write((before * (1-alpha) + 255 * alpha).astype('uint8'))
    for alpha in np.linspace(1, 0, 6):
        w.write((after * (1-alpha) + 255 * alpha).astype('uint8'))
    for _ in range(30):
        w.write(after)
    w.release()
    cuts = flash_cuts(path)
    assert len(cuts) == 1 and 1.1 <= cuts[0] <= 1.3
