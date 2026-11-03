"""Synthetic scene: a textured background with a slowly moving bar (normal) vs an inserted blob (anomaly)."""

import cv2
import numpy as np


def frame(t: int, anomaly: bool = False, seed: int = 0, size=(160, 120)) -> np.ndarray:
    w, h = size
    rng = np.random.default_rng(seed + t)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (120 + 40 * np.sin(xx / 6.0) * np.cos(yy / 9.0)).astype(np.float32)
    img = np.dstack([base, base * 0.9 + 10, base * 0.7 + 30])
    x = int((t * 3) % (w - 20))
    img[:, x : x + 8] = (40, 160, 60)
    img += rng.normal(0, 3, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if anomaly:
        cv2.circle(img, (w * 3 // 4, h // 3), 12, (255, 30, 200), -1)
    return img
