"""Dataset loaders: MVTec AD 2 (still-image anomalies) and MSAD (video anomalies).

MVTec AD 2 expected layout (as distributed):
    <root>/<object>/train/good/*.png
    <root>/<object>/validation/good/*.png
    <root>/<object>/test_public/good/*.png
    <root>/<object>/test_public/bad/*.png
    <root>/<object>/test_public/ground_truth/bad/*_mask.png
Only test_public has public ground truth; test_private / test_private_mixed are scored by the
official evaluation server and are not used here.

MSAD (Multi-Scenario Anomaly Detection) — the loader expects:
    <root>/train/**/<video>.(mp4|avi|...)    normal training videos
    <root>/test/**/<video>.(mp4|avi|...)     test videos
    labels: one of
      * a JSON file {"<video stem>": [[start_frame, end_frame], ...], ...}  (inclusive, native fps)
      * a directory with <video stem>.npy / .txt holding one 0/1 label per native frame
If your copy of MSAD ships annotations in another format, convert them to one of these first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .scenes import IMAGE_EXT, VIDEO_EXT, read_rgb

MVTEC_AD2_OBJECTS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMAGE_EXT) if d.is_dir() else []


def load_resized(path: Path, max_side: int = 512) -> np.ndarray:
    im = read_rgb(path)
    h, w = im.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        im = cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return im


@dataclass
class MVTecSample:
    image: Path
    mask: Path | None
    label: int  # 0 good, 1 bad


class MVTecAD2:
    def __init__(self, root: str | Path, category: str):
        self.root = Path(root) / category
        if not self.root.is_dir():
            raise FileNotFoundError(f"MVTec AD 2 category not found: {self.root}")
        self.category = category

    @staticmethod
    def categories(root: str | Path) -> list[str]:
        return sorted(p.name for p in Path(root).iterdir() if (p / "train").is_dir())

    def train(self) -> list[Path]:
        return _images(self.root / "train" / "good")

    def validation(self) -> list[Path]:
        return _images(self.root / "validation" / "good")

    def test(self) -> list[MVTecSample]:
        t = self.root / "test_public"
        out = [MVTecSample(p, None, 0) for p in _images(t / "good")]
        gt_dir = t / "ground_truth" / "bad"
        masks = {p.stem: p for p in _images(gt_dir)}
        for p in _images(t / "bad"):
            m = masks.get(f"{p.stem}_mask") or masks.get(p.stem) or next(
                (v for k, v in masks.items() if k.startswith(p.stem)), None)
            out.append(MVTecSample(p, m, 1))
        return out


def read_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, bool)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 127


def _videos(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in VIDEO_EXT) if d.is_dir() else []


class MSAD:
    def __init__(self, root: str | Path, labels: str | Path | None = None):
        self.root = Path(root)
        self.labels_src = Path(labels) if labels else None
        if not (self.root / "train").is_dir():
            raise FileNotFoundError(f"MSAD train/ folder not found under {self.root}")
        self._json: dict[str, list[list[int]]] | None = None
        if self.labels_src and self.labels_src.is_file():
            self._json = json.loads(self.labels_src.read_text())

    def train_videos(self) -> list[Path]:
        return _videos(self.root / "train")

    def test_videos(self) -> list[Path]:
        return _videos(self.root / "test")

    def frame_labels(self, video: Path, n_frames: int) -> np.ndarray:
        y = np.zeros(n_frames, np.uint8)
        if self._json is not None:
            for a, b in self._json.get(video.stem, []):
                y[max(0, a) : min(n_frames, b + 1)] = 1
            return y
        d = self.labels_src or (self.root / "test_labels")
        for ext in (".npy", ".txt"):
            p = d / f"{video.stem}{ext}"
            if p.exists():
                arr = np.load(p) if ext == ".npy" else np.loadtxt(p)
                arr = np.asarray(arr).astype(np.uint8).ravel()
                y[: min(n_frames, len(arr))] = arr[:n_frames]
                return y
        raise FileNotFoundError(f"no frame labels for {video.stem} (looked in {d})")
