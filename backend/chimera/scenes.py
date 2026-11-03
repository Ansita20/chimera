"""A *scene* is one notion of normality (a camera view, a product type) plus its trained detectors.

Disk layout: data/scenes/<id>/
    normal/images/*.png          still normal images
    normal/seq_<id>/00000.jpg    contiguous normal frame sequences (webcam capture / normal videos)
    models/<detector>.pt
    profile.npz                  per-location colour/texture profile used for explanations
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Settings
from .explain import NormalityProfile
from .models import DETECTORS, AnomalyDetector, TemporalDetector
from .models.base import Progress

log = logging.getLogger(__name__)
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48]
    return s or uuid.uuid4().hex[:8]


def decode_image(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_rgb(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"could not read {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def iter_video(path: Path, target_fps: float = 10.0, max_frames: int | None = None):
    """Yield (frame_idx, ts_seconds, rgb) sampled at roughly target_fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / target_fps)))
    i = n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield i, i / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                n += 1
                if max_frames and n >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


def video_info(path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(path))
    info = {"fps": cap.get(cv2.CAP_PROP_FPS) or 25.0, "frames": cap.get(cv2.CAP_PROP_FRAME_COUNT)}
    cap.release()
    return info


class SceneStore:
    def __init__(self, scene_id: str, settings: Settings):
        self.id, self.s = scene_id, settings
        self.dir = settings.scenes_dir / scene_id
        self.normal_dir = self.dir / "normal"
        self.models_dir = self.dir / "models"
        for d in (self.normal_dir / "images", self.models_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.detectors: dict[str, AnomalyDetector] = {}
        self.profile: NormalityProfile | None = None
        self.reload()

    # ---- models ---------------------------------------------------------------
    def reload(self) -> None:
        dets: dict[str, AnomalyDetector] = {}
        for name, cls in DETECTORS.items():
            p = self.models_dir / f"{name}.pt"
            if p.exists():
                try:
                    dets[name] = cls(self.s).load(p)
                except Exception:
                    log.exception("failed to load %s for scene %s", name, self.id)
        prof = self.dir / "profile.npz"
        with self.lock:
            self.detectors = dets
            self.profile = NormalityProfile.load(prof) if prof.exists() else None

    def available(self) -> list[str]:
        return list(self.detectors)

    # ---- normal data ------------------------------------------------------------
    def add_image(self, data: bytes) -> None:
        img = decode_image(data)
        cv2.imwrite(str(self.normal_dir / "images" / f"{uuid.uuid4().hex[:12]}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def new_sequence(self) -> Path:
        d = self.normal_dir / f"seq_{uuid.uuid4().hex[:8]}"
        d.mkdir(parents=True)
        return d

    def add_sequence_frame(self, seq_dir: Path, idx: int, rgb: np.ndarray) -> None:
        cv2.imwrite(str(seq_dir / f"{idx:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    def add_video(self, path: Path, target_fps: float = 10.0, max_frames: int = 600) -> int:
        seq = self.new_sequence()
        n = 0
        for n, (_, _, rgb) in enumerate(iter_video(path, target_fps, max_frames), start=1):
            self.add_sequence_frame(seq, n - 1, rgb)
        if n == 0:
            shutil.rmtree(seq, ignore_errors=True)
        return n

    def add_upload(self, filename: str, data: bytes, tmp_dir: Path) -> dict[str, int]:
        ext = Path(filename).suffix.lower()
        if ext in VIDEO_EXT:
            tmp = tmp_dir / f"{uuid.uuid4().hex}{ext}"
            tmp.write_bytes(data)
            try:
                return {"frames": self.add_video(tmp)}
            finally:
                tmp.unlink(missing_ok=True)
        self.add_image(data)
        return {"images": 1}

    def normal_stats(self) -> dict[str, Any]:
        images = len(list((self.normal_dir / "images").glob("*")))
        seqs = [d for d in self.normal_dir.glob("seq_*") if d.is_dir()]
        return {"images": images, "sequences": len(seqs), "sequence_frames": sum(len(list(d.glob("*.jpg"))) for d in seqs)}

    def clear_normal(self) -> None:
        shutil.rmtree(self.normal_dir, ignore_errors=True)
        (self.normal_dir / "images").mkdir(parents=True, exist_ok=True)

    def load_normal(self, max_stills: int = 400) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
        stills = [read_rgb(p) for p in sorted((self.normal_dir / "images").glob("*")) if p.suffix.lower() in IMAGE_EXT]
        seqs = [[read_rgb(p) for p in sorted(d.glob("*.jpg"))] for d in sorted(self.normal_dir.glob("seq_*"))]
        seqs = [s for s in seqs if s]
        seq_frames = [f for s in seqs for f in s]
        if seq_frames:  # consecutive frames are near-duplicates: subsample for appearance models
            budget = max(0, max_stills - len(stills))
            if budget and len(seq_frames) > budget:
                idx = np.linspace(0, len(seq_frames) - 1, budget).astype(int)
                seq_frames = [seq_frames[i] for i in idx]
            stills = stills + seq_frames[:budget] if budget else stills
        return stills, seqs

    # ---- training ---------------------------------------------------------------
    def train(self, models: list[str], epochs: int | None = None, progress: Progress | None = None) -> dict[str, Any]:
        progress = progress or (lambda f, m: None)
        stills, seqs = self.load_normal()
        if not stills:
            raise ValueError("no normal data: upload images/videos or capture from the webcam first")
        n = len(models)
        results: dict[str, Any] = {}
        prof = NormalityProfile.from_images(stills)
        prof.save(self.dir / "profile.npz")
        for i, name in enumerate(models):
            sub = lambda f, m, i=i, name=name: progress((i + f) / n, f"[{name}] {m}")  # noqa: E731
            if name == "temporal":
                det = TemporalDetector(self.s, epochs=epochs)
                if not seqs:
                    results[name] = {"error": "temporal model needs normal video or a webcam capture sequence"}
                    continue
                meta = det.fit_sequences(seqs, sub)
            else:
                det = DETECTORS[name](self.s, epochs=epochs) if name == "autoencoder" else DETECTORS[name](self.s)
                meta = det.fit(stills, sub)
            det.save(self.models_dir / f"{name}.pt")
            results[name] = {**meta, "calibration": det.calib}
        self.reload()
        return results


class SceneRegistry:
    def __init__(self, settings: Settings):
        self.s = settings
        self._scenes: dict[str, SceneStore] = {}
        self._lock = threading.Lock()

    def get(self, scene_id: str) -> SceneStore:
        with self._lock:
            if scene_id not in self._scenes:
                self._scenes[scene_id] = SceneStore(scene_id, self.s)
            return self._scenes[scene_id]

    def delete(self, scene_id: str) -> None:
        with self._lock:
            self._scenes.pop(scene_id, None)
        shutil.rmtree(self.s.scenes_dir / scene_id, ignore_errors=True)
