"""Research comparison: autoencoder vs feature-embedding vs temporal detector.

* MVTec AD 2 (stills): image AUROC, pixel AUROC, AU-PRO@30% FPR, F1 at the calibrated threshold,
  best-F1, training time and per-image latency. The temporal model does not apply to stills.
* MSAD (video): frame-level AUROC (micro over all frames and macro mean over videos), F1 at the
  calibrated threshold, latency. Appearance models score each frame independently; the temporal
  model scores each frame from the preceding window.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .config import Settings
from .datasets import MSAD, MVTecAD2, load_resized, read_mask
from .metrics import au_pro, best_f1, f1_at, pixel_auroc, safe_auroc, smooth
from .models import DETECTORS, AnomalyDetector, TemporalDetector
from .scenes import iter_video, video_info

Progress = Callable[[float, str], None]
EVAL_SIDE = 256


def _sub(progress: Progress, lo: float, hi: float) -> Progress:
    return lambda f, m: progress(lo + (hi - lo) * f, m)


def _build(name: str, s: Settings, epochs: int | None) -> AnomalyDetector:
    cls = DETECTORS[name]
    return cls(s, epochs=epochs) if name in ("autoencoder", "temporal") else cls(s)


def bench_mvtec(root: str, category: str, models: list[str], s: Settings, progress: Progress,
                epochs: int | None = None, max_train: int | None = None) -> dict[str, Any]:
    ds = MVTecAD2(root, category)
    models = [m for m in models if m != "temporal"]
    train_p, val_p, test = ds.train(), ds.validation(), ds.test()
    if max_train:
        train_p = train_p[:max_train]
    progress(0.02, f"loading {len(train_p)} train / {len(val_p)} val / {len(test)} test images")
    train = [load_resized(p) for p in train_p]
    val = [load_resized(p) for p in val_p]
    results: dict[str, Any] = {"_dataset": {"train": len(train), "validation": len(val), "test": len(test)}}
    span = 1.0 / max(1, len(models))
    for mi, name in enumerate(models):
        base = mi * span
        det = _build(name, s, epochs)
        t0 = time.perf_counter()
        det.fit(train, _sub(progress, base, base + 0.6 * span))
        if val:
            det.calibrate(det.preprocess(val))  # official validation split defines the threshold
        train_s = time.perf_counter() - t0
        ys, scores, preds, masks, maps, lat = [], [], [], [], [], []
        for i, smp in enumerate(test):
            im = load_resized(smp.image)
            t1 = time.perf_counter()
            d = det.predict(im)
            lat.append((time.perf_counter() - t1) * 1000)
            h, w = im.shape[:2]
            eh, ew = int(round(EVAL_SIDE * h / max(h, w))), int(round(EVAL_SIDE * w / max(h, w)))
            maps.append(cv2.resize(d.heatmap, (ew, eh), interpolation=cv2.INTER_LINEAR))
            masks.append(read_mask(smp.mask, (eh, ew)))
            ys.append(smp.label)
            scores.append(d.score)
            preds.append(d.anomalous)
            if i % 10 == 0:
                progress(base + span * (0.6 + 0.4 * i / len(test)), f"[{name}] testing {i}/{len(test)}")
        ys, scores = np.array(ys), np.array(scores)
        results[name] = {
            "image_auroc": safe_auroc(ys, scores),
            "pixel_auroc": pixel_auroc(masks, maps),
            "au_pro_30": au_pro(masks, maps),
            "f1_at_threshold": f1_at(ys, np.array(preds)),
            "best_f1": best_f1(ys, scores),
            "train_s": round(train_s, 2),
            "latency_ms": round(float(np.median(lat)), 2) if lat else None,
            "meta": {k: v for k, v in det.meta.items() if isinstance(v, (int, float, str, bool))},
        }
    progress(1.0, "done")
    return results


def _video_frames(path: Path, fps: float, max_frames: int | None = None) -> tuple[list[int], list[np.ndarray]]:
    idx, frames = [], []
    for i, _, rgb in iter_video(path, fps, max_frames):
        idx.append(i)
        frames.append(cv2.resize(rgb, (320, int(round(320 * rgb.shape[0] / rgb.shape[1])))))
    return idx, frames


def bench_msad(root: str, models: list[str], s: Settings, progress: Progress, labels: str | None = None,
               fps: float = 5.0, epochs: int | None = None, max_train_videos: int | None = None,
               max_test_videos: int | None = None, max_frames_per_video: int = 300) -> dict[str, Any]:
    ds = MSAD(root, labels)
    tv, te = ds.train_videos(), ds.test_videos()
    tv, te = tv[:max_train_videos] if max_train_videos else tv, te[:max_test_videos] if max_test_videos else te
    progress(0.01, f"reading {len(tv)} training videos")
    seqs = [_video_frames(p, fps, max_frames_per_video)[1] for p in tv]
    seqs = [q for q in seqs if len(q) > s.temporal_window + 1]
    stills = [f for q in seqs for f in q]
    if len(stills) > 500:
        stills = [stills[i] for i in np.linspace(0, len(stills) - 1, 500).astype(int)]
    results: dict[str, Any] = {"_dataset": {"train_videos": len(seqs), "test_videos": len(te), "fps": fps}}
    dets: dict[str, AnomalyDetector] = {}
    span = 0.5 / max(1, len(models))
    for mi, name in enumerate(models):
        det = _build(name, s, epochs)
        t0 = time.perf_counter()
        sub = _sub(progress, 0.02 + mi * span, 0.02 + (mi + 1) * span)
        if isinstance(det, TemporalDetector):
            det.fit_sequences(seqs, sub)
        else:
            det.fit(stills, sub)
        dets[name] = det
        results[name] = {"train_s": round(time.perf_counter() - t0, 2)}

    per: dict[str, dict[str, list]] = {m: {"y": [], "s": [], "pred": [], "auc": [], "lat": []} for m in models}
    for vi, vp in enumerate(te):
        progress(0.55 + 0.45 * vi / max(1, len(te)), f"testing video {vi + 1}/{len(te)}")
        n_native = int(video_info(vp)["frames"])
        idx, frames = _video_frames(vp, fps)
        if not frames:
            continue
        y_all = ds.frame_labels(vp, max(n_native, idx[-1] + 1))
        y = y_all[idx]
        for name, det in dets.items():
            sc = np.zeros(len(frames))
            t1 = time.perf_counter()
            if isinstance(det, TemporalDetector):
                st = det.new_state()
                for i, f in enumerate(frames):
                    r = det.step(st, f)
                    sc[i] = r.score if r else 0.0
            else:
                sc = np.array([d.score for d in det.predict_batch(frames)])
            per[name]["lat"].append((time.perf_counter() - t1) * 1000 / len(frames))
            sc = smooth(sc, 2.0)
            per[name]["y"].append(y)
            per[name]["s"].append(sc)
            per[name]["pred"].append(sc >= 1.0)
            a = safe_auroc(y, sc)
            if a is not None:
                per[name]["auc"].append(a)
    for name, p in per.items():
        if not p["y"]:
            continue
        y, sc, pr = np.concatenate(p["y"]), np.concatenate(p["s"]), np.concatenate(p["pred"])
        results[name].update({
            "frame_auroc_micro": safe_auroc(y, sc),
            "frame_auroc_macro": float(np.mean(p["auc"])) if p["auc"] else None,
            "f1_at_threshold": f1_at(y, pr),
            "best_f1": best_f1(y, sc),
            "latency_ms": round(float(np.median(p["lat"])), 2),
        })
    progress(1.0, "done")
    return results
