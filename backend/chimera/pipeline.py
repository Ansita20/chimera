"""Per-frame analysis: run detectors, fuse maps, extract + track regions, explain, render."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .explain import explain_region
from .models import Detection, TemporalDetector
from .scenes import SceneStore
from .tracking import RegionTracker, Track, extract_regions

WORK_WIDTH = 256


def heatmap_png(heat: np.ndarray, max_width: int = 192) -> str:
    """Calibrated map -> transparent RGBA PNG data URL (alpha ramps in from 0.4× threshold)."""
    h, w = heat.shape
    if w > max_width:
        heat = cv2.resize(heat, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    v = np.clip(heat / 2.0, 0, 1)
    bgr = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (np.clip((heat - 0.4) / 0.6, 0, 1) * 210).astype(np.uint8)
    ok, buf = cv2.imencode(".png", np.dstack([bgr, alpha]))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def jpeg_data_url(rgb: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def render_snapshot(rgb: np.ndarray, heat: np.ndarray, bbox: list[float], path: Path, label: str = "") -> None:
    H, W = rgb.shape[:2]
    big = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap((np.clip(big / 2, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    a = (np.clip((big - 0.4) / 0.6, 0, 1) * 0.55)[..., None]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    out = (bgr * (1 - a) + color.astype(np.float32) * a).astype(np.uint8)
    x, y, w, h = bbox
    p1, p2 = (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H))
    cv2.rectangle(out, p1, p2, (80, 220, 255), 2)
    if label:
        cv2.putText(out, label, (p1[0], max(14, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])


@dataclass
class FrameAnalysis:
    payload: dict[str, Any]
    fused: np.ndarray
    detections: dict[str, Detection]
    confirmed: list[Track] = field(default_factory=list)
    ended: list[Track] = field(default_factory=list)


class FrameAnalyzer:
    """Stateful analyzer for one stream (keeps temporal buffer, tracker and drift statistics)."""

    def __init__(
        self,
        scene: SceneStore,
        models: list[str] | None = None,
        fusion: str = "max",
        threshold: float = 1.0,
        min_hits: int = 3,
        max_misses: int = 6,
        min_region_frac: float = 0.002,
        include_model_maps: bool = False,
        stream: bool = True,
    ):
        self.scene = scene
        avail = scene.available()
        chosen = [m for m in (models or avail) if m in avail]
        if not chosen:
            raise ValueError(f"scene '{scene.id}' has no trained detectors (requested {models}, available {avail})")
        self.detectors = {m: scene.detectors[m] for m in chosen}
        self.fusion, self.threshold = fusion, max(1e-3, threshold)
        self.min_region_frac = min_region_frac
        self.include_model_maps = include_model_maps
        self.stream = stream
        self.tracker = RegionTracker(min_hits=min_hits, max_misses=max_misses)
        t = self.detectors.get("temporal")
        self.temporal_state = t.new_state() if isinstance(t, TemporalDetector) else None
        self.ema: float | None = None
        self.ema_hi_frames = 0
        self.frame_idx = 0

    @property
    def models(self) -> list[str]:
        return list(self.detectors)

    def analyze(self, rgb: np.ndarray, ts: float | None = None) -> FrameAnalysis:
        t0 = time.perf_counter()
        ts = time.time() if ts is None else ts
        dets: dict[str, Detection] = {}
        for name, d in self.detectors.items():
            if isinstance(d, TemporalDetector):
                r = d.step(self.temporal_state, rgb)
                if r is not None:
                    dets[name] = r
            else:
                dets[name] = d.predict(rgb)

        H, W = rgb.shape[:2]
        ww, wh = WORK_WIDTH, max(16, int(round(WORK_WIDTH * H / W)))
        maps = {n: cv2.resize(d.heatmap, (ww, wh), interpolation=cv2.INTER_LINEAR) / self.threshold for n, d in dets.items()}
        if maps:
            stack = np.stack(list(maps.values()))
            fused = stack.max(0) if self.fusion == "max" else stack.mean(0)
            scores = np.array([d.score for d in dets.values()]) / self.threshold
            score = float(scores.max() if self.fusion == "max" else scores.mean())
        else:  # temporal warming up
            fused, score = np.zeros((wh, ww), np.float32), 0.0

        regions, labels = extract_regions(fused, 1.0, self.min_region_frac) if score >= 1.0 else ([], None)
        flags = [d.anomalous for d in dets.values()]
        disagreement = 1.0 if flags and 0 < sum(flags) < len(flags) else 0.0
        unc = float(np.clip(np.mean([d.uncertainty for d in dets.values()]) + 0.2 * disagreement, 0, 1)) if dets else 1.0

        assign, confirmed, ended = self.tracker.update(regions, ts, self.frame_idx, unc)
        out_regions = []
        for ri, r in enumerate(regions):
            mask = labels == r["_label"]
            peaks = {n: float(m[mask].max()) for n, m in maps.items()}
            track = assign.get(ri)
            expl = explain_region(r, peaks, dets, self.scene.profile, rgb, track if self.stream else None)
            if track is not None:
                track.explanation = expl
                track.models |= set(expl["models_flagged"])
            out_regions.append({
                "track_id": track.id if track else None,
                "confirmed": bool(track and track.confirmed),
                "bbox": [round(v, 4) for v in r["bbox"]],
                "area_frac": round(r["area_frac"], 4),
                "peak": round(r["peak"], 3),
                "model_peaks": {k: round(v, 3) for k, v in peaks.items()},
                "age_s": round(track.age_s, 2) if track else 0.0,
                "hits": track.hits if track else 1,
                "explanation": expl,
            })

        # drift: slow EMA of the image score; sustained elevation suggests normality itself changed
        self.ema = score if self.ema is None else 0.97 * self.ema + 0.03 * score
        self.ema_hi_frames = self.ema_hi_frames + 1 if self.ema > 0.8 else 0
        drift_alert = self.ema_hi_frames > 60

        if score >= 1.0 and out_regions:
            verdict = "uncertain" if unc > 0.6 else "anomalous"
        elif score >= 1.0:
            verdict = "uncertain"
        else:
            verdict = "uncertain" if unc > 0.6 and score > 0.7 else "normal"
        if not dets:
            verdict = "warming_up"

        payload: dict[str, Any] = {
            "frame_idx": self.frame_idx,
            "ts": ts,
            "score": round(score, 4),
            "uncertainty": round(unc, 4),
            "verdict": verdict,
            "models": {n: d.summary() for n, d in dets.items()},
            "regions": out_regions,
            "heatmap": heatmap_png(fused),
            "drift": {"ema": round(self.ema, 4), "alert": drift_alert},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        if self.include_model_maps:
            payload["model_heatmaps"] = {n: heatmap_png(m) for n, m in maps.items()}
            if "autoencoder" in dets:
                payload["reconstruction"] = jpeg_data_url(dets["autoencoder"].extras["reconstruction"])
        self.frame_idx += 1
        return FrameAnalysis(payload, fused, dets, confirmed, ended)

    def finish(self) -> list[Track]:
        return self.tracker.flush()
