"""Region extraction and IoU tracking of anomalous regions across frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


def extract_regions(heatmap: np.ndarray, threshold: float = 1.0, min_area_frac: float = 0.002) -> list[dict[str, Any]]:
    """Connected components of heatmap >= threshold. bbox is normalised (x, y, w, h)."""
    H, W = heatmap.shape
    mask = (heatmap >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area_frac * H * W:
            continue
        vals = heatmap[labels == i]
        regions.append({
            "bbox": [x / W, y / H, w / W, h / H],
            "area_frac": float(area / (H * W)),
            "peak": float(vals.max()),
            "mean": float(vals.mean()),
            "centroid": [float(cents[i][0] / W), float(cents[i][1] / H)],
            "_label": i,
        })
    regions.sort(key=lambda r: -r["peak"])
    return regions, labels


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    bbox: list[float]
    first_ts: float
    last_ts: float
    first_frame: int
    last_frame: int
    hits: int = 1
    misses: int = 0
    peak: float = 0.0
    unc_sum: float = 0.0
    confirmed: bool = False
    event_id: int | None = None
    history: list[tuple[float, float, float]] = field(default_factory=list)  # ts, cx, cy
    explanation: dict[str, Any] | None = None
    models: set[str] = field(default_factory=set)

    @property
    def age_s(self) -> float:
        return self.last_ts - self.first_ts

    @property
    def speed(self) -> float:
        """Centroid speed in frame-widths per second over the recent history."""
        h = self.history[-10:]
        if len(h) < 2 or h[-1][0] - h[0][0] <= 0:
            return 0.0
        d = np.hypot(h[-1][1] - h[0][1], h[-1][2] - h[0][2])
        return float(d / (h[-1][0] - h[0][0]))

    @property
    def mean_uncertainty(self) -> float:
        return self.unc_sum / max(1, self.hits)


class RegionTracker:
    def __init__(self, min_hits: int = 3, max_misses: int = 6, iou_thr: float = 0.15, dist_thr: float = 0.12):
        self.min_hits, self.max_misses = min_hits, max_misses
        self.iou_thr, self.dist_thr = iou_thr, dist_thr
        self.tracks: list[Track] = []
        self._next = 1

    def update(self, regions: list[dict[str, Any]], ts: float, frame_idx: int, uncertainty: float):
        """Match regions to tracks. Returns (assignments region_idx->track, newly_confirmed, ended)."""
        pairs = []
        for ti, t in enumerate(self.tracks):
            for ri, r in enumerate(regions):
                s = iou(t.bbox, r["bbox"])
                if s < self.iou_thr:
                    c = t.history[-1] if t.history else (0, t.bbox[0] + t.bbox[2] / 2, t.bbox[1] + t.bbox[3] / 2)
                    d = np.hypot(c[1] - r["centroid"][0], c[2] - r["centroid"][1])
                    s = (1 - d / self.dist_thr) * self.iou_thr if d < self.dist_thr else 0.0
                if s > 0:
                    pairs.append((s, ti, ri))
        pairs.sort(reverse=True)
        used_t, used_r, assign = set(), set(), {}
        for _, ti, ri in pairs:
            if ti in used_t or ri in used_r:
                continue
            used_t.add(ti)
            used_r.add(ri)
            assign[ri] = self.tracks[ti]
        for ri, r in enumerate(regions):
            if ri not in used_r:
                t = Track(self._next, r["bbox"], ts, ts, frame_idx, frame_idx)
                self._next += 1
                self.tracks.append(t)
                assign[ri] = t
                t.hits = 0  # incremented below
        confirmed, ended = [], []
        for ri, t in assign.items():
            r = regions[ri]
            t.bbox, t.last_ts, t.last_frame = r["bbox"], ts, frame_idx
            t.hits += 1
            t.misses = 0
            t.peak = max(t.peak, r["peak"])
            t.unc_sum += uncertainty
            t.history.append((ts, r["centroid"][0], r["centroid"][1]))
            if len(t.history) > 50:
                t.history = t.history[-50:]
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
                confirmed.append(t)
        assigned = set(id(t) for t in assign.values())
        keep = []
        for t in self.tracks:
            if id(t) not in assigned:
                t.misses += 1
            if t.misses > self.max_misses:
                if t.confirmed:
                    ended.append(t)
            else:
                keep.append(t)
        self.tracks = keep
        return assign, confirmed, ended

    def flush(self) -> list[Track]:
        ended = [t for t in self.tracks if t.confirmed]
        self.tracks = []
        return ended
