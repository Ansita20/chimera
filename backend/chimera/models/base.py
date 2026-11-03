"""Shared detector contract, calibration and uncertainty handling.

Every detector produces a *raw* anomaly map whose units differ per model (reconstruction error,
patch distance, prediction error). Calibration on held-out normal data maps raw values onto a
common scale where 0 ~= typical normal and 1.0 == the decision threshold, so maps from different
models can be fused, compared and explained together.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..config import Settings, resolve_device

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


def _noop(_: float, __: str) -> None:
    pass


@dataclass
class Detection:
    model: str
    score: float  # calibrated image-level score; >= 1.0 means anomalous
    raw_score: float
    heatmap: np.ndarray  # (H, W) float32 calibrated map; >= 1.0 means anomalous pixel
    uncertainty: float  # 0..1 combined uncertainty
    epistemic: float  # 0..1 model uncertainty (MC dropout / memory-bank disagreement)
    ambiguity: float  # 0..1 decision ambiguity (score close to threshold)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def anomalous(self) -> bool:
        return self.score >= 1.0

    def summary(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "raw_score": round(self.raw_score, 6),
            "uncertainty": round(self.uncertainty, 4),
            "epistemic": round(self.epistemic, 4),
            "ambiguity": round(self.ambiguity, 4),
            "anomalous": self.anomalous,
        }


def rgb_to_tensor(images: list[np.ndarray], size: int, normalize: bool = False) -> torch.Tensor:
    """List of HxWx3 uint8 RGB images -> (N, 3, size, size) float tensor in [0, 1]."""
    arr = np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) for im in images])
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(xs**2) / (2 * sigma**2))
    return k / k.sum()


def blur_maps(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Separable gaussian blur of (N, H, W) maps."""
    if sigma <= 0:
        return maps
    k = gaussian_kernel(sigma, maps.device)
    r = k.numel() // 2
    x = maps.unsqueeze(1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x.squeeze(1)


def cv_to_unit(cv: np.ndarray | float, scale: float = 0.15) -> np.ndarray | float:
    """Coefficient of variation -> 0..1 epistemic uncertainty."""
    return 1.0 - np.exp(-np.asarray(cv) / scale)


def decision_ambiguity(score: float, width: float = 0.35) -> float:
    """High when the calibrated score sits near the decision threshold (1.0)."""
    return float(math.exp(-(((score - 1.0) / width) ** 2)))


def combine_uncertainty(epistemic: float, ambiguity: float) -> float:
    return float(np.clip(0.6 * epistemic + 0.4 * ambiguity, 0.0, 1.0))


class AnomalyDetector(ABC):
    """Base class: subclasses implement preprocessing, training and raw map inference."""

    name: str = "base"
    title: str = "Base detector"

    def __init__(self, settings: Settings):
        self.s = settings
        self.device = resolve_device(settings.device)
        self.calib: dict[str, float] | None = None
        self.meta: dict[str, Any] = {}

    # ---- subclass contract -------------------------------------------------
    @abstractmethod
    def preprocess(self, samples: list[Any]) -> torch.Tensor: ...

    @abstractmethod
    def _train(self, x: torch.Tensor, progress: Progress) -> None: ...

    @abstractmethod
    def _raw_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, np.ndarray, list[dict[str, Any]]]:
        """Return (N, H, W) raw maps, (N,) epistemic in 0..1 and per-sample extras."""

    @abstractmethod
    def state(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None: ...

    # ---- training / calibration --------------------------------------------
    @property
    def fitted(self) -> bool:
        return self.calib is not None

    def fit(self, samples: list[Any], progress: Progress | None = None, seed: int = 0) -> dict[str, Any]:
        progress = progress or _noop
        if len(samples) < 4:
            raise ValueError(f"{self.name}: need at least 4 normal samples, got {len(samples)}")
        x = self.preprocess(samples)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(x), generator=g)
        n_cal = max(2, int(round(len(x) * self.s.calib_fraction)))
        cal, train = x[perm[:n_cal]], x[perm[n_cal:]]
        self._train(train, lambda f, m: progress(0.9 * f, m))
        progress(0.92, "calibrating on held-out normal samples")
        self.calibrate(cal)
        self.meta.update(n_train=len(train), n_calib=len(cal))
        progress(1.0, "done")
        return dict(self.meta)

    def _batched_raw(self, x: torch.Tensor, batch: int = 8):
        maps, epi, extras = [], [], []
        for i in range(0, len(x), batch):
            m, e, ex = self._raw_maps(x[i : i + batch])
            maps.append(m.float().cpu())
            epi.append(e)
            extras.extend(ex)
        return torch.cat(maps), np.concatenate(epi), extras

    def calibrate(self, x: torch.Tensor) -> None:
        maps, _, _ = self._batched_raw(x)
        scores = maps.flatten(1).max(1).values.numpy()
        px = maps.flatten().numpy()
        if px.size > 2_000_000:
            px = np.random.default_rng(0).choice(px, 2_000_000, replace=False)
        m = self.s.calib_margin
        px_lo, px_q = np.quantile(px, [0.5, 0.999])
        img_lo, img_q = np.quantile(scores, [0.5, 0.99])
        px_hi = px_q + m * (px_q - px_lo)
        img_hi = img_q + m * (img_q - img_lo)
        eps = 1e-6 + 1e-3 * abs(float(px_hi))
        self.calib = {
            "px_lo": float(px_lo),
            "px_hi": float(max(px_hi, px_lo + eps)),
            "img_lo": float(img_lo),
            "img_hi": float(max(img_hi, img_lo + eps)),
            "normal_img_mean": float(scores.mean()),
            "normal_img_std": float(scores.std()),
        }

    # ---- inference ---------------------------------------------------------
    def _normalize(self, maps: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.calib is not None, f"{self.name} is not fitted"
        c = self.calib
        raw_scores = maps.flatten(1).max(1).values.cpu().numpy()
        norm_maps = ((maps - c["px_lo"]) / (c["px_hi"] - c["px_lo"])).clamp_(min=0).cpu().numpy()
        scores = np.maximum((raw_scores - c["img_lo"]) / (c["img_hi"] - c["img_lo"]), 0.0)
        return norm_maps.astype(np.float32), scores, raw_scores

    @torch.no_grad()
    def detect_tensor(self, x: torch.Tensor) -> list[Detection]:
        maps, epi, extras = self._raw_maps(x)
        norm_maps, scores, raw = self._normalize(maps.float())
        out = []
        for i in range(len(x)):
            amb = decision_ambiguity(float(scores[i]))
            e = float(epi[i])
            out.append(
                Detection(
                    model=self.name,
                    score=float(scores[i]),
                    raw_score=float(raw[i]),
                    heatmap=norm_maps[i],
                    uncertainty=combine_uncertainty(e, amb),
                    epistemic=e,
                    ambiguity=amb,
                    extras=extras[i],
                )
            )
        return out

    def predict(self, sample: Any) -> Detection:
        return self.detect_tensor(self.preprocess([sample]))[0]

    def predict_batch(self, samples: list[Any], batch: int = 8) -> list[Detection]:
        out: list[Detection] = []
        for i in range(0, len(samples), batch):
            out.extend(self.detect_tensor(self.preprocess(samples[i : i + batch])))
        return out

    # ---- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"name": self.name, "calib": self.calib, "meta": self.meta, "state": self.state()}, path)

    def load(self, path: Path) -> "AnomalyDetector":
        blob = torch.load(path, map_location=self.device, weights_only=False)
        if blob["name"] != self.name:
            raise ValueError(f"checkpoint is for {blob['name']}, not {self.name}")
        self.calib = blob["calib"]
        self.meta = blob.get("meta", {})
        self.load_state(blob["state"])
        return self


def enable_mc_dropout(module: torch.nn.Module) -> None:
    """Eval mode everywhere except dropout layers (Monte-Carlo dropout)."""
    module.eval()
    for m in module.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()
