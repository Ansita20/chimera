"""Temporal detector: future-frame prediction.

A small U-Net sees the last K grayscale frames and predicts the next one (as a residual on the last
frame). Normal motion is predictable; unexpected motion, sudden appearances and abrupt changes are
not, so prediction error localises temporal anomalies.

Complementary failure mode: an unusual object that sits perfectly still becomes predictable after a
few frames, so the temporal model reports *changes* while the appearance models (embedding /
autoencoder) report *static* abnormality. The comparison benchmark measures exactly this.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AnomalyDetector, Detection, Progress, blur_maps, cv_to_unit, enable_mc_dropout


def _cb(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class FramePredictor(nn.Module):
    def __init__(self, k: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.e1, self.e2, self.e3, self.b = _cb(k, 32), _cb(32, 64), _cb(64, 128), _cb(128, 256)
        self.drop = nn.Dropout2d(p_drop)
        self.u3, self.d3 = nn.ConvTranspose2d(256, 128, 2, 2), _cb(256, 128)
        self.u2, self.d2 = nn.ConvTranspose2d(128, 64, 2, 2), _cb(128, 64)
        self.u1, self.d1 = nn.ConvTranspose2d(64, 32, 2, 2), _cb(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.drop(self.b(F.max_pool2d(e3, 2)))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return (x[:, -1:] + torch.tanh(self.out(d1))).clamp(0, 1)


def _grad_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def g(t):
        return (t[..., 1:, :] - t[..., :-1, :]).abs(), (t[..., :, 1:] - t[..., :, :-1]).abs()

    (ay, ax), (by, bx) = g(a), g(b)
    return (ay - by).abs().mean() + (ax - bx).abs().mean()


class TemporalState:
    """Per-stream frame buffer, kept outside the detector so many streams can share one model."""

    def __init__(self, k: int):
        self.frames: deque[np.ndarray] = deque(maxlen=k + 1)


class TemporalDetector(AnomalyDetector):
    """Samples are windows: arrays of shape (K+1, H, W) float32 grayscale in [0, 1]."""

    name = "temporal"
    title = "Future-frame prediction U-Net (temporal)"

    def __init__(self, settings, epochs: int | None = None):
        super().__init__(settings)
        self.size = settings.temporal_size
        self.k = settings.temporal_window
        self.epochs = epochs or settings.temporal_epochs
        self.net = FramePredictor(self.k).to(self.device)

    def frame_to_gray(self, rgb: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        return g.astype(np.float32) / 255.0

    def windows_from_sequences(self, sequences: list[list[np.ndarray]], stride: int = 1) -> list[np.ndarray]:
        wins = []
        for seq in sequences:
            g = [self.frame_to_gray(f) for f in seq]
            for t in range(self.k, len(g), stride):
                wins.append(np.stack(g[t - self.k : t + 1]))
        return wins

    def preprocess(self, samples: list[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack(samples)).float()

    def fit_sequences(self, sequences: list[list[np.ndarray]], progress: Progress | None = None, stride: int = 1):
        wins = self.windows_from_sequences(sequences, stride)
        if len(wins) < 8:
            raise ValueError(
                f"temporal: need contiguous normal frame sequences (>= {self.k + 8} frames); got {len(wins)} windows"
            )
        return self.fit(wins, progress)

    def _train(self, x: torch.Tensor, progress: Progress) -> None:
        net = self.net.train()
        opt = torch.optim.Adam(net.parameters(), lr=2e-4)
        bs = min(8, len(x))
        epochs = max(self.epochs, -(-self.s.temporal_min_steps // -(-len(x) // bs)))
        loss_v = float("nan")
        for ep in range(epochs):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), bs):
                w = x[perm[i : i + bs]].to(self.device)
                inp, tgt = w[:, : self.k], w[:, self.k :]
                pred = net(inp)
                loss = F.mse_loss(pred, tgt) + _grad_loss(pred, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_v = loss.item()
            progress((ep + 1) / epochs, f"epoch {ep + 1}/{epochs} loss={loss_v:.4f}")
        self.meta.update(final_loss=loss_v, epochs=epochs)
        enable_mc_dropout(self.net)

    @torch.no_grad()
    def _raw_maps(self, x: torch.Tensor):
        enable_mc_dropout(self.net)
        x = x.to(self.device)
        inp, tgt = x[:, : self.k], x[:, self.k :]
        T = max(1, self.s.mc_samples // 2)
        maps = torch.stack([blur_maps((self.net(inp) - tgt).abs().squeeze(1), sigma=2.0) for _ in range(T)])
        per_pass = maps.flatten(2).max(2).values
        cv = (per_pass.std(0) / per_pass.mean(0).clamp(min=1e-8)).cpu().numpy() if T > 1 else np.zeros(len(x))
        motion = (tgt - inp[:, -1:]).abs().mean(dim=(1, 2, 3)).cpu().numpy()
        extras = [{"motion_energy": float(m)} for m in motion]
        return maps.mean(0), np.asarray(cv_to_unit(cv, 0.15)), extras

    def new_state(self) -> TemporalState:
        return TemporalState(self.k)

    def step(self, state: TemporalState, rgb: np.ndarray) -> Detection | None:
        """Push a frame; returns a detection once the window is full."""
        state.frames.append(self.frame_to_gray(rgb))
        if len(state.frames) < self.k + 1:
            return None
        return self.detect_tensor(self.preprocess([np.stack(state.frames)]))[0]

    def state(self) -> dict[str, Any]:
        return {"net": self.net.state_dict(), "size": self.size, "k": self.k}

    def load_state(self, state: dict[str, Any]) -> None:
        self.size = state["size"]
        if state["k"] != self.k:
            self.k = state["k"]
            self.net = FramePredictor(self.k).to(self.device)
        self.net.load_state_dict(state["net"])
        enable_mc_dropout(self.net)
