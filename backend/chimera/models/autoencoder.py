"""Convolutional autoencoder: anomalies are regions the model cannot reconstruct.

Anomaly map = 0.5 * L1 error + 0.5 * (1 - SSIM), averaged over Monte-Carlo dropout passes.
Epistemic uncertainty = spread of the image score across those passes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AnomalyDetector, Progress, blur_maps, cv_to_unit, enable_mc_dropout, gaussian_kernel, rgb_to_tensor


def _down(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True))


def _up(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ConvAE(nn.Module):
    def __init__(self, latent: int = 64, p_drop: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(_down(3, 32), _down(32, 64), _down(64, 128), _down(128, 256), nn.Conv2d(256, latent, 1))
        self.drop = nn.Dropout2d(p_drop)
        self.dec = nn.Sequential(
            nn.Conv2d(latent, 256, 1),
            nn.ReLU(inplace=True),
            _up(256, 128),
            _up(128, 64),
            _up(64, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.drop(self.enc(x)))


def _gray(x: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def ssim_map(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Per-pixel SSIM between (N, 1, H, W) tensors in [0, 1]."""
    k1 = gaussian_kernel(sigma, x.device)
    r = k1.numel() // 2
    kh, kv = k1.view(1, 1, 1, -1), k1.view(1, 1, -1, 1)

    def filt(t: torch.Tensor) -> torch.Tensor:
        t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), kh)
        return F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), kv)

    mx, my = filt(x), filt(y)
    sxx = filt(x * x) - mx * mx
    syy = filt(y * y) - my * my
    sxy = filt(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))


class AutoencoderDetector(AnomalyDetector):
    name = "autoencoder"
    title = "Convolutional autoencoder (reconstruction)"

    def __init__(self, settings, epochs: int | None = None):
        super().__init__(settings)
        self.size = settings.ae_size
        self.epochs = epochs or settings.ae_epochs
        self.net = ConvAE(latent=settings.ae_latent).to(self.device)

    def preprocess(self, samples: list[np.ndarray]) -> torch.Tensor:
        return rgb_to_tensor(samples, self.size)

    def _train(self, x: torch.Tensor, progress: Progress) -> None:
        net = self.net.train()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        bs = min(16, len(x))
        steps_per_epoch = -(-len(x) // bs)
        epochs = max(self.epochs, -(-self.s.ae_min_steps // steps_per_epoch))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_v = float("nan")
        for ep in range(epochs):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), bs):
                xb = x[perm[i : i + bs]].to(self.device)
                # light photometric jitter so the model does not memorise exact exposure
                xb = (xb * (1 + 0.1 * (torch.rand(len(xb), 1, 1, 1, device=self.device) - 0.5))).clamp(0, 1)
                rec = net(xb)
                loss = F.mse_loss(rec, xb) + 0.5 * (1 - ssim_map(_gray(rec), _gray(xb)).mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_v = loss.item()
            sched.step()
            progress((ep + 1) / epochs, f"epoch {ep + 1}/{epochs} loss={loss_v:.4f}")
        self.meta.update(final_loss=loss_v, epochs=epochs)
        enable_mc_dropout(self.net)

    @torch.no_grad()
    def _raw_maps(self, x: torch.Tensor):
        enable_mc_dropout(self.net)
        x = x.to(self.device)
        T = max(1, self.s.mc_samples)
        maps, recs = [], []
        for _ in range(T):
            rec = self.net(x)
            l1 = (rec - x).abs().mean(1)
            ss = 1 - ssim_map(_gray(rec), _gray(x)).squeeze(1)
            maps.append(blur_maps(0.5 * l1 + 0.5 * ss.clamp(min=0), sigma=2.0))
            recs.append(rec)
        stack = torch.stack(maps)  # T, N, H, W
        mean_map = stack.mean(0)
        per_pass = stack.flatten(2).max(2).values  # T, N
        cv = (per_pass.std(0) / per_pass.mean(0).clamp(min=1e-8)).cpu().numpy() if T > 1 else np.zeros(len(x))
        rec_mean = torch.stack(recs).mean(0).clamp(0, 1)
        rec_u8 = (rec_mean.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        extras: list[dict[str, Any]] = [{"reconstruction": rec_u8[i]} for i in range(len(x))]
        return mean_map, np.asarray(cv_to_unit(cv, 0.15)), extras

    def state(self) -> dict[str, Any]:
        return {"net": self.net.state_dict(), "size": self.size}

    def load_state(self, state: dict[str, Any]) -> None:
        self.size = state["size"]
        self.net.load_state_dict(state["net"])
        enable_mc_dropout(self.net)
