"""Feature-embedding detector (PatchCore-style).

A frozen ImageNet backbone turns every image patch into a locally-aware feature vector. Normal
patches are stored in a memory bank reduced by greedy coreset selection. At test time each patch is
scored by its distance to the nearest normal patch.

Epistemic uncertainty: the bank is randomly partitioned into M sub-banks. If the sub-banks disagree
about how far the most anomalous patch is from normality, the bank covers that part of feature
space sparsely, so the verdict is less trustworthy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .base import AnomalyDetector, Progress, blur_maps, cv_to_unit, rgb_to_tensor

log = logging.getLogger(__name__)


class PatchEmbedder(nn.Module):
    def __init__(self, backbone: str = "resnet18", proj_dim: int = 256, seed: int = 0):
        super().__init__()
        ctor = getattr(torchvision.models, backbone)
        try:
            weights = torchvision.models.get_model_weights(backbone).DEFAULT
            net = ctor(weights=weights)
            self.pretrained = True
        except Exception as e:  # offline / no cache: still runs, but quality drops a lot
            log.warning("could not load pretrained %s weights (%s); using random init", backbone, e)
            net = ctor(weights=None)
            self.pretrained = False
        net.fc = nn.Identity()
        del net.layer4
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            c = self._concat(torch.zeros(1, 3, 64, 64)).shape[1]
        self.in_dim = c
        self.out_dim = min(proj_dim, c)
        g = torch.Generator().manual_seed(seed)
        proj = torch.randn(c, self.out_dim, generator=g) / self.out_dim**0.5 if self.out_dim < c else torch.eye(c)
        self.register_buffer("proj", proj)

    def _concat(self, x: torch.Tensor) -> torch.Tensor:
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        f2 = n.layer2(n.layer1(x))
        f3 = n.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], 1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self._concat(x)  # N, C, h, w
        return torch.einsum("nchw,cd->ndhw", f, self.proj)


def greedy_coreset(x: torch.Tensor, n: int, proj_dim: int = 128, progress: Progress | None = None) -> torch.Tensor:
    """Greedy k-center selection; returns indices of n points covering x."""
    if n >= len(x):
        return torch.arange(len(x))
    g = torch.Generator().manual_seed(0)
    z = x
    if x.shape[1] > proj_dim:
        z = x @ (torch.randn(x.shape[1], proj_dim, generator=g).to(x) / proj_dim**0.5)
    idx = torch.empty(n, dtype=torch.long)
    idx[0] = int(torch.randint(len(x), (1,), generator=g))
    min_d = torch.linalg.vector_norm(z - z[idx[0]], dim=1)
    for i in range(1, n):
        j = int(torch.argmax(min_d))
        idx[i] = j
        min_d = torch.minimum(min_d, torch.linalg.vector_norm(z - z[j], dim=1))
        if progress and i % 500 == 0:
            progress(i / n, f"coreset {i}/{n}")
    return idx


class EmbeddingDetector(AnomalyDetector):
    name = "embedding"
    title = "Feature-embedding memory bank (PatchCore-style)"

    def __init__(self, settings, backbone: str | None = None):
        super().__init__(settings)
        self.size = settings.embed_size
        self.backbone = backbone or settings.backbone
        self.embedder = PatchEmbedder(self.backbone, settings.embed_proj_dim).to(self.device)
        self.bank: torch.Tensor | None = None
        self.partition: torch.Tensor | None = None
        self.meta["pretrained"] = self.embedder.pretrained
        self.meta["backbone"] = self.backbone

    def preprocess(self, samples: list[np.ndarray]) -> torch.Tensor:
        return rgb_to_tensor(samples, self.size, normalize=True)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x.to(self.device))

    def _train(self, x: torch.Tensor, progress: Progress) -> None:
        feats = []
        for i in range(0, len(x), 16):
            f = self._embed(x[i : i + 16])
            feats.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).cpu())
            progress(0.3 * min(1.0, (i + 16) / len(x)), f"embedding {min(i + 16, len(x))}/{len(x)} images")
        allf = torch.cat(feats)
        g = torch.Generator().manual_seed(0)
        if len(allf) > self.s.coreset_presample:
            allf = allf[torch.randperm(len(allf), generator=g)[: self.s.coreset_presample]]
        target = min(self.s.coreset_max, max(1000, len(allf) // 10))
        idx = greedy_coreset(allf.to(self.device), target, progress=lambda f, m: progress(0.3 + 0.7 * f, m))
        self.bank = allf[idx.cpu()].to(self.device)
        self.partition = torch.randint(0, self.s.bank_partitions, (len(self.bank),), generator=g).to(self.device)
        self.meta.update(bank_size=len(self.bank), patches_seen=int(sum(len(f) for f in feats)))

    @torch.no_grad()
    def _raw_maps(self, x: torch.Tensor):
        assert self.bank is not None and self.partition is not None
        f = self._embed(x)
        n, c, h, w = f.shape
        patches = f.permute(0, 2, 3, 1).reshape(n, h * w, c)
        maps, epis, extras = [], [], []
        M = int(self.partition.max()) + 1
        for i in range(n):
            d = torch.cdist(patches[i], self.bank)  # hw, B
            nn_d = d.min(1).values
            # per-partition nearest distances -> how stable is the verdict at the worst patch
            sub = torch.stack([d[:, self.partition == m].min(1).values for m in range(M)])  # M, hw
            worst = int(torch.argmax(nn_d))
            vals = sub[:, worst]
            cv = float(vals.std() / vals.mean().clamp(min=1e-8))
            maps.append(nn_d.view(h, w))
            epis.append(cv)
            extras.append({"patch_grid": [h, w], "nearest_normal_dist": float(nn_d[worst])})
        m = torch.stack(maps).unsqueeze(1)
        m = F.interpolate(m, size=(self.size, self.size), mode="bilinear", align_corners=False).squeeze(1)
        return blur_maps(m, sigma=4.0), np.asarray(cv_to_unit(np.array(epis), 0.08)), extras

    def state(self) -> dict[str, Any]:
        return {
            "bank": self.bank.cpu() if self.bank is not None else None,
            "partition": self.partition.cpu() if self.partition is not None else None,
            "proj": self.embedder.proj.cpu(),
            "backbone": self.backbone,
            "size": self.size,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if state["backbone"] != self.backbone:
            self.backbone = state["backbone"]
            self.embedder = PatchEmbedder(self.backbone, state["proj"].shape[1]).to(self.device)
        self.size = state["size"]
        self.embedder.proj = state["proj"].to(self.device)
        self.bank = state["bank"].to(self.device)
        self.partition = state["partition"].to(self.device)
