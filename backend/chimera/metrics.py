from __future__ import annotations

import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y).ravel()
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, np.asarray(s).ravel()))


def best_f1(y: np.ndarray, s: np.ndarray) -> float | None:
    if np.asarray(y).min() == np.asarray(y).max():
        return None
    p, r, _ = precision_recall_curve(y, s)
    f = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(f.max())


def f1_at(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp = (y & pred).sum()
    fp = (~y & pred).sum()
    fn = (y & ~pred).sum()
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def pixel_auroc(masks: list[np.ndarray], maps: list[np.ndarray], max_px: int = 3_000_000) -> float | None:
    y = np.concatenate([m.ravel() for m in masks])
    s = np.concatenate([m.ravel() for m in maps])
    if len(y) > max_px:
        idx = np.random.default_rng(0).choice(len(y), max_px, replace=False)
        y, s = y[idx], s[idx]
    return safe_auroc(y, s)


def au_pro(masks: list[np.ndarray], maps: list[np.ndarray], fpr_limit: float = 0.3, steps: int = 200) -> float | None:
    """Area under the per-region-overlap curve up to fpr_limit, normalised to [0, 1]."""
    comps = []
    for i, m in enumerate(masks):
        n, lab = cv2.connectedComponents(m.astype(np.uint8))
        comps.extend((i, lab == k) for k in range(1, n))
    if not comps:
        return None
    allv = np.concatenate([m.ravel() for m in maps])
    ths = np.quantile(allv, np.linspace(1.0, 0.0, steps))
    neg = sum(int((~m).sum()) for m in masks)
    fprs, pros = [], []
    for th in ths:
        fp = sum(int(((mp >= th) & ~m).sum()) for mp, m in zip(maps, masks))
        fpr = fp / max(1, neg)
        pro = np.mean([((maps[i] >= th) & c).sum() / c.sum() for i, c in comps])
        fprs.append(fpr)
        pros.append(pro)
        if fpr > fpr_limit:
            break
    fprs, pros = np.array(fprs), np.array(pros)
    if len(fprs) > 1 and fprs[-1] > fpr_limit:  # interpolate the last segment to the limit
        f0, f1, p0, p1 = fprs[-2], fprs[-1], pros[-2], pros[-1]
        pros[-1] = p0 + (p1 - p0) * (fpr_limit - f0) / max(1e-12, f1 - f0)
        fprs[-1] = fpr_limit
    return float(np.trapezoid(pros, fprs) / fpr_limit)


def smooth(s: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    if sigma <= 0 or len(s) < 3:
        return s
    r = int(3 * sigma)
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma**2))
    k /= k.sum()
    return np.convolve(np.pad(s, r, mode="edge"), k, mode="valid")
