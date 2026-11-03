"""Why was a region flagged?

Explanations describe *how* a region deviates from learned normality — appearance novelty,
reconstruction failure, unexpected motion, colour / brightness / texture shifts against the
per-location normal profile, size, persistence and movement. They deliberately do not claim *what*
the object is: detecting an unusual pattern and recognising an unfamiliar object are different
problems, and the second needs an open-set recogniser that is outside this system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from .models.base import Detection
    from .tracking import Track

PROFILE_SIZE = 64


def _lab_tex(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(rgb, (PROFILE_SIZE, PROFILE_SIZE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(cv2.resize(rgb, (PROFILE_SIZE * 2, PROFILE_SIZE * 2)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    tex = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    tex = cv2.resize(cv2.blur(tex, (5, 5)), (PROFILE_SIZE, PROFILE_SIZE), interpolation=cv2.INTER_AREA)
    return lab, tex


@dataclass
class NormalityProfile:
    """Per-location colour (Lab) and texture statistics of the normal training images."""

    lab_mean: np.ndarray
    lab_std: np.ndarray
    tex_mean: np.ndarray
    tex_std: np.ndarray
    n: int

    @classmethod
    def from_images(cls, images: list[np.ndarray]) -> "NormalityProfile":
        labs, texs = zip(*(_lab_tex(im) for im in images))
        lab, tex = np.stack(labs), np.stack(texs)
        # floor std so a perfectly static scene does not turn sensor noise into "huge" z-scores
        return cls(lab.mean(0), np.maximum(lab.std(0), 4.0), tex.mean(0), np.maximum(tex.std(0), 2.0), len(images))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, lab_mean=self.lab_mean, lab_std=self.lab_std, tex_mean=self.tex_mean,
                            tex_std=self.tex_std, n=self.n)

    @classmethod
    def load(cls, path: Path) -> "NormalityProfile":
        d = np.load(path)
        return cls(d["lab_mean"], d["lab_std"], d["tex_mean"], d["tex_std"], int(d["n"]))

    def compare(self, rgb: np.ndarray, bbox: tuple[float, float, float, float]) -> dict[str, float]:
        """Mean z-scores inside a normalised bbox (x, y, w, h in 0..1)."""
        lab, tex = _lab_tex(rgb)
        x, y, w, h = bbox
        x0, y0 = int(x * PROFILE_SIZE), int(y * PROFILE_SIZE)
        x1 = max(x0 + 1, int(np.ceil((x + w) * PROFILE_SIZE)))
        y1 = max(y0 + 1, int(np.ceil((y + h) * PROFILE_SIZE)))
        sl = (slice(y0, y1), slice(x0, x1))
        z = (lab[sl] - self.lab_mean[sl]) / self.lab_std[sl]
        zt = (tex[sl] - self.tex_mean[sl]) / self.tex_std[sl]
        return {"dL": float(z[..., 0].mean()), "da": float(z[..., 1].mean()), "db": float(z[..., 2].mean()),
                "texture": float(zt.mean())}


MODEL_REASON = {
    "embedding": ("novel_appearance", "Unfamiliar appearance",
                  "Visual features here are {x:.1f}× beyond the normal threshold — nothing like this patch appeared in the normal data."),
    "autoencoder": ("reconstruction", "Unreconstructable structure",
                    "The autoencoder could not reproduce this area ({x:.1f}× threshold); its structure or texture is not part of the learned normal appearance."),
    "temporal": ("motion", "Unexpected change",
                 "What happened here did not match the predicted next frame ({x:.1f}× threshold) — unexpected motion or a sudden appearance/disappearance."),
}


def explain_region(
    region: dict[str, Any],
    model_peaks: dict[str, float],
    detections: dict[str, "Detection"],
    profile: NormalityProfile | None,
    frame_rgb: np.ndarray | None,
    track: "Track | None" = None,
) -> dict[str, Any]:
    reasons: list[dict[str, Any]] = []

    def add(factor: str, tag: str, text: str, strength: float) -> None:
        reasons.append({"factor": factor, "tag": tag, "text": text, "strength": round(float(np.clip(strength, 0, 1)), 3)})

    peak = region["peak"]
    fired = [m for m, p in model_peaks.items() if p >= 1.0]
    for m, p in sorted(model_peaks.items(), key=lambda kv: -kv[1]):
        if p >= 1.0 and m in MODEL_REASON:
            factor, tag, tmpl = MODEL_REASON[m]
            add(factor, tag, tmpl.format(x=p), min(1.0, (p - 1.0) / 2 + 0.5))

    if profile is not None and frame_rgb is not None:
        z = profile.compare(frame_rgb, tuple(region["bbox"]))
        if abs(z["dL"]) >= 2:
            word = "brighter" if z["dL"] > 0 else "darker"
            add("brightness", word.capitalize(), f"Noticeably {word} than this location normally is (z={z['dL']:+.1f}).",
                abs(z["dL"]) / 6)
        hues = []
        if abs(z["da"]) >= 2:
            hues.append("redder" if z["da"] > 0 else "greener")
        if abs(z["db"]) >= 2:
            hues.append("more yellow" if z["db"] > 0 else "bluer")
        if hues:
            add("color", "Colour shift", f"Colour is {' and '.join(hues)} than normal here.",
                max(abs(z["da"]), abs(z["db"])) / 6)
        if z["texture"] >= 2:
            add("texture", "Extra texture", "More edges/texture than usual — consistent with a foreign object, clutter, a scratch or a crack.",
                z["texture"] / 6)
        elif z["texture"] <= -2:
            add("texture", "Smoother", "Smoother than usual — something normally visible may be missing, covered or out of focus.",
                -z["texture"] / 6)

    if region["area_frac"] >= 0.35:
        add("global", "Scene-wide change",
            f"{region['area_frac']:.0%} of the frame deviates — more likely a lighting change, camera movement or a new scene than a local anomaly.",
            0.7)

    if track is not None:
        age = track.age_s
        if track.hits <= 3:
            add("persistence", "Just appeared", "Appeared in the last few frames.", 0.4)
        else:
            add("persistence", "Persistent", f"Has persisted for {age:.1f}s ({track.hits} frames).", min(1.0, age / 10))
        speed = track.speed
        if speed > 0.05:
            add("movement", "Moving", f"Moving across the frame (~{speed:.0%} of frame width per second).", min(1.0, speed))

    # confidence: model agreement + low per-model uncertainty
    active = [d for d in detections.values()]
    mean_unc = float(np.mean([d.uncertainty for d in active])) if active else 0.5
    agreement = len(fired) / max(1, len(model_peaks))
    confidence = float(np.clip((1 - mean_unc) * (0.6 + 0.4 * agreement), 0, 1))
    if mean_unc > 0.5:
        add("uncertainty", "Low confidence",
            "The detectors are unsure here (score close to threshold or unstable across model samples).", mean_unc)
    if len(model_peaks) > 1 and 0 < len(fired) < len(model_peaks):
        quiet = [m for m in model_peaks if m not in fired]
        add("disagreement", "Partial agreement", f"Flagged by {', '.join(fired)} but not by {', '.join(quiet)}.", 0.3)

    reasons.sort(key=lambda r: -r["strength"])
    tags = [r["tag"] for r in reasons if r["factor"] not in ("uncertainty", "disagreement")][:3]
    summary = " · ".join(tags) if tags else "Deviation from normal"
    return {
        "summary": summary,
        "reasons": reasons,
        "confidence": round(confidence, 3),
        "peak": round(peak, 3),
        "models_flagged": fired,
        "kind": "deviation",  # this is not an object identity
    }
