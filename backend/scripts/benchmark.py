"""Compare autoencoder / feature-embedding / temporal detectors on MVTec AD 2 or MSAD.

Examples:
  python scripts/benchmark.py mvtec_ad2 --root /data/mvtec_ad2 --category can
  python scripts/benchmark.py mvtec_ad2 --root /data/mvtec_ad2 --category all --out results/mvtec.json
  python scripts/benchmark.py msad --root /data/msad --labels /data/msad/labels.json --fps 5
Add --save-db to store results so they appear in the dashboard's Research tab.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chimera.benchmark import bench_msad, bench_mvtec  # noqa: E402
from chimera.config import get_settings  # noqa: E402
from chimera.datasets import MVTecAD2  # noqa: E402


def progress(f: float, m: str) -> None:
    print(f"\r[{f:6.1%}] {m[:100]:<100}", end="", flush=True)


def table(results: dict) -> str:
    rows = []
    for ds, res in results.items():
        for model, m in res.items():
            if model.startswith("_"):
                continue
            keys = [k for k in m if k != "meta"]
            rows.append(f"{ds:<14} {model:<12} " + "  ".join(
                f"{k}={m[k]:.4f}" if isinstance(m[k], float) else f"{k}={m[k]}" for k in keys))
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["mvtec_ad2", "msad"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", default="all")
    ap.add_argument("--labels")
    ap.add_argument("--models", default="autoencoder,embedding,temporal")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train", type=int)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out")
    ap.add_argument("--save-db", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    results = {}
    if a.dataset == "mvtec_ad2":
        cats = MVTecAD2.categories(a.root) if a.category == "all" else [a.category]
        for c in cats:
            print(f"\n== {c}")
            results[c] = bench_mvtec(a.root, c, models, s, progress, a.epochs, a.max_train)
    else:
        results["msad"] = bench_msad(a.root, models, s, progress, a.labels, a.fps, a.epochs)
    print("\n\n" + table(results))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {a.out}")
    if a.save_db:
        from chimera.db import BenchmarkRun, SessionLocal, init_db

        init_db()
        with SessionLocal() as db:
            for cat, res in results.items():
                db.add(BenchmarkRun(dataset=a.dataset, category=cat, results=json.loads(json.dumps(res, default=str))))
            db.commit()
        print("saved to database")


if __name__ == "__main__":
    main()
