from __future__ import annotations

import json
import logging
import uuid
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import __version__
from ..config import get_settings, resolve_device
from ..db import BenchmarkRun, Event, Job, Sample, Scene, StreamSession, get_db, init_db, utcnow
from ..jobs import JobContext, submit
from ..models import DETECTORS
from ..pipeline import FrameAnalyzer
from ..recorder import SessionRecorder, as_utc
from ..scenes import SceneRegistry, decode_image, iter_video, slugify, video_info

log = logging.getLogger("chimera.api")
settings = get_settings()
registry = SceneRegistry(settings)
UPLOADS = settings.data_dir / "uploads"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    UPLOADS.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="CHIMERA", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- helpers
def iso(dt) -> str | None:
    dt = as_utc(dt)
    return dt.isoformat() if dt else None


def get_scene(db: Session, scene_id: str) -> Scene:
    sc = db.get(Scene, scene_id)
    if sc is None:
        raise HTTPException(404, f"scene '{scene_id}' not found")
    return sc


def scene_dict(db: Session, sc: Scene) -> dict[str, Any]:
    store = registry.get(sc.id)
    models = {}
    for name, cls in DETECTORS.items():
        d = store.detectors.get(name)
        models[name] = {
            "title": cls.title,
            "trained": d is not None,
            "meta": {k: v for k, v in (d.meta if d else {}).items() if isinstance(v, (int, float, str, bool))},
            "calibration": d.calib if d else None,
        }
    n_events = db.scalar(select(func.count(Event.id)).where(Event.scene_id == sc.id)) or 0
    return {"id": sc.id, "name": sc.name, "description": sc.description, "created_at": iso(sc.created_at),
            "models": models, "normal": store.normal_stats(), "events": n_events}


def event_dict(e: Event) -> dict[str, Any]:
    return {"id": e.id, "session_id": e.session_id, "scene_id": e.scene_id, "track_id": e.track_id,
            "started_at": iso(e.started_at), "duration_s": round(e.duration_s, 2), "frame_start": e.frame_start,
            "frame_end": e.frame_end, "peak_score": round(e.peak_score, 3), "uncertainty": round(e.uncertainty, 3),
            "bbox": e.bbox, "models": e.models, "summary": e.summary, "explanation": e.explanation,
            "status": e.status, "snapshot_url": f"/api/events/{e.id}/snapshot" if e.snapshot else None}


def session_dict(s: StreamSession) -> dict[str, Any]:
    return {"id": s.id, "scene_id": s.scene_id, "source": s.source, "config": s.config,
            "started_at": iso(s.started_at), "ended_at": iso(s.ended_at), "frames": s.frames,
            "anomalous_frames": s.anomalous_frames, "max_score": round(s.max_score, 3)}


def job_dict(j: Job) -> dict[str, Any]:
    return {"id": j.id, "kind": j.kind, "scene_id": j.scene_id, "status": j.status, "progress": round(j.progress, 4),
            "message": j.message, "params": j.params, "result": j.result, "created_at": iso(j.created_at),
            "finished_at": iso(j.finished_at)}


def parse_models(models: str | None) -> list[str] | None:
    if not models:
        return None
    out = [m.strip() for m in models.split(",") if m.strip()]
    bad = [m for m in out if m not in DETECTORS]
    if bad:
        raise HTTPException(400, f"unknown models: {bad}")
    return out


# ---------------------------------------------------------------- system
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "device": str(resolve_device(settings.device)),
            "backbone": settings.backbone, "detectors": {k: v.title for k, v in DETECTORS.items()}}


# ---------------------------------------------------------------- scenes
class SceneIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


@app.get("/api/scenes")
def list_scenes(db: Session = Depends(get_db)):
    return [scene_dict(db, s) for s in db.scalars(select(Scene).order_by(Scene.created_at))]


@app.post("/api/scenes", status_code=201)
def create_scene(body: SceneIn, db: Session = Depends(get_db)):
    base = slugify(body.name)
    sid, n = base, 2
    while db.get(Scene, sid):
        sid, n = f"{base}-{n}", n + 1
    sc = Scene(id=sid, name=body.name, description=body.description)
    db.add(sc)
    db.commit()
    return scene_dict(db, sc)


@app.get("/api/scenes/{scene_id}")
def read_scene(scene_id: str, db: Session = Depends(get_db)):
    return scene_dict(db, get_scene(db, scene_id))


@app.delete("/api/scenes/{scene_id}", status_code=204)
def delete_scene(scene_id: str, db: Session = Depends(get_db)):
    sc = get_scene(db, scene_id)
    sess_ids = list(db.scalars(select(StreamSession.id).where(StreamSession.scene_id == scene_id)))
    if sess_ids:
        db.query(Sample).filter(Sample.session_id.in_(sess_ids)).delete(synchronize_session=False)
        db.query(Event).filter(Event.session_id.in_(sess_ids)).delete(synchronize_session=False)
        db.query(StreamSession).filter(StreamSession.id.in_(sess_ids)).delete(synchronize_session=False)
    db.delete(sc)
    db.commit()
    registry.delete(scene_id)


@app.post("/api/scenes/{scene_id}/normal")
async def upload_normal(scene_id: str, files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    get_scene(db, scene_id)
    store = registry.get(scene_id)
    added: Counter = Counter()
    errors = []
    for f in files:
        data = await f.read()
        try:
            added.update(await run_in_threadpool(store.add_upload, f.filename or "upload.png", data, UPLOADS))
        except ValueError as e:
            errors.append(f"{f.filename}: {e}")
    return {"added": dict(added), "errors": errors, "normal": store.normal_stats()}


@app.delete("/api/scenes/{scene_id}/normal")
def clear_normal(scene_id: str, db: Session = Depends(get_db)):
    get_scene(db, scene_id)
    registry.get(scene_id).clear_normal()
    return registry.get(scene_id).normal_stats()


class TrainIn(BaseModel):
    models: list[str] = ["embedding", "autoencoder", "temporal"]
    epochs: int | None = Field(default=None, ge=1, le=500)


@app.post("/api/scenes/{scene_id}/train", status_code=202)
def train_scene(scene_id: str, body: TrainIn, db: Session = Depends(get_db)):
    sc = get_scene(db, scene_id)
    bad = [m for m in body.models if m not in DETECTORS]
    if bad or not body.models:
        raise HTTPException(400, f"invalid models: {bad or body.models}")
    store = registry.get(scene_id)
    if not store.normal_stats()["images"] and not store.normal_stats()["sequence_frames"]:
        raise HTTPException(400, "no normal data yet — upload images/videos or capture from the webcam")

    def run(ctx: JobContext) -> dict[str, Any]:
        res = store.train(body.models, body.epochs, ctx.progress)
        from ..db import SessionLocal

        with SessionLocal() as s2:
            row = s2.get(Scene, sc.id)
            if row:
                row.models = {**(row.models or {}), **json.loads(json.dumps(res, default=str))}
                s2.commit()
        return json.loads(json.dumps(res, default=str))

    return {"job_id": submit("train", body.model_dump(), run, scene_id)}


# ---------------------------------------------------------------- detection
@app.post("/api/scenes/{scene_id}/detect")
async def detect_image(
    scene_id: str,
    file: UploadFile = File(...),
    models: str | None = Form(None),
    threshold: float = Form(1.0),
    fusion: str = Form("max"),
    save: bool = Form(True),
    db: Session = Depends(get_db),
):
    get_scene(db, scene_id)
    store = registry.get(scene_id)
    try:
        rgb = decode_image(await file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    chosen = [m for m in (parse_models(models) or store.available()) if m != "temporal"]
    try:
        analyzer = FrameAnalyzer(store, chosen, fusion=fusion, threshold=threshold, min_hits=1,
                                 min_region_frac=settings.min_region_frac, include_model_maps=True, stream=False)
    except ValueError as e:
        raise HTTPException(409, str(e))
    a = await run_in_threadpool(analyzer.analyze, rgb)
    out = dict(a.payload)
    if save:
        rec = SessionRecorder(scene_id, f"upload:{file.filename}", {"models": analyzer.models, "threshold": threshold}, settings)
        out["events_new"] = await run_in_threadpool(rec.record, rgb, a)
        await run_in_threadpool(rec.close, analyzer)
        out["session_id"] = rec.session_id
    out["width"], out["height"] = rgb.shape[1], rgb.shape[0]
    return out


@app.post("/api/scenes/{scene_id}/videos", status_code=202)
async def analyze_video(
    scene_id: str,
    file: UploadFile = File(...),
    models: str | None = Form(None),
    threshold: float = Form(1.0),
    fusion: str = Form("max"),
    fps: float = Form(10.0),
    db: Session = Depends(get_db),
):
    get_scene(db, scene_id)
    store = registry.get(scene_id)
    chosen = parse_models(models)
    if not store.available():
        raise HTTPException(409, "scene has no trained detectors")
    path = UPLOADS / f"{uuid.uuid4().hex}{Path(file.filename or 'v.mp4').suffix}"
    path.write_bytes(await file.read())
    name = file.filename or path.name

    def run(ctx: JobContext) -> dict[str, Any]:
        try:
            analyzer = FrameAnalyzer(store, chosen, fusion=fusion, threshold=threshold, min_hits=settings.track_min_hits,
                                     max_misses=settings.track_max_misses, min_region_frac=settings.min_region_frac)
            rec = SessionRecorder(scene_id, f"video:{name}", {"models": analyzer.models, "threshold": threshold,
                                                              "fps": fps, "fusion": fusion}, settings)
            total = max(1.0, video_info(path)["frames"])
            n_ev = 0
            for i, ts, rgb in iter_video(path, fps):
                a = analyzer.analyze(rgb, ts=ts)
                n_ev += len(rec.record(rgb, a))
                ctx.progress(i / total, f"frame {i}/{int(total)} · {n_ev} events")
            rec.close(analyzer, wall_end=rec.start + timedelta(seconds=total / video_info(path)["fps"]))
            return {"session_id": rec.session_id, "events": n_ev, "frames": rec.frames}
        finally:
            path.unlink(missing_ok=True)

    return {"job_id": submit("video", {"file": name, "models": chosen, "threshold": threshold, "fps": fps}, run, scene_id)}


@app.websocket("/api/scenes/{scene_id}/stream")
async def stream(ws: WebSocket, scene_id: str):
    """Protocol — client→server text: {"type":"start", models?, threshold?, fusion?, record?}
    | {"type":"capture"} | {"type":"stop"} | {"type":"config", threshold?}; binary: one JPEG frame.
    Server replies to each frame with {"type":"result"|"captured", ...}."""
    await ws.accept()
    from ..db import SessionLocal

    with SessionLocal() as db:
        if db.get(Scene, scene_id) is None:
            await ws.send_json({"type": "error", "message": f"scene '{scene_id}' not found"})
            await ws.close()
            return
    store = registry.get(scene_id)
    analyzer: FrameAnalyzer | None = None
    recorder: SessionRecorder | None = None
    mode, seq_dir, cap_n = "idle", None, 0

    async def finish() -> dict[str, Any]:
        nonlocal analyzer, recorder, mode, seq_dir, cap_n
        info: dict[str, Any] = {}
        if recorder is not None:
            await run_in_threadpool(recorder.close, analyzer)
            info["session_id"] = recorder.session_id
        if mode == "capture":
            info["captured"] = cap_n
            if cap_n == 0 and seq_dir is not None:
                import shutil

                shutil.rmtree(seq_dir, ignore_errors=True)
        analyzer, recorder, mode, seq_dir, cap_n = None, None, "idle", None, 0
        return info

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                cmd = json.loads(msg["text"])
                kind = cmd.get("type")
                if kind == "start":
                    await finish()
                    try:
                        analyzer = FrameAnalyzer(store, cmd.get("models"), fusion=cmd.get("fusion", "max"),
                                                 threshold=float(cmd.get("threshold", 1.0)),
                                                 min_hits=settings.track_min_hits, max_misses=settings.track_max_misses,
                                                 min_region_frac=settings.min_region_frac)
                    except ValueError as e:
                        await ws.send_json({"type": "error", "message": str(e)})
                        continue
                    if cmd.get("record", True):
                        recorder = await run_in_threadpool(
                            SessionRecorder, scene_id, cmd.get("source", "webcam"),
                            {"models": analyzer.models, "threshold": analyzer.threshold, "fusion": analyzer.fusion}, settings)
                    mode = "detect"
                    await ws.send_json({"type": "started", "models": analyzer.models,
                                        "session_id": recorder.session_id if recorder else None})
                elif kind == "capture":
                    await finish()
                    seq_dir, cap_n, mode = store.new_sequence(), 0, "capture"
                    await ws.send_json({"type": "capturing"})
                elif kind == "config":
                    if analyzer is not None and "threshold" in cmd:
                        analyzer.threshold = max(1e-3, float(cmd["threshold"]))
                    await ws.send_json({"type": "configured", "threshold": analyzer.threshold if analyzer else None})
                elif kind == "stop":
                    info = await finish()
                    await ws.send_json({"type": "stopped", **info, "normal": store.normal_stats()})
                else:
                    await ws.send_json({"type": "error", "message": f"unknown command {kind!r}"})
                continue
            data = msg.get("bytes")
            if not data or mode == "idle":
                await ws.send_json({"type": "error", "message": "send a start or capture command first"})
                continue
            try:
                rgb = decode_image(data)
            except ValueError:
                await ws.send_json({"type": "error", "message": "bad frame"})
                continue
            if mode == "capture":
                await run_in_threadpool(store.add_sequence_frame, seq_dir, cap_n, rgb)
                cap_n += 1
                await ws.send_json({"type": "captured", "count": cap_n})
            elif analyzer is not None:
                a = await run_in_threadpool(analyzer.analyze, rgb)
                events = await run_in_threadpool(recorder.record, rgb, a) if recorder else []
                await ws.send_json({"type": "result", **a.payload, "events_new": events,
                                    "session_id": recorder.session_id if recorder else None})
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("stream error")
    finally:
        await finish()


# ---------------------------------------------------------------- jobs
@app.get("/api/jobs")
def list_jobs(scene_id: str | None = None, limit: int = 20, db: Session = Depends(get_db)):
    q = select(Job).order_by(Job.id.desc()).limit(limit)
    if scene_id:
        q = q.where(Job.scene_id == scene_id)
    return [job_dict(j) for j in db.scalars(q)]


@app.get("/api/jobs/{job_id}")
def read_job(job_id: int, db: Session = Depends(get_db)):
    j = db.get(Job, job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    return job_dict(j)


# ---------------------------------------------------------------- sessions, events, reports
@app.get("/api/sessions")
def list_sessions(scene_id: str | None = None, limit: int = 50, db: Session = Depends(get_db)):
    q = select(StreamSession).order_by(StreamSession.id.desc()).limit(limit)
    if scene_id:
        q = q.where(StreamSession.scene_id == scene_id)
    return [session_dict(s) for s in db.scalars(q)]


def _reason_hist(events: list[Event]) -> dict[str, int]:
    c: Counter = Counter()
    for e in events:
        for r in (e.explanation or {}).get("reasons", []):
            c[r["factor"]] += 1
    return dict(c.most_common())


@app.get("/api/sessions/{session_id}/report")
def session_report(session_id: int, db: Session = Depends(get_db)):
    s = db.get(StreamSession, session_id)
    if s is None:
        raise HTTPException(404, "session not found")
    samples = list(db.scalars(select(Sample).where(Sample.session_id == session_id).order_by(Sample.frame_idx)))
    events = list(db.scalars(select(Event).where(Event.session_id == session_id).order_by(Event.started_at)))
    per_model: dict[str, list[float]] = defaultdict(list)
    for sm in samples:
        for k, v in (sm.per_model or {}).items():
            per_model[k].append(v)
    return {
        "session": session_dict(s),
        "timeline": [{"t": round(x.t, 3), "frame": x.frame_idx, "score": x.score, "uncertainty": x.uncertainty,
                      "regions": x.n_regions, **{f"m_{k}": v for k, v in (x.per_model or {}).items()}} for x in samples],
        "events": [event_dict(e) for e in events],
        "stats": {
            "frames": s.frames,
            "anomalous_ratio": round(s.anomalous_frames / s.frames, 4) if s.frames else 0.0,
            "max_score": round(s.max_score, 3),
            "mean_uncertainty": round(sum(x.uncertainty for x in samples) / len(samples), 4) if samples else None,
            "events": len(events),
            "per_model_mean_score": {k: round(sum(v) / len(v), 4) for k, v in per_model.items()},
            "reasons": _reason_hist(events),
        },
    }


@app.get("/api/events")
def list_events(scene_id: str | None = None, session_id: int | None = None, min_score: float = 0.0,
                limit: int = Query(100, le=1000), db: Session = Depends(get_db)):
    q = select(Event).where(Event.peak_score >= min_score).order_by(Event.id.desc()).limit(limit)
    if scene_id:
        q = q.where(Event.scene_id == scene_id)
    if session_id:
        q = q.where(Event.session_id == session_id)
    return [event_dict(e) for e in db.scalars(q)]


@app.get("/api/events/{event_id}")
def read_event(event_id: int, db: Session = Depends(get_db)):
    e = db.get(Event, event_id)
    if e is None:
        raise HTTPException(404, "event not found")
    return event_dict(e)


@app.get("/api/events/{event_id}/snapshot")
def event_snapshot(event_id: int, db: Session = Depends(get_db)):
    e = db.get(Event, event_id)
    if e is None or not e.snapshot:
        raise HTTPException(404, "snapshot not found")
    p = (settings.snapshots_dir / e.snapshot).resolve()
    if settings.snapshots_dir.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "snapshot not found")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/reports/overview")
def overview(scene_id: str | None = None, days: int = Query(14, ge=1, le=365), db: Session = Depends(get_db)):
    since = utcnow() - timedelta(days=days)
    q = select(Event).where(Event.started_at >= since)
    if scene_id:
        q = q.where(Event.scene_id == scene_id)
    events = list(db.scalars(q.order_by(Event.started_at)))
    per_day: Counter = Counter(as_utc(e.started_at).date().isoformat() for e in events)
    day_list = [(since + timedelta(days=i + 1)).date().isoformat() for i in range(days)]
    by_scene: Counter = Counter(e.scene_id for e in events)
    by_model: Counter = Counter(m for e in events for m in (e.models or []))
    sq = select(func.count(StreamSession.id))
    if scene_id:
        sq = sq.where(StreamSession.scene_id == scene_id)
    return {
        "days": days,
        "totals": {"events": len(events), "sessions": db.scalar(sq) or 0,
                   "scenes": db.scalar(select(func.count(Scene.id))) or 0,
                   "mean_uncertainty": round(sum(e.uncertainty for e in events) / len(events), 4) if events else None},
        "events_per_day": [{"date": d, "events": per_day.get(d, 0)} for d in day_list],
        "by_scene": dict(by_scene),
        "by_model": dict(by_model),
        "reasons": _reason_hist(events),
        "score_hist": _hist([e.peak_score for e in events]),
        "recent": [event_dict(e) for e in events[-12:][::-1]],
    }


def _hist(vals: list[float], edges=(1, 1.5, 2, 3, 5, 10)) -> list[dict[str, Any]]:
    bins = []
    lo = 0.0
    for hi in [*edges, float("inf")]:
        label = f"{lo:g}–{hi:g}×" if hi != float("inf") else f"≥{lo:g}×"
        bins.append({"bin": label, "count": sum(lo <= v < hi for v in vals)})
        lo = hi
    return [b for b in bins if b["bin"] != "0–1×" or b["count"]]


# ---------------------------------------------------------------- benchmarks
class BenchIn(BaseModel):
    dataset: str = Field(pattern="^(mvtec_ad2|msad)$")
    root: str
    category: str | None = None  # MVTec AD 2 object; "all" runs every object
    labels: str | None = None  # MSAD labels file/dir
    models: list[str] = ["autoencoder", "embedding", "temporal"]
    epochs: int | None = Field(default=None, ge=1, le=500)
    max_train: int | None = None
    fps: float = 5.0


@app.get("/api/benchmarks")
def list_benchmarks(db: Session = Depends(get_db)):
    return [{"id": b.id, "dataset": b.dataset, "category": b.category, "created_at": iso(b.created_at),
             "results": b.results} for b in db.scalars(select(BenchmarkRun).order_by(BenchmarkRun.id.desc()))]


@app.post("/api/benchmarks", status_code=202)
def run_benchmark(body: BenchIn):
    from ..benchmark import bench_msad, bench_mvtec
    from ..datasets import MVTecAD2
    from ..db import SessionLocal

    if not Path(body.root).is_dir():
        raise HTTPException(400, f"dataset root not found on the server: {body.root}")

    def run(ctx: JobContext) -> dict[str, Any]:
        out = {}
        if body.dataset == "mvtec_ad2":
            cats = MVTecAD2.categories(body.root) if body.category in (None, "all") else [body.category]
            for ci, cat in enumerate(cats):
                prog = lambda f, m, ci=ci, cat=cat: ctx.progress((ci + f) / len(cats), f"{cat}: {m}")  # noqa: E731
                out[cat] = bench_mvtec(body.root, cat, body.models, settings, prog, body.epochs, body.max_train)
                with SessionLocal() as db:
                    db.add(BenchmarkRun(dataset="mvtec_ad2", category=cat, results=out[cat]))
                    db.commit()
        else:
            out["msad"] = bench_msad(body.root, body.models, settings, ctx.progress, body.labels, body.fps, body.epochs)
            with SessionLocal() as db:
                db.add(BenchmarkRun(dataset="msad", category="all", results=out["msad"]))
                db.commit()
        return out

    return {"job_id": submit("benchmark", body.model_dump(), run)}
