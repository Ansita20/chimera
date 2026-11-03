"""Persists stream sessions: timeline samples, anomaly events (one per confirmed track) and snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from .config import Settings
from .db import Event, Sample, SessionLocal, StreamSession, utcnow
from .pipeline import FrameAnalysis, FrameAnalyzer, render_snapshot
from .tracking import Track


class SessionRecorder:
    def __init__(self, scene_id: str, source: str, config: dict[str, Any], settings: Settings, sample_every: int = 3,
                 wall_clock_start: datetime | None = None):
        self.s, self.scene_id, self.sample_every = settings, scene_id, sample_every
        self.start = wall_clock_start or utcnow()
        with SessionLocal() as db:
            sess = StreamSession(scene_id=scene_id, source=source, config=config, started_at=self.start)
            db.add(sess)
            db.commit()
            self.session_id = sess.id
        self.t0: float | None = None
        self.frames = self.anomalous = 0
        self.max_score = 0.0

    def _dt(self, ts: float) -> datetime:
        return self.start + timedelta(seconds=ts - (self.t0 or ts))

    def _close(self, db, t: Track) -> None:
        if t.event_id is None:
            return
        ev = db.get(Event, t.event_id)
        if ev is None:
            return
        ev.duration_s = t.age_s
        ev.frame_end = t.last_frame
        ev.peak_score = t.peak
        ev.uncertainty = t.mean_uncertainty
        ev.models = sorted(t.models)
        ev.status = "closed"
        if t.explanation:
            ev.explanation, ev.summary = t.explanation, t.explanation["summary"][:300]

    def record(self, rgb: np.ndarray, a: FrameAnalysis) -> list[dict[str, Any]]:
        p = a.payload
        if self.t0 is None:
            self.t0 = p["ts"]
        self.frames += 1
        self.anomalous += int(p["verdict"] == "anomalous")
        self.max_score = max(self.max_score, p["score"])
        new_events: list[dict[str, Any]] = []
        with SessionLocal() as db:
            if p["frame_idx"] % self.sample_every == 0 or p["regions"]:
                db.add(Sample(session_id=self.session_id, t=p["ts"] - self.t0, frame_idx=p["frame_idx"], score=p["score"],
                              uncertainty=p["uncertainty"], n_regions=len(p["regions"]),
                              per_model={k: v["score"] for k, v in p["models"].items()}))
            by_track = {r["track_id"]: r for r in p["regions"]}
            for t in a.confirmed:
                r = by_track.get(t.id)
                expl = t.explanation or (r["explanation"] if r else {})
                rel = f"{self.session_id}/{t.id}.jpg"
                render_snapshot(rgb, a.fused, t.bbox, self.s.snapshots_dir / rel, f"#{t.id} {t.peak:.1f}x")
                ev = Event(session_id=self.session_id, scene_id=self.scene_id, track_id=t.id,
                           started_at=self._dt(t.first_ts), duration_s=t.age_s, frame_start=t.first_frame,
                           frame_end=t.last_frame, peak_score=t.peak, uncertainty=t.mean_uncertainty, bbox=t.bbox,
                           models=sorted(t.models), summary=expl.get("summary", "")[:300], explanation=expl, snapshot=rel)
                db.add(ev)
                db.flush()
                t.event_id = ev.id
                new_events.append({"id": ev.id, "track_id": t.id, "summary": ev.summary, "peak": round(t.peak, 3)})
            for t in a.ended:
                self._close(db, t)
            if self.frames % 30 == 0:
                self._flush_session(db)
            db.commit()
        return new_events

    def _flush_session(self, db, ended: bool = False) -> None:
        sess = db.get(StreamSession, self.session_id)
        if sess:
            sess.frames, sess.anomalous_frames, sess.max_score = self.frames, self.anomalous, self.max_score
            if ended:
                sess.ended_at = utcnow()

    def close(self, analyzer: FrameAnalyzer | None, wall_end: datetime | None = None) -> None:
        with SessionLocal() as db:
            if analyzer is not None:
                for t in analyzer.finish():
                    self._close(db, t)
            self._flush_session(db, ended=True)
            if wall_end is not None:
                sess = db.get(StreamSession, self.session_id)
                sess.ended_at = wall_end
            db.commit()


def as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
