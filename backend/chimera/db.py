"""SQLAlchemy models. PostgreSQL in Docker; SQLite fallback for local development."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Scene(Base):
    __tablename__ = "scenes"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    models: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # name -> meta / calibration


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32))  # train | video | benchmark
    scene_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|error
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class StreamSession(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scene_id: Mapped[str] = mapped_column(String(64), ForeignKey("scenes.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(200))  # webcam | video:<name> | upload
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    frames: Mapped[int] = mapped_column(Integer, default=0)
    anomalous_frames: Mapped[int] = mapped_column(Integer, default=0)
    max_score: Mapped[float] = mapped_column(Float, default=0.0)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    scene_id: Mapped[str] = mapped_column(String(64), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    frame_start: Mapped[int] = mapped_column(Integer, default=0)
    frame_end: Mapped[int] = mapped_column(Integer, default=0)
    peak_score: Mapped[float] = mapped_column(Float, default=0.0)
    uncertainty: Mapped[float] = mapped_column(Float, default=0.0)
    bbox: Mapped[list[Any]] = mapped_column(JSON, default=list)
    models: Mapped[list[Any]] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(String(300), default="")
    explanation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    snapshot: Mapped[str] = mapped_column(String(500), default="")
    status: Mapped[str] = mapped_column(String(16), default="open")  # open | closed


class Sample(Base):
    """Down-sampled per-frame timeline for reports."""

    __tablename__ = "samples"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    t: Mapped[float] = mapped_column(Float)  # seconds since session start
    frame_idx: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    uncertainty: Mapped[float] = mapped_column(Float)
    n_regions: Mapped[int] = mapped_column(Integer, default=0)
    per_model: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class BenchmarkRun(Base):
    __tablename__ = "benchmarks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    results: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # model -> metrics


_settings = get_settings()
engine = create_engine(
    _settings.db_url,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if _settings.db_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
