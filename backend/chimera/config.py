from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden with a CHIMERA_<NAME> env var."""

    model_config = SettingsConfigDict(env_prefix="CHIMERA_", env_file=".env", extra="ignore")

    data_dir: Path = Path("./data")
    database_url: str = ""  # empty -> sqlite file inside data_dir
    device: str = "auto"  # auto | cpu | cuda | mps
    cors_origins: list[str] = ["*"]

    # Feature-embedding detector (PatchCore-style)
    backbone: str = "resnet18"  # resnet18 | resnet50 | wide_resnet50_2
    embed_size: int = 224
    embed_proj_dim: int = 256
    coreset_max: int = 8000
    coreset_presample: int = 40000
    bank_partitions: int = 4

    # Autoencoder
    ae_size: int = 128
    ae_epochs: int = 40
    ae_latent: int = 64
    ae_min_steps: int = 1500  # small normal sets still get enough optimisation

    # Temporal (future-frame prediction)
    temporal_size: int = 128
    temporal_window: int = 4
    temporal_epochs: int = 25
    temporal_min_steps: int = 600

    # Uncertainty / calibration
    mc_samples: int = 6
    calib_fraction: float = 0.15
    calib_margin: float = 0.25

    # Streaming
    track_min_hits: int = 3
    track_max_misses: int = 6
    min_region_frac: float = 0.002

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'chimera.db').resolve()}"

    @property
    def scenes_dir(self) -> Path:
        return self.data_dir / "scenes"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.scenes_dir.mkdir(parents=True, exist_ok=True)
    s.snapshots_dir.mkdir(parents=True, exist_ok=True)
    return s


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple MPS is opt-in (CHIMERA_DEVICE=mps): some torch/macOS combinations abort the whole
    # process with an uncaught NSException inside MPS convolutions instead of raising.
    return torch.device("cpu")
