from .autoencoder import AutoencoderDetector
from .base import AnomalyDetector, Detection
from .embedding import EmbeddingDetector
from .temporal import TemporalDetector

DETECTORS: dict[str, type[AnomalyDetector]] = {
    "autoencoder": AutoencoderDetector,
    "embedding": EmbeddingDetector,
    "temporal": TemporalDetector,
}

__all__ = ["AnomalyDetector", "Detection", "AutoencoderDetector", "EmbeddingDetector", "TemporalDetector", "DETECTORS"]
