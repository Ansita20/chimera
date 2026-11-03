import numpy as np
import pytest

from chimera.config import get_settings
from chimera.metrics import au_pro, safe_auroc
from chimera.models import AutoencoderDetector, EmbeddingDetector, TemporalDetector
from chimera.tracking import RegionTracker, extract_regions
from synthetic import frame


@pytest.fixture(scope="module")
def s():
    return get_settings()


@pytest.fixture(scope="module")
def normal():
    return [frame(t) for t in range(40)]


@pytest.mark.parametrize("cls", [EmbeddingDetector, AutoencoderDetector])
def test_appearance_detector_ranks_anomaly_higher(cls, s, normal, tmp_path):
    det = cls(s)
    det.fit(normal)
    good = det.predict(frame(100))
    bad = det.predict(frame(100, anomaly=True))
    assert bad.score > good.score
    assert 0 <= bad.uncertainty <= 1
    # heatmap peak should fall on the inserted blob (upper-right quadrant)
    h, w = bad.heatmap.shape
    y, x = np.unravel_index(np.argmax(bad.heatmap), bad.heatmap.shape)
    if cls is EmbeddingDetector:
        assert x > w / 2 and y < h * 0.6
    p = tmp_path / "m.pt"
    det.save(p)
    again = cls(s).load(p).predict(frame(100, anomaly=True))
    assert abs(again.score - bad.score) < 0.5 * max(1.0, bad.score)


def test_temporal_detector_runs(s):
    seq = [frame(t) for t in range(30)]
    det = TemporalDetector(s)
    det.fit_sequences([seq])
    st = det.new_state()
    outs = [det.step(st, frame(t, anomaly=(t >= 40))) for t in range(35, 42)]
    assert outs[0] is None and outs[-1] is not None


def test_tracker_confirms_persistent_region():
    heat = np.zeros((60, 80), np.float32)
    heat[10:20, 50:65] = 3.0
    tr = RegionTracker(min_hits=3)
    confirmed_any = False
    for i in range(4):
        regions, _ = extract_regions(heat)
        _, conf, _ = tr.update(regions, ts=i * 0.1, frame_idx=i, uncertainty=0.1)
        confirmed_any |= bool(conf)
    assert confirmed_any and len(tr.tracks) == 1


def test_metrics():
    assert safe_auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    m = np.zeros((20, 20), bool)
    m[5:10, 5:10] = True
    assert au_pro([m], [m.astype(float)]) > 0.95
