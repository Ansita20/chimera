import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="chimera-test-")
os.environ.setdefault("CHIMERA_DATA_DIR", _tmp)
os.environ.setdefault("CHIMERA_DEVICE", "cpu")
os.environ.setdefault("CHIMERA_AE_EPOCHS", "3")
os.environ.setdefault("CHIMERA_TEMPORAL_EPOCHS", "2")
os.environ.setdefault("CHIMERA_MC_SAMPLES", "2")
os.environ.setdefault("CHIMERA_AE_MIN_STEPS", "1")
os.environ.setdefault("CHIMERA_TEMPORAL_MIN_STEPS", "1")
os.environ.setdefault("CHIMERA_CORESET_MAX", "1000")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
