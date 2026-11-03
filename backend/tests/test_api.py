import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from chimera.api.app import app
from synthetic import frame


def jpg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()


def wait(c, job_id, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.5)
    raise TimeoutError


def test_end_to_end():
    with TestClient(app) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        sc = c.post("/api/scenes", json={"name": "Test Bench"}).json()
        sid = sc["id"]
        # capture a normal sequence over the websocket, like the webcam UI does
        with c.websocket_connect(f"/api/scenes/{sid}/stream") as ws:
            ws.send_json({"type": "capture"})
            assert ws.receive_json()["type"] == "capturing"
            for t in range(30):
                ws.send_bytes(jpg(frame(t)))
                assert ws.receive_json()["type"] == "captured"
            ws.send_json({"type": "stop"})
            assert ws.receive_json()["captured"] == 30
        j = wait(c, c.post(f"/api/scenes/{sid}/train", json={"models": ["embedding", "temporal"]}).json()["job_id"])
        assert j["status"] == "done", j
        assert c.get(f"/api/scenes/{sid}").json()["models"]["embedding"]["trained"]

        r = c.post(f"/api/scenes/{sid}/detect", files={"file": ("a.jpg", jpg(frame(90, anomaly=True)), "image/jpeg")}).json()
        assert r["score"] >= 1.0 and r["regions"], r["score"]
        assert r["regions"][0]["explanation"]["reasons"]

        with c.websocket_connect(f"/api/scenes/{sid}/stream") as ws:
            ws.send_json({"type": "start"})
            started = ws.receive_json()
            assert started["type"] == "started"
            last = None
            for t in range(12):
                ws.send_bytes(jpg(frame(60 + t, anomaly=t >= 4)))
                last = ws.receive_json()
            assert last["type"] == "result" and last["verdict"] in ("anomalous", "uncertain")
            ws.send_json({"type": "stop"})
            sess = ws.receive_json()["session_id"]
        rep = c.get(f"/api/sessions/{sess}/report").json()
        assert rep["stats"]["events"] >= 1
        ev = rep["events"][0]
        assert c.get(ev["snapshot_url"]).status_code == 200
        ov = c.get("/api/reports/overview").json()
        assert ov["totals"]["events"] >= 2
