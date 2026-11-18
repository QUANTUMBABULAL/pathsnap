"""PathSnap FastAPI backend (Laptop 1: model + API).

Run from the repository root:

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health            service + model status
    GET  /trips             the reserved evaluation trips (ids + basic GPS stats, no labels)
    POST /predict           {"trip_id": "..."} -> full pipeline + server-side scoring
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from pathsnap.geo import haversine_m  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.pipeline import PathSnapModel  # noqa: E402
from pathsnap.viz import nearby_arcs  # noqa: E402
from scorer import score_prediction  # noqa: E402

# ----------------------------------------------------------------- config
DATA_DIRS = {
    "evaluator_test": Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "evaluator_test")),
    "student_test": Path(os.environ.get("PATHSNAP_STUDENT_DIR", ROOT / "data" / "student_test")),
}
ARTIFACTS = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "models" / "artifacts"))
MAX_GPS_POINTS = 4000
MAX_NETWORK_ARCS = 5000

# ----------------------------------------------------------------- logging
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False

# ----------------------------------------------------------------- app
app = FastAPI(title="PathSnap API", version="1.0.0",
              description="GPS map-matching: candidate classifier + HMM/Viterbi route reconstruction")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL: PathSnapModel | None = None


@app.on_event("startup")
def _load_model():
    global MODEL
    MODEL = PathSnapModel.load(str(ARTIFACTS))
    meta = MODEL.meta
    log.info("model loaded from %s | %s (tuned=%s) | features=%d | HMM alpha=%.2f beta=%.2f",
             ARTIFACTS, meta.get("selected_model"), meta.get("tuned"), len(MODEL.features),
             MODEL.hmm_cfg.alpha, MODEL.hmm_cfg.beta)
    for split, d in DATA_DIRS.items():
        n = len(list_trips(str(d))) if d.exists() else 0
        log.info("data split %-14s -> %s (%d trips)", split, d, n)


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    client = request.client.host if request.client else "?"
    response = await call_next(request)
    log.info("[%s] %s %s %s from %s -> %d in %.0f ms", rid, request.method, request.url.path,
             request.url.query or "", client, response.status_code, 1000 * (time.perf_counter() - t0))
    response.headers["X-Request-ID"] = rid
    return response


def _split_dir(split: str) -> Path:
    if split not in DATA_DIRS:
        raise HTTPException(400, f"unknown split '{split}'")
    d = DATA_DIRS[split]
    if not d.exists():
        raise HTTPException(500, f"data directory missing: {d}")
    return d


@lru_cache(maxsize=4)
def _trip_summaries(split: str):
    d = _split_dir(split)
    out = []
    for tid in list_trips(str(d)):
        raw = load_trip(str(d), tid)  # GPS + network only, no reference route
        tr = raw.track
        km = float(haversine_m(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1]).sum() / 1000)
        out.append({"trip_id": tid, "n_points": int(len(tr)), "duration_s": float(tr[-1, 2] - tr[0, 2]),
                    "length_km": round(km, 2), "n_arcs": int(len(raw.arcs)),
                    "center": [float(tr[:, 1].mean()), float(tr[:, 0].mean())]})
    return out


class PredictRequest(BaseModel):
    trip_id: str
    split: str = "evaluator_test"


@app.get("/health")
def health():
    return {"status": "ok" if MODEL is not None else "loading",
            "model": MODEL.meta.get("selected_model") if MODEL else None,
            "tuned": MODEL.meta.get("tuned") if MODEL else None,
            "hmm": MODEL.hmm_cfg.to_dict() if MODEL else None,
            "candidate_config": MODEL.cand_cfg.__dict__ if MODEL else None}


@app.get("/trips")
def trips(split: str = "evaluator_test"):
    return {"split": split, "trips": _trip_summaries(split)}


@app.post("/predict")
def predict(req: PredictRequest):
    d = _split_dir(req.split)
    if req.trip_id not in {t["trip_id"] for t in _trip_summaries(req.split)}:
        raise HTTPException(404, f"trip '{req.trip_id}' not found in {req.split}")
    log.info("predict request: split=%s trip=%s", req.split, req.trip_id)

    raw = load_trip(str(d), req.trip_id)
    pred = MODEL.predict_trip(raw)  # prediction pipeline: no reference data involved
    t0 = time.perf_counter()
    metrics = score_prediction(str(d), req.trip_id, pred)  # server-side scoring afterwards
    score_s = time.perf_counter() - t0

    ctx, net = pred.ctx, pred.ctx.net
    n = len(ctx.xy)
    step = max(1, int(np.ceil(n / MAX_GPS_POINTS)))
    gps = [[round(float(la), 7), round(float(lo), 7)] for lo, la in ctx.track[::step, :2]]
    # snapped positions on the chosen arcs (HMM output)
    a = pred.hmm_arcs
    sx = net.A[a, 0] + pred.hmm_choice_t * (net.B[a, 0] - net.A[a, 0])
    sy = net.A[a, 1] + pred.hmm_choice_t * (net.B[a, 1] - net.A[a, 1])
    slon, slat = ctx.proj.inverse(sx, sy)
    matched = [[round(float(la), 7), round(float(lo), 7)] for la, lo in zip(slat[::step], slon[::step])]
    near = nearby_arcs(ctx, buffer=150, max_arcs=MAX_NETWORK_ARCS)
    lat, lon = ctx.track[:, 1], ctx.track[:, 0]

    log.info("trip=%s points=%d candidates=%d | Top-1 HMM=%.4f classifier=%.4f | route F1=%.4f P=%.4f R=%.4f | "
             "continuity=%.3f | gaps filled=%d unfilled=%d | pipeline %.2fs + scoring %.2fs",
             req.trip_id, n, len(pred.cand), metrics["top1_accuracy_hmm"], metrics["top1_accuracy_classifier"],
             metrics["route_f1"], metrics["route_precision"], metrics["route_recall"], metrics["route_continuity"],
             pred.gaps_filled, pred.gaps_unfilled, pred.timings["total_s"], score_s)

    return {
        "trip_id": req.trip_id,
        "split": req.split,
        "model": MODEL.meta.get("selected_model"),
        "n_points": n,
        "n_candidates": int(len(pred.cand)),
        "bounds": [[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]],
        "gps": gps,
        "matched_points": matched,
        "predicted_route": net.route_polyline(pred.route),
        "n_route_arcs": int(len(pred.route)),
        "network": net.arc_lonlat(near),
        "metrics": metrics,
        "timings": {**{k: round(v, 4) for k, v in pred.timings.items()}, "scoring_s": round(score_s, 4)},
        "gaps": {"filled": pred.gaps_filled, "unfilled": pred.gaps_unfilled},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
