"""End-to-end inference: raw trip -> candidates -> features -> classifier -> HMM -> route.

The same code path is used by the training notebook (student-test evaluation)
and by the FastAPI server, so there is no train/serve skew. Nothing here reads
``.route`` files.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import joblib
import numpy as np

from .evaluation import top1_arcs
from .features import FEATURES, CandidateConfig, TripContext, extract_features, make_context
from .hmm import HMMConfig, reconstruct_route, viterbi
from .io import RawTrip


@dataclass
class TripPrediction:
    ctx: TripContext
    cand: "object"                  # candidate DataFrame with column ``p``
    classifier_arcs: np.ndarray     # per-point argmax
    hmm_arcs: np.ndarray            # per-point Viterbi output
    route: np.ndarray               # continuous reconstructed arc route
    hmm_choice_t: np.ndarray        # projection parameter of the chosen candidate
    timings: dict = field(default_factory=dict)
    gaps_filled: int = 0
    gaps_unfilled: int = 0


def decode_candidates(ctx: TripContext, df, p: np.ndarray, hmm_cfg: HMMConfig):
    """Run Viterbi over a trip's candidate table (rows ordered by ``pt``).

    Returns (arc per point, projection parameter of the chosen candidate per point).
    """
    pt = df["pt"].to_numpy()
    bounds = np.flatnonzero(np.diff(pt)) + 1
    cand_arcs = np.split(df["arc"].to_numpy(), bounds)
    cand_t = np.split(df["t_proj"].to_numpy(), bounds)
    cand_p = np.split(np.asarray(p), bounds)
    choice, arcs, _ = viterbi(ctx.net, ctx.xy, ctx.t, cand_arcs, cand_t, cand_p, hmm_cfg)
    chosen_t = np.array([cand_t[i][c] for i, c in enumerate(choice)])
    return arcs, chosen_t


class PathSnapModel:
    def __init__(self, model, features=None, cand_cfg: CandidateConfig | None = None,
                 hmm_cfg: HMMConfig | None = None, meta: dict | None = None):
        self.model = model
        self.features = list(features or FEATURES)
        self.cand_cfg = cand_cfg or CandidateConfig()
        self.hmm_cfg = hmm_cfg or HMMConfig()
        self.meta = meta or {}

    # ---------------------------------------------------------------- io
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, "final_model.joblib"))
        cfg = {
            "features": self.features,
            "candidate_config": self.cand_cfg.__dict__,
            "hmm_config": self.hmm_cfg.to_dict(),
            "meta": self.meta,
        }
        with open(os.path.join(directory, "pipeline_config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    @classmethod
    def load(cls, directory: str) -> "PathSnapModel":
        model = joblib.load(os.path.join(directory, "final_model.joblib"))
        with open(os.path.join(directory, "pipeline_config.json")) as f:
            cfg = json.load(f)
        return cls(model, cfg["features"], CandidateConfig(**cfg["candidate_config"]),
                   HMMConfig(**cfg["hmm_config"]), cfg.get("meta", {}))

    # ----------------------------------------------------------- predict
    def predict_proba(self, df) -> np.ndarray:
        return self.model.predict_proba(df[self.features].to_numpy(dtype=np.float32))[:, 1]

    def predict_trip(self, raw: RawTrip, use_hmm: bool = True) -> TripPrediction:
        tm = {}
        t0 = time.perf_counter()
        ctx = make_context(raw)
        tm["preprocess_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        df = extract_features(ctx, self.cand_cfg)
        tm["features_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        p = self.predict_proba(df)
        df["p"] = p
        tm["classifier_s"] = time.perf_counter() - t0

        n = len(ctx.xy)
        pt = df["pt"].to_numpy()
        arc = df["arc"].to_numpy()
        cls_arcs = top1_arcs(pt, arc, p, n)

        t0 = time.perf_counter()
        if use_hmm:
            hmm_arcs, chosen_t = decode_candidates(ctx, df, p, self.hmm_cfg)
        else:
            hmm_arcs = cls_arcs.copy()
            bounds = np.flatnonzero(np.diff(pt)) + 1
            cand_t, cand_p = np.split(df["t_proj"].to_numpy(), bounds), np.split(p, bounds)
            chosen_t = np.array([cand_t[i][np.argmax(cand_p[i])] for i in range(n)])
        tm["hmm_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        route, filled, unfilled = reconstruct_route(ctx.net, hmm_arcs)
        tm["route_s"] = time.perf_counter() - t0
        tm["total_s"] = sum(tm.values())
        return TripPrediction(ctx, df, cls_arcs, hmm_arcs, route, chosen_t, tm, filled, unfilled)
