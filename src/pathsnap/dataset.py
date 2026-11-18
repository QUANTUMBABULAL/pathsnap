"""Build labelled candidate tables for a whole partition (training / student-test only)."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .features import CandidateConfig, extract_features, make_context
from .io import list_trips, load_route, load_trip
from .labels import align_route, attach_labels


def build_trip(root: str, trip_id: str, cfg: CandidateConfig):
    raw = load_trip(root, trip_id)
    ctx = make_context(raw)
    df = extract_features(ctx, cfg)
    route = load_route(root, trip_id)
    ref, pos, adist = align_route(ctx.xy, route, ctx.net)
    df = attach_labels(df, ref)
    df.insert(0, "trip", trip_id)
    pts = pd.DataFrame({
        "trip": trip_id,
        "pt": np.arange(len(ctx.xy)),
        "ref_arc": ref,
        "align_dist": adist,
        "t": ctx.t,
    })
    return df, pts


def build_partition(root: str, cfg: CandidateConfig, n_jobs: int = -1, cache: str | None = None):
    """Returns (candidate table, point table). Optionally cached to ``cache`` (.pkl)."""
    if cache and os.path.exists(cache):
        return pd.read_pickle(cache)
    t0 = time.time()
    trips = list_trips(root)
    res = Parallel(n_jobs=n_jobs)(delayed(build_trip)(root, t, cfg) for t in trips)
    cand = pd.concat([r[0] for r in res], ignore_index=True)
    pts = pd.concat([r[1] for r in res], ignore_index=True)
    has = cand.groupby(["trip", "pt"])["label"].max().rename("ref_in_cands")
    pts = pts.merge(has, on=["trip", "pt"], how="left")
    pts["ref_in_cands"] = pts["ref_in_cands"].fillna(0).astype(bool)
    out = (cand, pts)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pd.to_pickle(out, cache)
    print(f"built {root}: {len(trips)} trips, {len(pts)} points, {len(cand)} candidates in {time.time()-t0:.1f}s")
    return out
