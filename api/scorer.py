"""Server-side scorer (Laptop 1 only).

This is the ONLY place in the API that reads reference ``.route`` files. It runs
*after* the prediction pipeline has finished, receives the finished prediction,
and returns aggregate metrics. Reference geometry and per-point labels are never
returned to the client and never passed to the model.
"""
from __future__ import annotations

import numpy as np

from pathsnap.evaluation import point_continuity, route_continuity, route_overlap
from pathsnap.io import load_route
from pathsnap.labels import align_route
from pathsnap.pipeline import TripPrediction


def score_prediction(data_root: str, trip_id: str, pred: TripPrediction) -> dict:
    ctx = pred.ctx
    ref_route = load_route(data_root, trip_id)
    ref_arc, _, _ = align_route(ctx.xy, ref_route, ctx.net)

    cand = pred.cand
    in_cands = np.zeros(len(ref_arc), dtype=bool)
    hit = cand["arc"].to_numpy() == ref_arc[cand["pt"].to_numpy()]
    in_cands[cand["pt"].to_numpy()[hit]] = True

    ov = route_overlap(ctx.net, pred.route, ref_route)
    return {
        "top1_accuracy_hmm": float(np.mean(pred.hmm_arcs == ref_arc)),
        "top1_accuracy_classifier": float(np.mean(pred.classifier_arcs == ref_arc)),
        "candidate_miss_rate": float(1.0 - in_cands.mean()),
        "point_continuity_hmm": point_continuity(ctx.net, pred.hmm_arcs),
        "point_continuity_classifier": point_continuity(ctx.net, pred.classifier_arcs),
        "route_continuity": route_continuity(ctx.net, pred.route),
        **ov,
        "n_reference_arcs": int(len(ref_route)),
    }
