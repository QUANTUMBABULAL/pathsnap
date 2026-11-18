"""Reference-label derivation (TRAINING / EVALUATION ONLY).

The ``.route`` file lists the arcs the vehicle traversed, in order, but it does
not say which GPS fix belongs to which arc. We recover a per-point reference arc
with a *monotone alignment*: dynamic programming finds the assignment of points
to route positions that minimises the total point-to-arc distance subject to the
constraint that the route position never moves backwards in time.

    C[i, j] = d(p_i, route[j]) + min_{k <= j} C[i-1, k]

This handles loops / revisits correctly (the same street driven twice maps to
two different route positions) and is robust to single noisy fixes. The labels
inherit the limitations of the reference route itself (it is not independently
verified ground truth), so points whose aligned distance is very large are
flagged as unreliable.

Nothing in this module is imported by the inference pipeline.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment
from .network import RoadNetwork


def align_route(xy: np.ndarray, route: np.ndarray, net: RoadNetwork):
    """Return (ref_arc per point, route position per point, aligned distance per point)."""
    route = np.asarray(route, dtype=np.int64)
    n, m = len(xy), len(route)
    ax, ay = net.A[route, 0], net.A[route, 1]
    bx, by = net.B[route, 0], net.B[route, 1]
    idx = np.arange(m)
    back = np.zeros((n, m), dtype=np.int32)
    d0, _ = point_segment(xy[0, 0], xy[0, 1], ax, ay, bx, by)
    cost = d0
    for i in range(1, n):
        pm = np.minimum.accumulate(cost)
        arg = np.maximum.accumulate(np.where(cost <= pm, idx, 0))
        back[i] = arg
        di, _ = point_segment(xy[i, 0], xy[i, 1], ax, ay, bx, by)
        cost = pm + di
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    ref = route[pos]
    dist, _ = point_segment(xy[:, 0], xy[:, 1], net.A[ref, 0], net.A[ref, 1], net.B[ref, 0], net.B[ref, 1])
    return ref, pos, dist


def attach_labels(df, ref_arc: np.ndarray):
    """Add a binary ``label`` column: 1 if the candidate is the point's reference arc."""
    df = df.copy()
    df["label"] = (df["arc"].to_numpy() == ref_arc[df["pt"].to_numpy()]).astype(np.int8)
    return df
