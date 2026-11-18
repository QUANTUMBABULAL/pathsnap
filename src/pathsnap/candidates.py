"""Candidate generation: for each GPS point, the road arcs it could lie on.

Arcs are straight segments, some of them several kilometres long, so indexing
only their endpoints would miss long roads. We *densify* every arc near the
trajectory into sample points (one every ``spacing`` metres), build a KD-tree on
those samples, and then compute the exact point-to-segment distance for every
arc that has a sample within ``radius + spacing/2`` of the GPS point.

The ``k`` nearest arcs within ``radius`` are kept. If a point has no arc inside
the radius (large GPS error, sparse network) we fall back to its ``k`` nearest
arcs regardless of distance, so every observation always has candidates and
the HMM chain never breaks for lack of states.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geo import point_segment
from .network import RoadNetwork


class CandidateIndex:
    def __init__(self, net: RoadNetwork, pts_xy: np.ndarray, margin: float = 400.0, spacing: float = 10.0):
        self.net = net
        self.spacing = spacing
        lo = pts_xy.min(axis=0) - margin
        hi = pts_xy.max(axis=0) + margin
        A, B = net.A, net.B
        amin = np.minimum(A, B)
        amax = np.maximum(A, B)
        inside = np.all(amax >= lo, axis=1) & np.all(amin <= hi, axis=1)
        self.arc_ids = np.flatnonzero(inside)
        if len(self.arc_ids) == 0:  # degenerate - use everything
            self.arc_ids = np.arange(net.n_arcs)

        L = net.length[self.arc_ids]
        n = np.maximum(2, np.ceil(L / spacing).astype(np.int64) + 1)
        owner = np.repeat(np.arange(len(self.arc_ids)), n)
        start = np.repeat(np.cumsum(n) - n, n)
        frac = (np.arange(n.sum()) - start) / np.repeat(n - 1, n)
        a = A[self.arc_ids][owner]
        b = B[self.arc_ids][owner]
        samples = a + (b - a) * frac[:, None]
        self.sample_owner = self.arc_ids[owner]
        self.tree = cKDTree(samples)

    def query(self, pts_xy: np.ndarray, radius: float = 50.0, k: int = 8):
        """Return a list (one entry per point) of (arc_ids, distances, t) sorted by distance."""
        net = self.net
        reach = radius + self.spacing
        hits = self.tree.query_ball_point(pts_xy, r=reach)
        out = []
        for i, h in enumerate(hits):
            p = pts_xy[i]
            if h:
                arcs = np.unique(self.sample_owner[h])
            else:
                arcs = np.array([], dtype=np.int64)
            if len(arcs):
                d, t = point_segment(p[0], p[1], net.A[arcs, 0], net.A[arcs, 1], net.B[arcs, 0], net.B[arcs, 1])
                ok = d <= radius
                arcs, d, t = arcs[ok], d[ok], t[ok]
            if len(arcs) == 0:  # fallback: nearest arcs whatever the distance
                kk = min(len(self.sample_owner), 25 * k)
                _, si = self.tree.query(p, k=kk)
                arcs = np.unique(self.sample_owner[np.atleast_1d(si)])
                d, t = point_segment(p[0], p[1], net.A[arcs, 0], net.A[arcs, 1], net.B[arcs, 0], net.B[arcs, 1])
            o = np.argsort(d, kind="stable")[:k]
            out.append((arcs[o], d[o], t[o]))
        return out
