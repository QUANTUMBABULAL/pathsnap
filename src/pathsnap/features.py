"""Feature extraction for (GPS point, candidate arc) pairs.

Problem framing
---------------
Map matching is cast as *candidate ranking*: every GPS observation gets a small
set of nearby candidate arcs, and a binary classifier estimates
P(candidate is the arc the vehicle was on | features). The per-point argmax is
the classifier-only prediction; the probabilities are the HMM emission scores.

Feature groups (all computed without any reference-route information)
---------------------------------------------------------------------
Geometric      dist, dist_rank, dist_delta, dist_ratio, n_cand, proj_frac, off_end
Directional    head_diff, head_cos, head_reliab (heading weighted by how far the
               vehicle actually moved - headings of a stopped car are noise)
Kinematic      speed, dt_prev, turn_rate
Context window win_dist_mean / win_dist_max: distance of the neighbouring GPS
               points (i-3 .. i+3) to the same arc. A single noisy fix can be
               closer to a parallel street; its neighbours usually are not.
               win_head_diff: heading agreement over the window.
Topology       arc_len, out_deg, in_deg, has_reverse, conn_prev / conn_next
               (is this candidate the same as / adjacent to the nearest arc of
               the previous / next GPS point).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidates import CandidateIndex
from .geo import LocalProjection, angle_diff_deg, bearing_deg, point_segment
from .io import RawTrip, clean_track
from .network import RoadNetwork

FEATURES = [
    "dist", "dist_rank", "dist_delta", "dist_ratio", "n_cand", "proj_frac", "off_end",
    "head_diff", "head_cos", "head_reliab",
    "speed", "dt_prev", "turn_rate",
    "win_dist_mean", "win_dist_max", "win_head_diff",
    "arc_len", "out_deg", "in_deg", "has_reverse", "conn_prev", "conn_next",
]


@dataclass
class CandidateConfig:
    radius: float = 50.0     # metres
    k: int = 8               # max candidates per GPS point
    spacing: float = 10.0    # densification step for the spatial index
    heading_window: int = 2  # i-w .. i+w for the GPS heading
    context_window: int = 3  # i-c .. i+c for window features


@dataclass
class TripContext:
    trip_id: str
    track: np.ndarray        # cleaned (N, 3) lon, lat, t
    kept_idx: np.ndarray     # row indices of the cleaned track in the raw file
    proj: LocalProjection
    net: RoadNetwork
    xy: np.ndarray           # (N, 2) projected GPS
    t: np.ndarray            # (N,)


def make_context(raw: RawTrip) -> TripContext:
    track, kept = clean_track(raw.track)
    proj = LocalProjection.from_points(track[:, 0], track[:, 1])
    net = RoadNetwork(raw.nodes, raw.arcs, proj)
    x, y = proj.forward(track[:, 0], track[:, 1])
    return TripContext(raw.trip_id, track, kept, proj, net, np.column_stack([x, y]), track[:, 2].astype(float))


def gps_kinematics(xy: np.ndarray, t: np.ndarray, w: int = 2):
    """Heading (deg), displacement (m) over i-w..i+w, speed (m/s), dt to previous fix, turn rate."""
    n = len(xy)
    lo = np.clip(np.arange(n) - w, 0, n - 1)
    hi = np.clip(np.arange(n) + w, 0, n - 1)
    d = xy[hi] - xy[lo]
    disp = np.hypot(d[:, 0], d[:, 1])
    heading = bearing_deg(d[:, 0], d[:, 1])
    span = np.maximum(t[hi] - t[lo], 1e-6)
    speed = np.where(hi > lo, disp / span, 0.0)
    dt_prev = np.concatenate([[1.0], np.diff(t)])
    turn = np.concatenate([[0.0], angle_diff_deg(heading[1:], heading[:-1])]) / np.maximum(dt_prev, 1.0)
    return heading, disp, speed, dt_prev, turn


def extract_features(ctx: TripContext, cfg: CandidateConfig | None = None):
    """Build the candidate table for a trip.

    Returns
    -------
    df : DataFrame with one row per (point, candidate) - columns ``pt``, ``arc``,
         ``t_proj`` (projection parameter, used by the HMM) and every name in FEATURES.
    """
    cfg = cfg or CandidateConfig()
    net, xy, t = ctx.net, ctx.xy, ctx.t
    n = len(xy)
    index = CandidateIndex(net, xy, spacing=cfg.spacing)
    cands = index.query(xy, radius=cfg.radius, k=cfg.k)

    counts = np.array([len(c[0]) for c in cands])
    pt = np.repeat(np.arange(n), counts)
    arc = np.concatenate([c[0] for c in cands]).astype(np.int64)
    dist = np.concatenate([c[1] for c in cands])
    tproj = np.concatenate([c[2] for c in cands])
    rank = np.concatenate([np.arange(c) for c in counts])

    # per-point aggregates
    start = np.cumsum(counts) - counts
    dmin = dist[start][pt]
    nearest_arc = arc[start]          # rank-0 candidate of every point

    heading, disp, speed, dt_prev, turn = gps_kinematics(xy, t, cfg.heading_window)
    hd = angle_diff_deg(heading[pt], net.bearing[arc])
    reliab = np.clip(disp[pt] / 10.0, 0.0, 1.0)  # ~1 once the car moved >= 10 m

    L = net.length[arc]
    off_end = np.maximum(0.0, np.maximum(-tproj, tproj - 1.0)) * L

    # context window: neighbours' distance / heading agreement to the same arc
    c = cfg.context_window
    wd = []
    wh = []
    ax, ay, bx, by = net.A[arc, 0], net.A[arc, 1], net.B[arc, 0], net.B[arc, 1]
    for o in range(-c, c + 1):
        j = np.clip(pt + o, 0, n - 1)
        dj, _ = point_segment(xy[j, 0], xy[j, 1], ax, ay, bx, by)
        wd.append(dj)
        wh.append(angle_diff_deg(heading[j], net.bearing[arc]) * np.clip(disp[j] / 10.0, 0, 1))
    wd = np.vstack(wd)
    wh = np.vstack(wh)

    prev_nn = nearest_arc[np.clip(pt - 1, 0, n - 1)]
    next_nn = nearest_arc[np.clip(pt + 1, 0, n - 1)]
    conn_prev = np.where(arc == prev_nn, 2, np.where(net.touching(arc, prev_nn), 1, 0))
    conn_next = np.where(arc == next_nn, 2, np.where(net.touching(arc, next_nn), 1, 0))

    df = pd.DataFrame({
        "pt": pt,
        "arc": arc,
        "t_proj": np.clip(tproj, 0.0, 1.0),
        "dist": dist,
        "dist_rank": rank,
        "dist_delta": dist - dmin,
        "dist_ratio": dist / (dmin + 1.0),
        "n_cand": counts[pt],
        "proj_frac": np.clip(tproj, 0.0, 1.0),
        "off_end": off_end,
        "head_diff": hd,
        "head_cos": np.cos(np.radians(hd)),
        "head_reliab": hd * reliab,
        "speed": speed[pt],
        "dt_prev": dt_prev[pt],
        "turn_rate": turn[pt],
        "win_dist_mean": wd.mean(axis=0),
        "win_dist_max": wd.max(axis=0),
        "win_head_diff": wh.mean(axis=0),
        "arc_len": L,
        "out_deg": net.out_deg[net.arcs[arc, 1]],
        "in_deg": net.in_deg[net.arcs[arc, 0]],
        "has_reverse": (net.reverse[arc] >= 0).astype(np.int8),
        "conn_prev": conn_prev.astype(np.int8),
        "conn_next": conn_next.astype(np.int8),
    })
    return df
