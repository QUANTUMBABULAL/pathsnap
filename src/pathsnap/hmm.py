"""HMM + Viterbi route reconstruction.

Hidden states
    At GPS time step i the hidden state is "the vehicle is on candidate arc c"
    for c in the candidate set C_i (at most k arcs). Each state keeps its arc id
    and the projection of the GPS point onto that arc, so the
    candidate <-> road-segment correspondence is preserved end to end.

Emission
    log P(z_i | c) = alpha * log(p_c + eps), where p_c is the selected
    classifier's probability for that candidate. The classifier already
    combines distance, heading, context and topology, so it replaces the classic
    Gaussian-distance emission.

Transition (Newson & Krumm style)
    The network distance d_route travelled between the projection on arc a at
    step i-1 and the projection on arc b at step i is compared with the
    straight-line distance d_gc between the two GPS fixes:

        log P(b | a) = -|d_route - d_gc| / beta_eff

    * a == b moving forward: d_route = (t_b - t_a) * len(a)
    * a == b moving backward (GPS jitter): penalised by ``backtrack_factor``
    * a != b: remaining part of a + shortest path(to(a) -> from(b)) + part of b
    beta_eff grows with the time gap (sqrt(dt)) so that long GPS dropouts allow
    more detour. Shortest paths are bounded Dijkstra searches with a cutoff of
    ``cutoff_factor * d_gc + cutoff_base`` metres and are cached per node.

Disconnected / missing transitions
    If b is unreachable from a within the cutoff the transition gets the
    constant ``unreachable_logp`` instead of -inf, so the chain never breaks:
    Viterbi can still "jump" when the GPS genuinely demands it (e.g. a network
    gap), but only when emissions strongly support it. Points with no candidate
    inside the search radius already received fallback candidates upstream.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .network import RoadNetwork


@dataclass
class HMMConfig:
    alpha: float = 1.0             # emission weight
    beta: float = 5.0              # transition scale (m)
    backtrack_factor: float = 3.0  # cost multiplier for moving backwards on the same arc
    cutoff_factor: float = 3.0
    cutoff_base: float = 150.0
    unreachable_logp: float = -40.0
    eps: float = 1e-4

    def to_dict(self):
        return asdict(self)


def _route_distances(net: RoadNetwork, arcs_a, ta, arcs_b, tb, gc, cfg: HMMConfig):
    """Matrix (len(a), len(b)) of network distances between projected positions."""
    La = net.length[arcs_a]
    Lb = net.length[arcs_b]
    pa = ta * La
    pb = tb * Lb
    D = np.full((len(arcs_a), len(arcs_b)), np.inf)
    cutoff = cfg.cutoff_factor * gc + cfg.cutoff_base
    to_a = net.arcs[arcs_a, 1]
    from_b = net.arcs[arcs_b, 0]
    for i, a in enumerate(arcs_a):
        rest = La[i] - pa[i]
        dist = None
        for j, b in enumerate(arcs_b):
            if a == b:
                step = pb[j] - pa[i]
                D[i, j] = step if step >= 0 else -step * cfg.backtrack_factor
                continue
            if dist is None:
                dist, _ = net.dijkstra(int(to_a[i]), cutoff)
            nd = dist.get(int(from_b[j]))
            if nd is not None:
                D[i, j] = rest + nd + pb[j]
    return D


def viterbi(net: RoadNetwork, xy: np.ndarray, t: np.ndarray, cand_arcs: list[np.ndarray],
            cand_t: list[np.ndarray], cand_p: list[np.ndarray], cfg: HMMConfig | None = None):
    """Decode the most likely candidate sequence.

    Returns (chosen index per step into cand_arcs[i], chosen arc per step, n_unreachable_steps).
    """
    cfg = cfg or HMMConfig()
    n = len(cand_arcs)
    emis = [cfg.alpha * np.log(np.asarray(p) + cfg.eps) for p in cand_p]
    score = emis[0].copy()
    backptr: list[np.ndarray] = [np.zeros(len(cand_arcs[0]), dtype=np.int64)]
    n_unreach = 0
    for i in range(1, n):
        gc = float(np.hypot(*(xy[i] - xy[i - 1])))
        dt = max(float(t[i] - t[i - 1]), 1.0)
        beta = cfg.beta * np.sqrt(dt)
        D = _route_distances(net, cand_arcs[i - 1], cand_t[i - 1], cand_arcs[i], cand_t[i], gc, cfg)
        trans = np.where(np.isfinite(D), -np.abs(D - gc) / beta, cfg.unreachable_logp)
        if not np.isfinite(D).any():
            n_unreach += 1
        tot = score[:, None] + trans
        bp = np.argmax(tot, axis=0)
        score = tot[bp, np.arange(tot.shape[1])] + emis[i]
        score -= score.max()  # keep numbers bounded
        backptr.append(bp)
    choice = np.empty(n, dtype=np.int64)
    choice[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        choice[i - 1] = backptr[i][choice[i]]
    arcs = np.array([cand_arcs[i][choice[i]] for i in range(n)], dtype=np.int64)
    return choice, arcs, n_unreach


def reconstruct_route(net: RoadNetwork, point_arcs: np.ndarray, max_fill: float = 3000.0):
    """Turn per-point arcs into a continuous arc route.

    Consecutive duplicate arcs are collapsed; when two consecutive matched arcs
    are not directly connected, the gap is filled with the shortest network path
    (if one exists within ``max_fill`` metres). Returns (route arcs, n_gaps_filled,
    n_gaps_unfilled).
    """
    seq = [int(point_arcs[0])]
    for a in point_arcs[1:]:
        if int(a) != seq[-1]:
            seq.append(int(a))
    route = [seq[0]]
    filled = unfilled = 0
    for a in seq[1:]:
        prev = route[-1]
        if net.arcs[prev, 1] == net.arcs[a, 0]:
            route.append(a)
            continue
        path = net.path_arcs(int(net.arcs[prev, 1]), int(net.arcs[a, 0]), max_fill)
        if path is not None:
            route.extend(path)
            filled += 1
        else:
            unfilled += 1
        route.append(a)
    return np.array(route, dtype=np.int64), filled, unfilled
