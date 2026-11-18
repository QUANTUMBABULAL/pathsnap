"""Evaluation metrics: candidate-level, point-level and route-level."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from .network import RoadNetwork


def candidate_metrics(y_true, p, threshold: float = 0.5) -> dict:
    """Binary candidate classification metrics (macro averaged over the two classes)."""
    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    yhat = (p >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, yhat),
        "precision_macro": precision_score(y_true, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, yhat, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
    }


def top1_arcs(pt: np.ndarray, arc: np.ndarray, p: np.ndarray, n_points: int) -> np.ndarray:
    """Per-point arc with the highest score (classifier-only prediction)."""
    order = np.lexsort((-p, pt))
    pt_s = pt[order]
    first = np.ones(len(pt_s), dtype=bool)
    first[1:] = pt_s[1:] != pt_s[:-1]
    out = np.full(n_points, -1, dtype=np.int64)
    out[pt_s[first]] = arc[order][first]
    return out


def top1_accuracy(df, p, ref_arc) -> float:
    pred = top1_arcs(df["pt"].to_numpy(), df["arc"].to_numpy(), np.asarray(p), len(ref_arc))
    return float(np.mean(pred == ref_arc))


def candidate_miss_rate(df, ref_arc) -> float:
    has = np.zeros(len(ref_arc), dtype=bool)
    lab = df["arc"].to_numpy() == ref_arc[df["pt"].to_numpy()]
    has[df["pt"].to_numpy()[lab]] = True
    return float(1.0 - has.mean())


def point_continuity(net: RoadNetwork, point_arcs: np.ndarray) -> float:
    """Share of arc changes between consecutive GPS points that follow a real road connection."""
    a, b = point_arcs[:-1], point_arcs[1:]
    change = a != b
    if not change.any():
        return 1.0
    return float(net.connected(a[change], b[change]).mean())


def route_continuity(net: RoadNetwork, route: np.ndarray) -> float:
    if len(route) < 2:
        return 1.0
    return float(net.connected(route[:-1], route[1:]).mean())


def route_overlap(net: RoadNetwork, pred_route: np.ndarray, ref_route: np.ndarray) -> dict:
    """Length-weighted precision / recall / F1 between predicted and reference arc sets."""
    P = set(map(int, pred_route))
    R = set(map(int, ref_route))
    inter = P & R
    lp = net.length[list(P)].sum() if P else 0.0
    lr = net.length[list(R)].sum() if R else 0.0
    li = net.length[list(inter)].sum() if inter else 0.0
    prec = li / lp if lp > 0 else 0.0
    rec = li / lr if lr > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"route_precision": float(prec), "route_recall": float(rec), "route_f1": float(f1)}
