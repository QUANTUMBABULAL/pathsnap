"""Readers for the raw PathSnap trip files.

Each trip lives in its own folder ``<split>/<trip_id>/`` containing

* ``<id>.track`` - lon, lat, timestamp(s)            (GPS observations)
* ``<id>.nodes`` - lon, lat                           (row index = node id)
* ``<id>.arcs``  - from_node, to_node                 (row index = arc id)
* ``<id>.route`` - chronological ground-truth arc ids (reference only)

The ``.route`` file is deliberately *not* read by :func:`load_trip`; it is only
read through :func:`load_route`, which the training labels and the server-side
evaluator use. The prediction pipeline never calls it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RawTrip:
    trip_id: str
    track: np.ndarray  # (N, 3) lon, lat, t
    nodes: np.ndarray  # (Nn, 2) lon, lat
    arcs: np.ndarray   # (Na, 2) int from, to


def _read(path, dtype=float, ncols=None):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=dtype)
    arr = df.to_numpy()
    if ncols is not None and arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def trip_dir(root: str, trip_id: str) -> str:
    return os.path.join(root, trip_id)


def list_trips(root: str) -> list[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def load_trip(root: str, trip_id: str) -> RawTrip:
    d = trip_dir(root, trip_id)
    track = _read(os.path.join(d, f"{trip_id}.track"), float, 3)
    nodes = _read(os.path.join(d, f"{trip_id}.nodes"), float, 2)
    arcs = _read(os.path.join(d, f"{trip_id}.arcs"), np.int64, 2)
    return RawTrip(trip_id, track, nodes, arcs)


def load_route(root: str, trip_id: str) -> np.ndarray:
    """Reference arc sequence. Only for label derivation / evaluation."""
    path = os.path.join(trip_dir(root, trip_id), f"{trip_id}.route")
    return _read(path, np.int64, 1).ravel()


def clean_track(track: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Basic GPS cleaning.

    Removes rows with non-finite values, implausible coordinates, and
    non-increasing timestamps (duplicates). Returns the cleaned track and the
    indices of the kept rows in the original file.
    """
    keep = np.all(np.isfinite(track), axis=1)
    keep &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    idx = np.flatnonzero(keep)
    tr = track[idx]
    if len(tr) > 1:
        mono = np.concatenate([[True], np.diff(tr[:, 2]) > 0])
        idx, tr = idx[mono], tr[mono]
    return tr, idx
