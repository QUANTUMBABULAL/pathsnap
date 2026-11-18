"""Geographic helpers.

The dataset contains trips from many countries, each covering at most a few tens
of kilometres. For every trip we therefore use a *local equirectangular
projection* centred on the trip's GPS centroid:

    x = R * (lon - lon0) * cos(lat0)
    y = R * (lat - lat0)

At city scale (< ~50 km) the distortion of this projection is well below GPS noise
(a few metres), and it lets every later step use fast Euclidean geometry in metres
instead of degree-based distances (which are not physically meaningful).
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


class LocalProjection:
    """Equirectangular projection around a reference point (lon0, lat0)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._kx = EARTH_RADIUS_M * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self._ky = EARTH_RADIUS_M * np.pi / 180.0

    @classmethod
    def from_points(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjection":
        return cls(float(np.mean(lon)), float(np.mean(lat)))

    def forward(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return (lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky

    def inverse(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return x / self._kx + self.lon0, y / self._ky + self.lat0


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (used to validate the projection)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, clockwise) of a vector in projected metres."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (all broadcastable arrays).

    Returns (distance, t) where t is the *unclipped* projection parameter
    (t in [0, 1] means the foot of the perpendicular lies on the segment).
    """
    vx, vy = bx - ax, by - ay
    len2 = vx * vx + vy * vy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * vx + (py - ay) * vy) / len2
    t = np.where(len2 > 0, t, 0.0)
    tc = np.clip(t, 0.0, 1.0)
    qx, qy = ax + tc * vx, ay + tc * vy
    return np.hypot(px - qx, py - qy), t
