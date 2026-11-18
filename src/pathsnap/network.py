"""Road-network representation.

Arcs are *directed* straight segments between two nodes. Most streets appear as
two arcs (u->v and v->u), so the direction of travel matters: heading features
and the HMM transition model are what separate an arc from its reverse twin.
"""
from __future__ import annotations

import heapq

import numpy as np

from .geo import LocalProjection, bearing_deg


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        x, y = proj.forward(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.xy = np.column_stack([x, y])
        self.arcs = np.asarray(arcs, dtype=np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(self.arcs)

        u, v = self.arcs[:, 0], self.arcs[:, 1]
        self.A = self.xy[u]
        self.B = self.xy[v]
        d = self.B - self.A
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = bearing_deg(d[:, 0], d[:, 1])

        # CSR adjacency: node -> outgoing arc ids
        order = np.argsort(u, kind="stable")
        self.out_arcs = order
        self.out_ptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.add.at(self.out_ptr, u + 1, 1)
        self.out_ptr = np.cumsum(self.out_ptr)
        self.out_deg = np.diff(self.out_ptr)
        self.in_deg = np.bincount(v, minlength=self.n_nodes)

        # reverse twin of each arc (-1 if the street is one-way)
        key = u * self.n_nodes + v
        rkey = v * self.n_nodes + u
        sorter = np.argsort(key)
        pos = np.searchsorted(key, rkey, sorter=sorter)
        pos = np.clip(pos, 0, len(key) - 1)
        hit = key[sorter[pos]] == rkey
        self.reverse = np.where(hit, sorter[pos], -1)

        self._sp_cache: dict[int, tuple[float, dict, dict]] = {}

    # ------------------------------------------------------------------ #
    def outgoing(self, node: int) -> np.ndarray:
        return self.out_arcs[self.out_ptr[node]:self.out_ptr[node + 1]]

    def dijkstra(self, src: int, cutoff: float):
        """Bounded single-source shortest paths over nodes.

        Returns (dist, pred_arc) dictionaries. Results are cached per source and
        reused whenever a cached search already covers the requested cutoff.
        """
        cached = self._sp_cache.get(src)
        if cached is not None and cached[0] >= cutoff:
            return cached[1], cached[2]
        dist = {src: 0.0}
        pred = {}
        heap = [(0.0, src)]
        out_ptr, out_arcs, arcs, length = self.out_ptr, self.out_arcs, self.arcs, self.length
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, np.inf):
                continue
            for a in out_arcs[out_ptr[n]:out_ptr[n + 1]]:
                nd = d + length[a]
                if nd > cutoff:
                    continue
                m = arcs[a, 1]
                if nd < dist.get(m, np.inf):
                    dist[m] = nd
                    pred[m] = a
                    heapq.heappush(heap, (nd, m))
        if len(self._sp_cache) > 20000:
            self._sp_cache.clear()
        self._sp_cache[src] = (cutoff, dist, pred)
        return dist, pred

    def path_arcs(self, src: int, dst: int, cutoff: float) -> list[int] | None:
        """Arc sequence of the shortest path src -> dst (None if not within cutoff)."""
        if src == dst:
            return []
        dist, pred = self.dijkstra(src, cutoff)
        if dst not in dist:
            return None
        path = []
        n = dst
        while n != src:
            a = int(pred[n])
            path.append(a)
            n = int(self.arcs[a, 0])
        return path[::-1]

    def connected(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """True where arc a is directly followed by arc b (or a == b)."""
        a = np.asarray(a)
        b = np.asarray(b)
        return (a == b) | (self.arcs[a, 1] == self.arcs[b, 0])

    def touching(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """True where arcs a and b share at least one node (undirected adjacency)."""
        ua, va = self.arcs[a, 0], self.arcs[a, 1]
        ub, vb = self.arcs[b, 0], self.arcs[b, 1]
        return (ua == ub) | (ua == vb) | (va == ub) | (va == vb)

    def arc_lonlat(self, arc_ids) -> list[list[list[float]]]:
        """[[ [lat, lon], [lat, lon] ], ...] for map rendering."""
        arc_ids = np.asarray(arc_ids, dtype=np.int64)
        u = self.nodes_lonlat[self.arcs[arc_ids, 0]]
        v = self.nodes_lonlat[self.arcs[arc_ids, 1]]
        return [[[float(p[1]), float(p[0])], [float(q[1]), float(q[0])]] for p, q in zip(u, v)]

    def route_polyline(self, arc_ids) -> list[list[float]]:
        """Concatenate an arc sequence into one [lat, lon] polyline (breaks are kept as jumps)."""
        pts: list[list[float]] = []
        for a in arc_ids:
            u, v = self.arcs[a]
            pu, pv = self.nodes_lonlat[u], self.nodes_lonlat[v]
            if not pts or pts[-1] != [float(pu[1]), float(pu[0])]:
                pts.append([float(pu[1]), float(pu[0])])
            pts.append([float(pv[1]), float(pv[0])])
        return pts
