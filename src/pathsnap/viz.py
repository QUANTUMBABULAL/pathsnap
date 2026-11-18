"""Folium map generation for trips (GPS, nearby network, reference / predicted routes)."""
from __future__ import annotations

import folium
import numpy as np
from scipy.spatial import cKDTree

from .features import TripContext


def nearby_arcs(ctx: TripContext, buffer: float = 120.0, max_arcs: int = 6000) -> np.ndarray:
    """Arc ids within ``buffer`` metres of the trajectory (one arc per two-way street)."""
    net = ctx.net
    tree = cKDTree(ctx.xy)
    mid = (net.A + net.B) / 2
    d = np.minimum.reduce([tree.query(net.A)[0], tree.query(net.B)[0], tree.query(mid)[0]])
    ids = np.flatnonzero(d <= buffer)
    rev = net.reverse[ids]
    keep = (rev < 0) | (ids < rev) | ~np.isin(rev, ids)
    ids = ids[keep]
    if len(ids) > max_arcs:
        ids = ids[np.argsort(d[ids])[:max_arcs]]
    return ids


def _legend(items):
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        f'<span style="display:inline-block;width:22px;height:{4 if kind == "line" else 10}px;'
        f'border-radius:{0 if kind == "line" else 5}px;background:{color}"></span>{name}</div>'
        for name, color, kind in items
    )
    return folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:10px 14px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.25);'
        f'font:13px/1.3 system-ui,sans-serif">{rows}</div>'
    )


def trip_map(ctx: TripContext, routes=(), points=(), show_network: bool = True,
             gps_max_points: int = 3000, title: str | None = None) -> folium.Map:
    """Interactive map.

    routes : iterable of (name, arc_ids, colour, weight)
    points : iterable of (name, lat array, lon array, colour) extra marker layers
    """
    lon, lat = ctx.track[:, 0], ctx.track[:, 1]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]], padding=(20, 20))
    legend = []

    if show_network:
        ids = nearby_arcs(ctx)
        fg = folium.FeatureGroup(name="Road network (nearby)")
        folium.PolyLine(ctx.net.arc_lonlat(ids), color="#8a8f98", weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", "#8a8f98", "line"))

    for name, arcs, colour, weight in routes:
        fg = folium.FeatureGroup(name=name)
        folium.PolyLine(ctx.net.route_polyline(arcs), color=colour, weight=weight, opacity=0.85,
                        tooltip=name).add_to(fg)
        fg.add_to(m)
        legend.append((name, colour, "line"))

    step = max(1, int(np.ceil(len(lat) / gps_max_points)))
    fg = folium.FeatureGroup(name="Raw GPS observations")
    for i in range(0, len(lat), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2.5, color="#1f6feb", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"GPS #{i}  t={ctx.t[i]:.0f}s").add_to(fg)
    fg.add_to(m)
    legend.append(("Raw GPS", "#1f6feb", "dot"))

    for name, plat, plon, colour in points:
        fg = folium.FeatureGroup(name=name)
        for a, b in zip(plat, plon):
            folium.CircleMarker([a, b], radius=4, color=colour, fill=True, fill_opacity=0.9, weight=1).add_to(fg)
        fg.add_to(m)
        legend.append((name, colour, "dot"))

    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:12px;left:60px;z-index:9999;background:white;padding:6px 12px;'
            f'border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.2);font:600 14px system-ui">{title}</div>'))
    m.get_root().html.add_child(_legend(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m
