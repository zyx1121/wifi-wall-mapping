"""More than one AP: AP sites per room and a local frame per AP.

The fusion code takes the AP's linear array along x with the clients at
y > 0, because the array measures angles from 0 to 180 degrees only. An AP
on another wall works in its own frame: rotate the plan so that the AP's
wall is the south wall, run the links and the fusion there, and rotate the
found walls back.

Each room gets one AP site: the middle of its longest outer wall, 0.1 m
inside. Every phone position is then in front of every AP.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .floorplan import FloorPlan, line_id


@dataclass
class Site:
    room: str
    pos: np.ndarray  # global position
    normal: np.ndarray  # into the room


def ap_sites(plan: FloorPlan, inset: float = 0.1) -> list[Site]:
    out = []
    w, d = plan.width, plan.depth
    for name, x0, y0, x1, y1 in plan.boxes:
        edges = []  # (length, middle, inward normal)
        if y0 == 0:
            edges.append((x1 - x0, ((x0 + x1) / 2, 0.0), (0.0, 1.0)))
        if y1 == d:
            edges.append((x1 - x0, ((x0 + x1) / 2, d), (0.0, -1.0)))
        if x0 == 0:
            edges.append((y1 - y0, (0.0, (y0 + y1) / 2), (1.0, 0.0)))
        if x1 == w:
            edges.append((y1 - y0, (w, (y0 + y1) / 2), (-1.0, 0.0)))
        if not edges:
            continue
        _, mid, n = max(edges, key=lambda e: e[0])
        n = np.array(n)
        out.append(Site(name, np.array(mid) + inset * n, n))
    return out


@dataclass
class Frame:
    """local = R @ global + c, with the AP's wall along local y = 0."""

    r: np.ndarray
    c: np.ndarray

    def to_local(self, p) -> np.ndarray:
        return np.asarray(p, float) @ self.r.T + self.c

    def to_global(self, q) -> np.ndarray:
        return (np.asarray(q, float) - self.c) @ self.r


def frame(plan: FloorPlan, site: Site) -> Frame:
    nx, ny = site.normal
    r = np.array([[ny, -nx], [nx, ny]])  # rotation that takes the normal to +y
    corners = np.array([[0, 0], [plan.width, 0], [0, plan.depth], [plan.width, plan.depth]], float) @ r.T
    return Frame(r, -corners.min(axis=0))


def local_plan(plan: FloorPlan, f: Frame) -> FloorPlan:
    segs = [(n, tuple(f.to_local(a)), tuple(f.to_local(b))) for n, a, b in plan.segments]
    size = np.abs(f.r @ np.array([plan.width, plan.depth]))
    return replace(plan, segments=segs, width=float(size[0]), depth=float(size[1]), boxes=[], doors=[], waypoints=[], rooms=[])


def label_segment(plan: FloorPlan, a, b, tol_deg: float = 10.0, tol_m: float = 0.5) -> str | None:
    """Line id of the true wall line that a found segment lies on, or None."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    m = (a + b) / 2
    for w in plan.walls():
        tw = (w.p1 - w.p0) / np.linalg.norm(w.p1 - w.p0)
        if abs(float(np.dot(t, tw))) < np.cos(np.deg2rad(tol_deg)):
            continue
        nw = np.array([-tw[1], tw[0]])
        if abs(float(np.dot(m - w.p0, nw))) > tol_m:
            continue
        s = float(np.dot(m - w.p0, tw))
        if -tol_m <= s <= np.linalg.norm(w.p1 - w.p0) + tol_m:
            return line_id(w.name)
    return None
