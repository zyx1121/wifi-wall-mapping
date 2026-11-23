"""Real apartments from the Swiss Dwellings dataset (CC BY 4.0).

Standfest, M. et al., Swiss Dwellings, Zenodo, doi:10.5281/zenodo.7788422.

geometries.csv holds one row per wall, opening and area of each apartment,
as polygons in metres. This module turns one apartment into a FloorPlan:

- Each wall polygon becomes 1 segment on its centre line, with its thickness.
- Each window becomes a glass segment. The entrance door becomes a wood
  door. Inner doors stay open (gaps).
- Rooms are polygons, for scoring and for the walk.
- The plan turns so that the walls run along x and y.

The dataset has no wall materials. 2 rules assign them (STYLES):
- "swiss": outer walls are concrete of their own thickness, inner walls
  of 15 cm or more are concrete, thinner ones are drywall.
- "taiwan": outer walls are 15 cm concrete, inner walls are 12 cm brick.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MPath
from scipy import ndimage

from .floorplan import FloorPlan

INDOOR = {
    "ROOM": "臥室", "BEDROOM": "臥室", "LIVING_ROOM": "客廳", "LIVING_DINING": "客廳", "DINING": "餐廳",
    "KITCHEN": "廚房", "KITCHEN_DINING": "廚房", "BATHROOM": "浴室", "CORRIDOR": "走廊",
    "CORRIDORS_AND_HALLS": "走廊", "STOREROOM": "儲藏室", "OFFICE": "書房", "WINTERGARTEN": "陽光室",
}
GLASS = (("glass", 0.004), ("air", 0.012), ("glass", 0.004))  # double glazing
DOOR = (("wood", 0.045),)
STYLES = ["swiss", "taiwan"]

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")


def polygon(wkt: str) -> np.ndarray | None:
    """Outer ring of a WKT POLYGON, or of the largest part of a MULTIPOLYGON."""
    if wkt.startswith("MULTIPOLYGON"):
        parts = [p for p in wkt.split(")), ((")]
        rings = [polygon("POLYGON ((" + p.strip("MULTIPOLYGON ()") + "))") for p in parts]
        rings = [r for r in rings if r is not None]
        return max(rings, key=lambda r: abs(_area(r))) if rings else None
    if not wkt.startswith("POLYGON"):
        return None
    outer = wkt.split("), (")[0]
    v = np.array([float(x) for x in _NUM.findall(outer)])
    return v.reshape(-1, 2) if len(v) >= 6 else None


def _area(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def centre_line(r: np.ndarray):
    """Centre segment and thickness of a thin polygon, and how box-like it is (0 to 1)."""
    c = r.mean(axis=0)
    _, _, vt = np.linalg.svd(r - c)
    t, n = vt[0], vt[1]
    s, h = (r - c) @ t, (r - c) @ n
    length, thick = s.max() - s.min(), h.max() - h.min()
    mid = c + n * (h.max() + h.min()) / 2
    a, b = mid + t * s.min(), mid + t * s.max()
    fill = abs(_area(r)) / max(length * thick, 1e-9)
    return a, b, float(thick), float(fill)


# ---------------------------------------------------------------- reading


def _rows(path):
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as f:
        yield from csv.DictReader(f)


def candidates(path, min_rooms: int = 4, max_rooms: int = 8) -> list[str]:
    """Residential apartments on 1 floor with a living room, a kitchen, a bathroom and 1 or more bedrooms."""
    info: dict[str, dict] = {}
    for r in _rows(path):
        a = info.setdefault(r["apartment_id"], {"floors": set(), "usage": set(), "types": {}})
        a["floors"].add(r["floor_id"])
        a["usage"].add(r["unit_usage"])
        if r["entity_type"] == "area":
            a["types"][r["entity_subtype"]] = a["types"].get(r["entity_subtype"], 0) + 1
    out = []
    for aid, a in info.items():
        t = a["types"]
        rooms = sum(v for k, v in t.items() if k in INDOOR and INDOOR[k] != "走廊")
        living = t.get("LIVING_ROOM", 0) + t.get("LIVING_DINING", 0)
        beds = t.get("ROOM", 0) + t.get("BEDROOM", 0)
        if len(a["floors"]) == 1 and a["usage"] == {"RESIDENTIAL"} and living == 1 and beds >= 1 and t.get("KITCHEN", 0) and t.get("BATHROOM", 0) and min_rooms <= rooms <= max_rooms:
            out.append(aid)
    return sorted(out)


def read(path, ids) -> dict[str, list[dict]]:
    ids = set(ids)
    out: dict[str, list[dict]] = {}
    for r in _rows(path):
        if r["apartment_id"] in ids:
            out.setdefault(r["apartment_id"], []).append({k: r[k] for k in ("entity_type", "entity_subtype", "geometry")})
    return out


# ---------------------------------------------------------------- plan


@dataclass
class Apartment:
    plan: FloorPlan
    outer: set[str]  # names of outer segments
    thick: dict[str, float]  # thickness per segment name
    kind: dict[str, str]  # wall, window or door per segment name
    rooms: list  # (name, polygon) per dataset area; plan.boxes merges open areas for scoring


def _cut_openings(raw, min_piece: float = 0.1):
    """Walls run through their doors and windows in the dataset: cut the openings out.

    Inner doors ("gap") leave a gap. Windows and the entrance door keep their own segment.
    """
    walls = [r for r in raw if r[3] == "wall"]
    opens = [r for r in raw if r[3] != "wall"]
    for oa, ob, ot, _ in opens:
        od = ob - oa
        ol = float(np.linalg.norm(od))
        if ol < 1e-6:
            continue
        ou = od / ol
        out = []
        for wa, wb, wt, k in walls:
            wd = wb - wa
            wl = float(np.linalg.norm(wd))
            wu = wd / wl
            n = np.array([-wu[1], wu[0]])
            if abs(float(np.dot(ou, wu))) < np.cos(np.deg2rad(5)) or abs(float(np.dot((oa + ob) / 2 - wa, n))) > wt / 2 + 0.05:
                out.append((wa, wb, wt, k))
                continue
            s0, s1 = sorted([float(np.dot(oa - wa, wu)), float(np.dot(ob - wa, wu))])
            if s1 <= 0 or s0 >= wl:
                out.append((wa, wb, wt, k))
                continue
            if s0 > min_piece:
                out.append((wa, wa + wu * s0, wt, k))
            if wl - s1 > min_piece:
                out.append((wa + wu * s1, wb, wt, k))
        walls = out
    return walls + [r for r in opens if r[3] != "gap"]


def _dominant(segs) -> float:
    ang = np.array([np.arctan2(*(b - a)[::-1]) for a, b, *_ in segs])
    w = np.array([np.linalg.norm(b - a) for a, b, *_ in segs])
    return float(np.arctan2(np.sum(w * np.sin(4 * ang)), np.sum(w * np.cos(4 * ang))) / 4)


def materials(apt: Apartment, style: str) -> dict:
    out = {}
    for name, kind in apt.kind.items():
        t = apt.thick[name]
        if kind == "window":
            out[name] = GLASS
        elif kind == "door":
            out[name] = DOOR
        elif style == "taiwan":
            out[name] = (("concrete", 0.15),) if name in apt.outer else (("brick", 0.12),)
        elif name in apt.outer or t >= 0.15:
            out[name] = (("concrete", round(t, 3)),)
        else:
            d = max(t - 0.025, 0.01)  # 2 boards of 12.5 mm and air between
            out[name] = (("plasterboard", 0.0125), ("air", round(d, 3)), ("plasterboard", 0.0125))
    return out


def build(rows: list[dict], min_len: float = 0.15) -> Apartment:
    raw = []  # (a, b, thickness, kind)
    rooms, doors = [], []
    for r in rows:
        g = polygon(r["geometry"])
        if g is None:
            continue
        et, st = r["entity_type"], r["entity_subtype"]
        if et == "separator" and st == "WALL":
            a, b, t, fill = centre_line(g)
            if np.linalg.norm(b - a) >= min_len and fill > 0.6 and t > 0.02:
                raw.append((a, b, t, "wall"))
        elif et == "opening" and st == "WINDOW":
            a, b, t, _ = centre_line(g)
            raw.append((a, b, t, "window"))
        elif et == "opening" and st == "ENTRANCE_DOOR":
            a, b, t, _ = centre_line(g)
            raw.append((a, b, t, "door"))
        elif et == "opening" and st == "DOOR":
            doors.append(g.mean(axis=0))
            a, b, t, _ = centre_line(g)
            raw.append((a, b, t, "gap"))
        elif et == "area" and st in INDOOR:
            rooms.append((INDOOR[st], g))
    raw = _cut_openings(raw)
    # turn the plan so the walls run along x and y, origin at the lower left corner
    th = _dominant([s for s in raw if s[3] == "wall"])
    rot = np.array([[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]])
    pts = np.vstack([np.vstack([a, b]) for a, b, *_ in raw] + [g for _, g in rooms])
    o = (pts @ rot.T).min(axis=0)

    def tf(p):
        return np.asarray(p) @ rot.T - o

    segs = [(tf(a), tf(b), t, k) for a, b, t, k in raw]
    rooms = [(n, tf(g)) for n, g in rooms]
    doors = [tf(d) for d in doors]
    # group collinear segments into lines (same direction, offset within 0.1 m)
    lines: list[tuple[float, float]] = []
    named, thick, kind = [], {}, {}
    count: dict[int, int] = {}
    for a, b, t, k in segs:
        d = b - a
        ang = np.arctan2(d[1], d[0]) % np.pi
        nrm = np.array([-np.sin(ang), np.cos(ang)])
        off = float(np.dot(a, nrm))
        li = next((i for i, (la, lo) in enumerate(lines) if abs((ang - la + np.pi / 2) % np.pi - np.pi / 2) < np.deg2rad(5) and abs(off - lo) < 0.1), None)
        if li is None:
            lines.append((ang, off))
            li = len(lines) - 1
        name = f"L{li}_{count.get(li, 0)}"
        count[li] = count.get(li, 0) + 1
        named.append((name, (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
        thick[name], kind[name] = t, k
    # outer: one side of the segment is outside every room
    paths = [MPath(g) for _, g in rooms]

    def indoor(p):
        return any(pp.contains_point(p) for pp in paths)

    outer = set()
    for name, a, b in named:
        a, b = np.array(a), np.array(b)
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([-t[1], t[0]])
        off = thick[name] / 2 + 0.15
        sides = [[indoor(a + (b - a) * f + s * off * n) for f in (0.25, 0.5, 0.75)] for s in (1, -1)]
        if kind[name] != "wall" or not (any(sides[0]) and any(sides[1])):
            outer.add(name)
    w, d = (pts @ rot.T - o).max(axis=0)
    labels = [(n, tuple(_label_point(g))) for n, g in rooms]
    spaces = _spaces(rooms)
    plan = FloorPlan(named, float(w), float(d), rooms=labels, boxes=spaces, doors=[tuple(p) for p in doors])
    return Apartment(plan, outer, thick, kind, rooms)


def _spaces(rooms, res: float = 0.05, touch: float = 0.06):
    """True rooms for scoring: dataset areas that touch without a wall between them form 1 space.

    The dataset splits an open living room, kitchen and corridor into
    separate areas. No wall separates them, so no method can tell them apart.
    Areas behind a wall or a door do not touch: the wall lies between them.
    """
    if not rooms:
        return []
    allp = np.vstack([g for _, g in rooms])
    lo, hi = allp.min(axis=0) - 0.2, allp.max(axis=0) + 0.2
    xs, ys = np.arange(lo[0], hi[0], res), np.arange(lo[1], hi[1], res)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    q = np.c_[gx.ravel(), gy.ravel()]
    masks = [MPath(g).contains_points(q).reshape(gx.shape) for _, g in rooms]
    k = int(np.ceil(touch / res))
    grown = [ndimage.binary_dilation(m, iterations=k) for m in masks]
    parent = list(range(len(rooms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            # touching along 0.5 m or more, not just at a corner
            if np.count_nonzero(grown[i] & masks[j]) * res > 0.5 * (k + 1):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(rooms)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        main = max(idx, key=lambda i: abs(_area(rooms[i][1])))
        out.append((rooms[main][0], [rooms[i][1] for i in idx]))
    return out


def _label_point(g: np.ndarray, res: float = 0.1) -> np.ndarray:
    """Point deep inside a room polygon."""
    lo, hi = g.min(axis=0), g.max(axis=0)
    xs, ys = np.arange(lo[0], hi[0], res), np.arange(lo[1], hi[1], res)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    m = MPath(g).contains_points(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    if not m.any():
        return g.mean(axis=0)
    i, j = np.unravel_index(np.argmax(ndimage.distance_transform_edt(m)), m.shape)
    return np.array([xs[i], ys[j]])


# ---------------------------------------------------------------- walk


def _grid(apt: Apartment, res: float, clear: float):
    p = apt.plan
    xs = np.arange(-0.5, p.width + 0.5, res) + res / 2
    ys = np.arange(-0.5, p.depth + 0.5, res) + res / 2
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.c_[gx.ravel(), gy.ravel()]
    room_id = np.zeros(len(pts), int)
    for k, (_, g) in enumerate(apt.rooms):
        room_id[MPath(g).contains_points(pts) & (room_id == 0)] = k + 1
    wall = np.zeros(len(pts), bool)
    for name, a, b in p.segments:
        a, b = np.array(a), np.array(b)
        t = (b - a) / np.linalg.norm(b - a)
        s = (pts - a) @ t
        h = np.abs((pts - a) @ np.array([-t[1], t[0]]))
        wall |= (h <= apt.thick[name] / 2) & (s >= 0) & (s <= np.linalg.norm(b - a))
    shape = gx.shape
    wall = wall.reshape(shape)
    room_id = room_id.reshape(shape)
    dist = ndimage.distance_transform_edt(~wall) * res
    # door cells join the rooms on both sides
    near_door = np.zeros(shape, bool)
    for dpt in p.doors:
        near_door |= (np.hypot(gx - dpt[0], gy - dpt[1]) < 0.45)
    free = ((room_id > 0) | near_door) & (dist >= clear)
    return xs, ys, free, room_id, dist


def _bfs(free, start):
    d = np.full(free.shape, -1, int)
    prev = np.full(free.shape + (2,), -1, int)
    q = [start]
    d[start] = 0
    head = 0
    while head < len(q):
        i, j = q[head]
        head += 1
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = i + di, j + dj
            if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and free[a, b] and d[a, b] < 0:
                d[a, b] = d[i, j] + 1
                prev[a, b] = (i, j)
                q.append((a, b))
    return d, prev


def waypoints(apt: Apartment, res: float = 0.1, clear: float = 0.3, spacing: float = 1.0, per_m2: float = 0.5) -> list[tuple[float, float]]:
    """A route that covers every reachable room: about 1 stop per 2 m2, 1 m apart, joined by shortest paths."""
    xs, ys, free, room_id, dist = _grid(apt, res, clear)
    lab, n = ndimage.label(free, structure=np.ones((3, 3)))
    if n == 0:
        return []
    main = 1 + int(np.argmax(ndimage.sum(free, lab, range(1, n + 1))))
    free = lab == main
    stops = []
    for k in range(1, len(apt.rooms) + 1):
        cells = np.argwhere(free & (room_id == k))
        if len(cells) < 10:
            continue
        area = len(cells) * res * res
        want = int(np.clip(area * per_m2, 2, 14))
        dd = dist[cells[:, 0], cells[:, 1]]
        chosen = [cells[int(np.argmax(dd))]]
        while len(chosen) < want:
            gap = np.min([np.hypot(*(cells - c).T) for c in chosen], axis=0) * res
            i = int(np.argmax(gap))
            if gap[i] < spacing:
                break
            chosen.append(cells[i])
        stops += [tuple(c) for c in chosen]
    # nearest-neighbour tour from the stop deepest in the largest room
    order = [stops.pop(0)]
    paths = []
    while stops:
        d, prev = _bfs(free, order[-1])
        j = int(np.argmin([d[s] if d[s] >= 0 else 10**9 for s in stops]))
        if d[stops[j]] < 0:
            stops.pop(j)
            continue
        nxt = stops.pop(j)
        path = [nxt]
        while path[-1] != order[-1]:
            path.append(tuple(prev[path[-1]]))
        paths.append(path[::-1])
        order.append(nxt)
    cells = [order[0]] + [c for p in paths for c in p[1:]]
    # keep the corners: drop cells that a straight free line can skip
    keep = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = i + 1
        while j + 1 < len(cells) and _clear_line(free, cells[i], cells[j + 1]) and _close(cells[i : j + 2], 3):
            j += 1
        keep.append(cells[j])
        i = j
    return [(float(xs[i]), float(ys[j])) for i, j in keep]


def _close(cells, tol: float) -> bool:
    """The route between the 2 ends stays within tol cells of the straight line."""
    c = np.array(cells, float)
    a, b = c[0], c[-1]
    d = b - a
    L = float(np.hypot(*d))
    if L == 0:
        return bool(np.max(np.hypot(*(c - a).T)) <= tol)
    s = np.clip((c - a) @ d / L**2, 0, 1)
    return bool(np.max(np.hypot(*(c - a - s[:, None] * d).T)) <= tol)


def _clear_line(free, a, b) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    ii = np.round(np.linspace(a[0], b[0], n + 1)).astype(int)
    jj = np.round(np.linspace(a[1], b[1], n + 1)).astype(int)
    return bool(free[ii, jj].all())


# ---------------------------------------------------------------- AP sites


def ap_sites(apt: Apartment, walk_pos: np.ndarray, inset: float = 0.1, front: float = 0.1, share: float = 0.8):
    """1 AP per room on the middle of its longest outer wall.

    The array cannot tell front from back. A site needs `share` of the walk
    at least `front` metres in front of it; the fusion drops the other
    positions (in_front), since the phone knows where it is.
    """
    from .multi_ap import Site

    p = apt.plan
    out = []
    for name, g in apt.rooms:
        paths = [MPath(g)]
        best = None
        for seg, a, b in p.segments:
            if seg not in apt.outer or apt.kind[seg] != "wall":
                continue
            a, b = np.array(a), np.array(b)
            length = float(np.linalg.norm(b - a))
            if length < 1.0:
                continue
            t = (b - a) / length
            n = np.array([-t[1], t[0]])
            mid = (a + b) / 2
            for s in (1, -1):
                pos = mid + s * n * (apt.thick[seg] / 2 + inset)
                if not any(pp.contains_point(pos + s * n * 0.2) for pp in paths):
                    continue
                if np.mean((walk_pos - pos) @ (s * n) >= front) < share:
                    continue
                if best is None or length > best[0]:
                    best = (length, pos, s * n)
        if best is not None:
            out.append(Site(name, best[1], best[2]))
    return out


def in_front(site, walk_pos: np.ndarray, front: float = 0.1) -> np.ndarray:
    """Walk positions in front of the AP of a site: the links the fusion may use."""
    return (np.asarray(walk_pos) - site.pos) @ site.normal >= front
