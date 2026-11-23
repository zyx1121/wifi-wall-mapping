"""Close fused wall segments into rooms.

Input: wall segments from fusion (end points and a size) and the walk.
Steps:

1. Snap wall angles to the 2 dominant directions (Manhattan plan).
2. Join collinear segments across door-sized gaps. Find where the walk
   crosses each wall. Crossings in one tight group are a door. Crossings
   spread along the wall mean a false wall: the person walked through it.
3. Cut the plane with the full line of every wall (a cell complex). Merge
   2 neighbouring cells when the walk steps between them and no seen wall
   covers most of the line between them.
4. A merged region the walk visits is a room. A region that leaks to the
   edge of the grid (a missing outer wall) is cut to the cells near the
   walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class Segment:
    a: np.ndarray
    b: np.ndarray
    size: int = 1

    @property
    def t(self) -> np.ndarray:
        d = self.b - self.a
        return d / np.linalg.norm(d)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))


@dataclass
class RoomMap:
    x0: float
    y0: float
    res: float
    labels: np.ndarray  # 0 = not a room, k = room k
    walls: list[Segment]
    doors: list[np.ndarray] = field(default_factory=list)
    removed: list[Segment] = field(default_factory=list)

    @property
    def n_rooms(self) -> int:
        return int(self.labels.max())

    def cell(self, p) -> tuple[int, int]:
        return int((p[0] - self.x0) / self.res), int((p[1] - self.y0) / self.res)


def from_fused(fused: list[dict]) -> list[Segment]:
    return [Segment(np.asarray(f["p0"], float), np.asarray(f["p1"], float), int(f.get("size", 1))) for f in fused]


def _dominant(segs: list[Segment]) -> float:
    """Dominant wall direction modulo 90 degrees, weighted by size and length."""
    ang = np.array([np.arctan2(s.t[1], s.t[0]) for s in segs])
    w = np.array([s.size * s.length for s in segs])
    return float(np.arctan2(np.sum(w * np.sin(4 * ang)), np.sum(w * np.cos(4 * ang))) / 4)


def snap(segs: list[Segment], tol_deg: float = 15.0) -> list[Segment]:
    if not segs:
        return []
    d = _dominant(segs)
    out = []
    for s in segs:
        a = np.arctan2(s.t[1], s.t[0])
        k = np.round((a - d) / (np.pi / 2))
        a2 = d + k * np.pi / 2
        if abs(a - a2) > np.deg2rad(tol_deg):
            out.append(s)
            continue
        t = np.array([np.cos(a2), np.sin(a2)])
        m = (s.a + s.b) / 2
        h = s.length / 2
        out.append(Segment(m - h * t, m + h * t, s.size))
    return out


def bridge(segs: list[Segment], gap: float = 1.2, off: float = 0.25, ang_deg: float = 5.0) -> list[Segment]:
    """Join collinear segments whose gap is door sized."""
    segs = list(segs)
    merged = True
    while merged:
        merged = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                s, o = segs[i], segs[j]
                if abs(float(np.dot(s.t, o.t))) < np.cos(np.deg2rad(ang_deg)):
                    continue
                nrm = np.array([-s.t[1], s.t[0]])
                if abs(float(np.dot((o.a + o.b) / 2 - s.a, nrm))) > off:
                    continue
                va = sorted([0.0, s.length])
                vb = sorted([float(np.dot(o.a - s.a, s.t)), float(np.dot(o.b - s.a, s.t))])
                if vb[0] - va[1] > gap or va[0] - vb[1] > gap:
                    continue
                lo, hi = min(va[0], vb[0]), max(va[1], vb[1])
                segs[i] = Segment(s.a + lo * s.t, s.a + hi * s.t, s.size + o.size)
                del segs[j]
                merged = True
                break
            if merged:
                break
    return segs


def _cross(p, q, s: Segment, ext: float = 0.0):
    """Position along s (from s.a) where segment p->q crosses s, or None."""
    t = s.t
    a = s.a - ext * t
    e = (s.b + ext * t) - a
    d = q - p
    den = d[0] * (-e[1]) - d[1] * (-e[0])
    if abs(den) < 1e-12:
        return None
    r = a - p
    u = (r[0] * (-e[1]) - r[1] * (-e[0])) / den
    v = (d[0] * r[1] - d[1] * r[0]) / den
    if u < 0 or u > 1 or v < 0 or v > 1:
        return None
    return float(v * np.linalg.norm(e) - ext)


def doors_and_ghosts(segs: list[Segment], walk: np.ndarray, gap: float = 1.0, max_spread: float = 1.2):
    """Split walk crossings into doors, and drop walls the walk passes through along their length."""
    keep, removed, doors = [], [], []
    for s in segs:
        xs = []
        for p, q in zip(walk[:-1], walk[1:]):
            v = _cross(p, q, s)
            if v is not None:
                xs.append(v)
        xs.sort()
        groups: list[list[float]] = []
        for v in xs:
            if groups and v - groups[-1][-1] < gap:
                groups[-1].append(v)
            else:
                groups.append([v])
        if any(g[-1] - g[0] > max_spread for g in groups):
            removed.append(s)
            continue
        keep.append(s)
        for g in groups:
            doors.append(s.a + s.t * float(np.mean(g)))
    return keep, removed, doors


def _draw(mask, seg: Segment, x0, y0, res):
    n = max(2, int(np.ceil(seg.length / (res / 3))))
    for f in np.linspace(0, 1, n):
        p = seg.a + f * (seg.b - seg.a)
        i, j = int((p[0] - x0) / res), int((p[1] - y0) / res)
        if 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]:
            mask[i, j] = True


def _draw_line(mask, seg: Segment, x0, y0, res, span: float = 50.0):
    """Draw the whole line through seg, clipped to the grid."""
    m = (seg.a + seg.b) / 2
    _draw(mask, Segment(m - span * seg.t, m + span * seg.t), x0, y0, res)


def close_rooms(
    fused: list[dict] | list[Segment],
    walk: np.ndarray,
    res: float = 0.1,
    cover: float = 0.5,
    near: float = 1.5,
    min_area: float = 1.0,
    min_visits: int = 3,
) -> RoomMap:
    """Cell complex: every wall line cuts the plane, then cells merge again.

    Two cells merge when the walk steps from one to the other and no seen
    wall covers `cover` of the line between them. A seen wall is often
    shorter than the true wall, so its line, not its seen extent, decides
    the corners. Cells the walk never steps between stay apart: the person
    went around, so a wall is likely there.
    """
    segs = fused if fused and isinstance(fused[0], Segment) else from_fused(fused)  # type: ignore[arg-type]
    segs = bridge(snap(segs))
    segs, removed, doors = doors_and_ghosts(segs, walk)

    pts = np.vstack([walk] + [np.array([s.a, s.b]) for s in segs]) if segs else walk
    lo = pts.min(axis=0) - 0.5
    hi = pts.max(axis=0) + 0.5
    shape = tuple(np.ceil((hi - lo) / res).astype(int))
    cross = ndimage.generate_binary_structure(2, 1)
    line = np.zeros(shape, bool)
    seen = np.zeros(shape, bool)
    for s in segs:
        _draw_line(line, s, lo[0], lo[1], res)
        _draw(seen, s, lo[0], lo[1], res)
    line = ndimage.binary_dilation(line, structure=cross)
    seen = ndimage.binary_dilation(seen, structure=cross, iterations=2)

    cells, n = ndimage.label(~line)
    # boundary pixels between exactly 2 cells, and how much of it a seen wall covers
    pad = np.pad(cells, 2)
    stack = np.stack([pad[2 + di : 2 + di + shape[0], 2 + dj : 2 + dj + shape[1]] for di in range(-2, 3) for dj in range(-2, 3)])
    tot: dict[tuple[int, int], int] = {}
    hit: dict[tuple[int, int], int] = {}
    for i, j in zip(*np.nonzero(line)):
        ids = np.unique(stack[:, i, j])
        ids = ids[ids > 0]
        if len(ids) != 2:
            continue
        key = (int(ids[0]), int(ids[1]))
        tot[key] = tot.get(key, 0) + 1
        hit[key] = hit.get(key, 0) + int(seen[i, j])
    # cell pairs the walk steps across directly
    _, (ni, nj) = ndimage.distance_transform_edt(line, return_indices=True)
    crossed = set()
    for p, q in zip(walk[:-1], walk[1:]):
        ip, jp = int((p[0] - lo[0]) / res), int((p[1] - lo[1]) / res)
        iq, jq = int((q[0] - lo[0]) / res), int((q[1] - lo[1]) / res)
        a, b = int(cells[ni[ip, jp], nj[ip, jp]]), int(cells[ni[iq, jq], nj[iq, jq]])
        if a != b:
            crossed.add((min(a, b), max(a, b)))
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key, t in tot.items():
        if hit[key] / t < cover and key in crossed:
            parent[find(key[0])] = find(key[1])
    group = np.array([find(c) for c in range(n + 1)])
    comp = np.where(cells > 0, group[cells], 0)

    visit = np.zeros(shape, bool)
    for p in walk:
        visit[int((p[0] - lo[0]) / res), int((p[1] - lo[1]) / res)] = True
    near_walk = ndimage.distance_transform_edt(~visit) * res <= near
    edge = np.zeros(shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True

    labels = np.zeros(shape, int)
    k = 0
    for c in np.unique(comp[comp > 0]):
        mask = comp == c
        if np.count_nonzero(mask & visit) < min_visits:
            continue
        if np.any(mask & edge):
            mask = mask & near_walk
        if np.count_nonzero(mask) * res * res < min_area:
            continue
        k += 1
        labels[mask] = k
    # give the line cells to the rooms beside them, so a room reaches its wall line
    for _ in range(2):
        grown = ndimage.grey_dilation(labels, footprint=cross)
        fill = (labels == 0) & line & (grown > 0)
        labels[fill] = grown[fill]
    return RoomMap(float(lo[0]), float(lo[1]), res, labels, segs, doors, removed)


# ---------------------------------------------------------------- scoring


def _box_mask(rm: RoomMap, box) -> np.ndarray:
    _, x0, y0, x1, y1 = box
    xs = rm.x0 + (np.arange(rm.labels.shape[0]) + 0.5) * rm.res
    ys = rm.y0 + (np.arange(rm.labels.shape[1]) + 0.5) * rm.res
    return ((xs[:, None] > x0) & (xs[:, None] < x1)) & ((ys[None, :] > y0) & (ys[None, :] < y1))


def score_rooms(rm: RoomMap, boxes, doors, iou_ok: float = 0.5, door_dist: float = 0.6) -> dict:
    """Match true rooms to found rooms one to one by IoU.

    Returns the mean IoU over true rooms (0 for an unmatched room), the
    rooms found with IoU >= iou_ok, the number of found rooms, and door
    recall and precision within door_dist metres.
    """
    ious = np.zeros((len(boxes), rm.n_rooms))
    for i, b in enumerate(boxes):
        t = _box_mask(rm, b)
        for k in range(rm.n_rooms):
            f = rm.labels == k + 1
            u = np.count_nonzero(t | f)
            ious[i, k] = np.count_nonzero(t & f) / u if u else 0.0
    best = np.zeros(len(boxes))
    used: set[int] = set()
    for flat in np.argsort(-ious, axis=None):
        i, k = np.unravel_index(flat, ious.shape)
        if ious[i, k] <= 0 or best[i] > 0 or k in used:
            continue
        best[i] = ious[i, k]
        used.add(int(k))
    hit_d = sum(1 for d in doors if any(np.linalg.norm(np.asarray(d) - f) < door_dist for f in rm.doors))
    true_f = sum(1 for f in rm.doors if any(np.linalg.norm(np.asarray(d) - f) < door_dist for d in doors))
    return {
        "mean_iou": float(best.mean()),
        "rooms_ok": int(np.sum(best >= iou_ok)),
        "rooms_true": len(boxes),
        "rooms_found": rm.n_rooms,
        "door_recall": hit_d / len(doors) if doors else 0.0,
        "door_precision": true_f / len(rm.doors) if rm.doors else 0.0,
    }
