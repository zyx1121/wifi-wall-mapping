"""Room geometry, image-source paths and wall inversion.

Coordinates: the room is the rectangle [0, W] x [0, D] (metres), floor z = 0,
ceiling z = H. The AP hangs on the south wall (y = 0). Its uniform linear
array lies along the x axis, so it measures only u_x, the x component of the
arrival direction. The client transmits and the AP receives (uplink).

A wall line is stored relative to the AP as (theta, rho): unit normal
n = (cos theta, sin theta) that points away from the AP, and distance rho > 0,
so the line is {x : n . (x - ap) = rho}.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C = 299_792_458.0


@dataclass
class Wall:
    name: str
    p0: np.ndarray
    p1: np.ndarray
    gamma: float = -0.5

    @property
    def mid(self) -> np.ndarray:
        return (self.p0 + self.p1) / 2

    @property
    def normal(self) -> np.ndarray:
        t = self.p1 - self.p0
        t = t / np.linalg.norm(t)
        return np.array([-t[1], t[0]])

    def mirror(self, p: np.ndarray) -> np.ndarray:
        n = self.normal
        return p - 2 * np.dot(p - self.p0, n) * n


@dataclass
class Room:
    width: float = 5.0  # x extent
    depth: float = 4.0  # y extent
    height: float = 2.8
    gamma_wall: float = -0.5
    gamma_floor: float = -0.5
    gamma_ceiling: float = -0.4

    def walls(self) -> list[Wall]:
        w, d, g = self.width, self.depth, self.gamma_wall
        a = np.array
        return [
            Wall("south", a([0.0, 0.0]), a([w, 0.0]), g),  # AP mount wall
            Wall("east", a([w, 0.0]), a([w, d]), g),
            Wall("north", a([w, d]), a([0.0, d]), g),
            Wall("west", a([0.0, d]), a([0.0, 0.0]), g),
        ]


@dataclass
class Array:
    pos: np.ndarray  # 2D position of the array centre
    n_ant: int = 4
    height: float = 1.5
    fc: float = 6.425e9

    @property
    def wavelength(self) -> float:
        return C / self.fc

    @property
    def spacing(self) -> float:
        return self.wavelength / 2

    def element_x(self) -> np.ndarray:
        m = np.arange(self.n_ant)
        return (m - (self.n_ant - 1) / 2) * self.spacing


@dataclass
class Path:
    walls: tuple[str, ...]  # 2D wall sequence (empty for LoS)
    vertical: str  # "", "floor" or "ceiling"
    image: np.ndarray  # 3D image source position
    length: float
    u: np.ndarray  # 3D unit vector from AP toward the image
    gain: complex  # amplitude without the propagation phase

    @property
    def delay(self) -> float:
        return self.length / C

    @property
    def order(self) -> int:
        return len(self.walls) + (1 if self.vertical else 0)

    @property
    def is_los(self) -> bool:
        return self.order == 0


def _seg_intersect(q: np.ndarray, target: np.ndarray, wall: Wall, tol: float = 1e-9):
    """Point where segment q->target crosses the wall segment, or None."""
    d = target - q
    e = wall.p1 - wall.p0
    den = d[0] * (-e[1]) - d[1] * (-e[0])
    if abs(den) < 1e-12:
        return None
    r = wall.p0 - q
    t = (r[0] * (-e[1]) - r[1] * (-e[0])) / den
    s = (d[0] * r[1] - d[1] * r[0]) / den
    if t <= tol or t >= 1 - tol or s < -tol or s > 1 + tol:
        return None
    return q + t * d


def _bounces(ap_pos, client, seq, by_name) -> list[np.ndarray]:
    """Points of the path from the AP: AP, bounce points, client."""
    pts = [ap_pos]
    q = ap_pos
    imgs = [client]
    for n in seq:
        imgs.append(by_name[n].mirror(imgs[-1]))
    for i in range(len(seq) - 1, -1, -1):
        hit = _seg_intersect(q, imgs[i + 1], by_name[seq[i]])
        pts.append(hit)
        q = hit
    pts.append(client)
    return pts


def _cos_inc(direction: np.ndarray, wall: Wall) -> float:
    return abs(float(np.dot(direction, wall.normal))) / float(np.linalg.norm(direction))


def _path_gain(room, ap_pos, client, seq, walls, by_name, fc) -> complex:
    """Wall reflections times the transmission of every wall the legs cross.

    A room with reflect() and transmit() sets both per wall and angle. A
    plain Room has a fixed reflection and no transmission.
    """
    if not hasattr(room, "reflect"):
        return complex(np.prod([by_name[n].gamma for n in seq])) if seq else 1.0 + 0j
    pts = _bounces(ap_pos, client, seq, by_name)
    g = 1.0 + 0j
    # pts[k] -> pts[k+1]; bounce k (k >= 1) is on wall seq[len(seq) - k]
    for k in range(1, len(pts) - 1):
        w = by_name[seq[len(seq) - k]]
        g *= room.reflect(w, _cos_inc(pts[k] - pts[k - 1], w), fc)
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        skip = set()
        if k > 0:
            skip.add(seq[len(seq) - k])
        if k < len(pts) - 2:
            skip.add(seq[len(seq) - 1 - k])
        for w in walls:
            if w.name not in skip and _seg_intersect(a, b, w) is not None:
                g *= room.transmit(w, _cos_inc(b - a, w), fc)
    return g


def image_paths(
    room: Room,
    ap: Array,
    client: np.ndarray,
    client_height: float | None = None,
    max_order: int = 2,
    floor_ceiling: bool = False,
) -> list[Path]:
    """All valid specular paths up to max_order wall bounces.

    With floor_ceiling, each 2D path also gets a floor and a ceiling copy
    (one vertical bounce). The vertical bounce counts toward the order.
    """
    walls = room.walls()
    by_name = {w.name: w for w in walls}
    hc = ap.height if client_height is None else client_height
    lam = ap.wavelength

    # 2D images: (image, wall sequence, chain of intermediate images)
    images = [(client.copy(), (), [client.copy()])]
    frontier = list(images)
    for _ in range(max_order):
        nxt = []
        for img, seq, chain in frontier:
            for w in walls:
                if seq and seq[-1] == w.name:
                    continue
                new = w.mirror(img)
                item = (new, seq + (w.name,), chain + [new])
                nxt.append(item)
        images.extend(nxt)
        frontier = nxt

    valid = []
    for img, seq, chain in images:
        # backtrack from the AP through each wall in reverse order
        q = ap.pos
        ok = True
        for i in range(len(seq) - 1, -1, -1):
            hit = _seg_intersect(q, chain[i + 1], by_name[seq[i]])
            if hit is None:
                ok = False
                break
            q = hit
        if ok:
            valid.append((img, seq))

    verticals = [("", hc, 1.0)]
    if floor_ceiling:
        verticals += [("floor", -hc, room.gamma_floor), ("ceiling", 2 * room.height - hc, room.gamma_ceiling)]

    paths = []
    for img, seq in valid:
        g2d = _path_gain(room, ap.pos, client, seq, walls, by_name, ap.fc)
        if floor_ceiling and len(seq) >= max_order:
            vlist = verticals[:1]
        else:
            vlist = verticals
        for vname, zimg, gv in vlist:
            p3 = np.array([img[0], img[1], zimg])
            a3 = np.array([ap.pos[0], ap.pos[1], ap.height])
            v = p3 - a3
            length = float(np.linalg.norm(v))
            gain = g2d * gv * lam / (4 * np.pi * length)
            paths.append(Path(seq, vname, p3, length, v / length, complex(gain)))
    return paths


# ---------------------------------------------------------------- inversion


def canonical(n: np.ndarray, rho: float) -> tuple[float, float]:
    if rho < 0:
        n, rho = -n, -rho
    return float(np.arctan2(n[1], n[0])), float(rho)


def wall_line(ap_pos: np.ndarray, wall: Wall) -> tuple[float, float]:
    n = wall.normal
    return canonical(n, float(np.dot(wall.p0 - ap_pos, n)))


def client_position(ap_pos: np.ndarray, phi0: float, d0: float) -> np.ndarray:
    return ap_pos + d0 * np.array([np.cos(phi0), np.sin(phi0)])


def walls_from_paths(
    ap_pos: np.ndarray,
    phi0: float,
    d0: float,
    phis: np.ndarray,
    dls: np.ndarray,
) -> np.ndarray:
    """Candidate wall lines (theta, rho) from one link.

    phi0: LoS angle from the array axis (rad), d0: LoS length (m).
    phis, dls: angle and excess length (path length minus d0) of each
    non-LoS path. Each path gives the perpendicular bisector of the client
    and its image.
    """
    pc = client_position(ap_pos, phi0, d0)
    out = []
    for phi, dl in zip(np.atleast_1d(phis), np.atleast_1d(dls)):
        img = ap_pos + (d0 + dl) * np.array([np.cos(phi), np.sin(phi)])
        v = img - pc
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            continue
        n = v / nv
        rho = float(np.dot((pc + img) / 2 - ap_pos, n))
        out.append(canonical(n, rho))
    return np.array(out, dtype=float).reshape(-1, 2)


def oracle_params(paths: list[Path]) -> tuple[float, float, np.ndarray, np.ndarray, list[Path]]:
    """What a perfect estimator would report: phi = arccos(u_x), excess length."""
    los = [p for p in paths if p.is_los][0]
    others = [p for p in paths if not p.is_los]
    phi0 = float(np.arccos(np.clip(los.u[0], -1, 1)))
    phis = np.array([np.arccos(np.clip(p.u[0], -1, 1)) for p in others])
    dls = np.array([p.length - los.length for p in others])
    return phi0, los.length, phis, dls, others


# ---------------------------------------------------------------- scoring


@dataclass
class Match:
    found: dict[str, bool] = field(default_factory=dict)
    dist_err: dict[str, float] = field(default_factory=dict)
    ang_err: dict[str, float] = field(default_factory=dict)
    n_false: int = 0
    n_est: int = 0


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def match_walls(
    est: np.ndarray,
    ap_pos: np.ndarray,
    room: Room,
    gate_dist: float = 0.5,
    gate_ang_deg: float = 10.0,
    targets: tuple[str, ...] = ("east", "north", "west"),
) -> Match:
    """Match estimated lines to the true walls.

    Distance error: distance from the estimated line to the true wall's
    midpoint. A line inside both gates of a target wall counts as found.
    A line that matches no wall (the mount wall included) counts as false.
    """
    m = Match(n_est=len(est))
    walls = room.walls()
    gate_ang = np.deg2rad(gate_ang_deg)
    cost = np.full((len(est), len(walls)), np.inf)
    derr = np.zeros_like(cost)
    aerr = np.zeros_like(cost)
    for j, w in enumerate(walls):
        th_t, _ = wall_line(ap_pos, w)
        for i, (th, rho) in enumerate(est):
            n = np.array([np.cos(th), np.sin(th)])
            de = abs(float(np.dot(w.mid - ap_pos, n)) - rho)
            ae = abs(_wrap(th - th_t))
            derr[i, j], aerr[i, j] = de, ae
            if de <= gate_dist and ae <= gate_ang:
                cost[i, j] = de / gate_dist + ae / gate_ang
    used_est = set()
    for j, w in enumerate(walls):
        if w.name not in targets:
            continue
        m.found[w.name] = False
        col = cost[:, j]
        order = np.argsort(col)
        for i in order:
            if not np.isfinite(col[i]):
                break
            if i in used_est:
                continue
            used_est.add(int(i))
            m.found[w.name] = True
            m.dist_err[w.name] = float(derr[i, j])
            m.ang_err[w.name] = float(np.rad2deg(aerr[i, j]))
            break
    for i in range(len(est)):
        if i in used_est:
            continue
        if not np.any(np.isfinite(cost[i])):
            m.n_false += 1
    return m
