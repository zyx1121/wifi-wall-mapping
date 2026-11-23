"""A furnished-free apartment floor plan and a walk through it.

The plan has the same interface as geometry.Room (walls(), height and the
floor and ceiling reflection coefficients), so image_paths works on it. Walls
are segments, so door openings are gaps. A path that crosses an interior wall
loses amplitude by `transmission` per crossing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Wall


@dataclass
class FloorPlan:
    segments: list[tuple[str, tuple[float, float], tuple[float, float]]]
    width: float
    depth: float
    height: float = 2.8
    gamma_wall: float = -0.5
    gamma_floor: float = -0.5
    gamma_ceiling: float = -0.4
    transmission: float = 0.4  # amplitude factor per interior wall crossing (-8 dB)
    rooms: list[tuple[str, tuple[float, float]]] = field(default_factory=list)
    # true rooms as boxes (name, x0, y0, x1, y1) and door centres, for scoring
    boxes: list[tuple[str, float, float, float, float]] = field(default_factory=list)
    doors: list[tuple[float, float]] = field(default_factory=list)
    waypoints: list[tuple[float, float]] = field(default_factory=list)

    def walls(self) -> list[Wall]:
        return [Wall(n, np.array(a, float), np.array(b, float), self.gamma_wall) for n, a, b in self.segments]


def apartment() -> FloorPlan:
    """10 x 8 m: living room, open kitchen, 2 bedrooms, bathroom.

    The AP hangs on the south wall of the living room. Wall names group
    collinear segments as one line: "y5a" and "y5b" are both on y = 5.
    """
    s = [
        ("south", (0, 0), (10, 0)),
        ("east", (10, 0), (10, 8)),
        ("north", (10, 8), (0, 8)),
        ("west", (0, 8), (0, 0)),
        # living room | bedroom 1 and bathroom, doors at x 3.2-4.1 and 4.9-5.7
        ("y5a", (0, 5), (3.2, 5)),
        ("y5b", (4.1, 5), (4.9, 5)),
        ("y5c", (5.7, 5), (6, 5)),
        # bedroom 1 | bathroom
        ("x4.5", (4.5, 5), (4.5, 8)),
        # kitchen and bedroom 2 | living room, open kitchen below y 2.3, door at y 3.8-4.7
        ("x6a", (6, 2.3), (6, 3.8)),
        ("x6b", (6, 4.7), (6, 8)),
        # kitchen | bedroom 2
        ("y3.5", (6, 3.5), (10, 3.5)),
    ]
    rooms = [("客廳", (3.0, 2.5)), ("廚房", (8.0, 1.7)), ("臥室", (8.0, 5.8)), ("臥室", (2.2, 6.5)), ("浴室", (5.25, 6.5))]
    boxes = [("客廳", 0, 0, 6, 5), ("廚房", 6, 0, 10, 3.5), ("臥室", 6, 3.5, 10, 8), ("臥室", 0, 5, 4.5, 8), ("浴室", 4.5, 5, 6, 8)]
    doors = [(3.65, 5), (5.3, 5), (6, 4.25)]
    return FloorPlan(s, 10.0, 8.0, rooms=rooms, boxes=boxes, doors=doors, waypoints=WAYPOINTS)


def apartment_b() -> FloorPlan:
    """9 x 7.5 m held-out plan: living room, closed kitchen, 2 bedrooms, bathroom.

    No parameter was tuned on this plan. Report results on it.
    """
    s = [
        ("south", (0, 0), (9, 0)),
        ("east", (9, 0), (9, 7.5)),
        ("north", (9, 7.5), (0, 7.5)),
        ("west", (0, 7.5), (0, 0)),
        # living room | bedroom A and bathroom, doors at x 1.2-2.0 and 4.0-4.8
        ("y4.5a", (0, 4.5), (1.2, 4.5)),
        ("y4.5b", (2.0, 4.5), (4.0, 4.5)),
        ("y4.5c", (4.8, 4.5), (5.5, 4.5)),
        # bedroom A | bathroom
        ("x3.2", (3.2, 4.5), (3.2, 7.5)),
        # living room | kitchen and bedroom B, doors at y 1.0-1.8 and 3.4-4.2
        ("x5.5a", (5.5, 0), (5.5, 1.0)),
        ("x5.5b", (5.5, 1.8), (5.5, 3.4)),
        ("x5.5c", (5.5, 4.2), (5.5, 7.5)),
        # kitchen | bedroom B
        ("y2.8", (5.5, 2.8), (9, 2.8)),
    ]
    rooms = [("客廳", (2.7, 2.2)), ("廚房", (7.3, 1.4)), ("臥室", (7.3, 5.2)), ("臥室", (1.6, 6.0)), ("浴室", (4.35, 6.0))]
    boxes = [("客廳", 0, 0, 5.5, 4.5), ("廚房", 5.5, 0, 9, 2.8), ("臥室", 5.5, 2.8, 9, 7.5), ("臥室", 0, 4.5, 3.2, 7.5), ("浴室", 3.2, 4.5, 5.5, 7.5)]
    doors = [(1.6, 4.5), (4.4, 4.5), (5.5, 1.4), (5.5, 3.8)]
    way = [
        (1.0, 1.0), (4.8, 1.0), (4.8, 1.4), (6.2, 1.4), (8.3, 1.2), (8.3, 2.2), (6.2, 2.2), (6.2, 1.4), (4.8, 1.4),
        (4.8, 3.8), (6.2, 3.8), (8.3, 3.5), (8.3, 6.8), (6.2, 6.8), (6.2, 3.8), (4.8, 3.8),
        (4.4, 3.9), (4.4, 5.2), (4.4, 6.8), (3.8, 6.8), (4.4, 5.2), (4.4, 3.9),
        (1.6, 3.9), (1.6, 5.2), (0.8, 6.8), (2.6, 6.8), (1.6, 5.2), (1.6, 3.9), (0.8, 2.5), (3.0, 2.5),
    ]
    return FloorPlan(s, 9.0, 7.5, rooms=rooms, boxes=boxes, doors=doors, waypoints=way)


# line id of each segment: collinear segments share one line
def line_id(name: str) -> str:
    return name.rstrip("abc") if name[0] in "xy" else name


WAYPOINTS = [
    (1.0, 1.2), (5.0, 1.0), (7.2, 1.0), (9.2, 1.0), (9.2, 2.8), (7.0, 2.8), (5.3, 2.0),
    (5.3, 4.25), (6.8, 4.25), (9.2, 4.3), (9.2, 7.3), (7.0, 7.3), (6.8, 4.25), (5.3, 4.25),
    (5.3, 5.6), (5.3, 7.3), (5.3, 4.4),
    (3.65, 4.4), (3.65, 5.6), (1.0, 6.0), (1.0, 7.4), (3.8, 7.4), (3.65, 5.6), (3.65, 4.4),
    (1.0, 3.5), (1.0, 1.8),
]


def walk(step: float = 0.3, jitter: float = 0.05, seed: int = 0, waypoints=None) -> np.ndarray:
    """Client positions every `step` metres along the waypoints."""
    rng = np.random.default_rng(seed)
    w = np.array(WAYPOINTS if waypoints is None else waypoints, float)
    pts = []
    carry = 0.0
    for a, b in zip(w[:-1], w[1:]):
        seg = np.linalg.norm(b - a)
        t = carry
        while t < seg:
            pts.append(a + (b - a) * t / seg)
            t += step
        carry = t - seg
    pts.append(w[-1])
    p = np.array(pts)
    return p + jitter * rng.standard_normal(p.shape)
