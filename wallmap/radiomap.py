"""Received power over a grid of phone positions.

Power is the incoherent sum of the path powers: the local average over
fading. A coherent sum on a 0.25 m grid would alias the 2 cm fading
pattern at 6 GHz.
"""

from __future__ import annotations

import numpy as np

from .floorplan import FloorPlan
from .geometry import Array, image_paths
from .rooms import Segment

TX_DBM = 15.0  # phone transmit power


def rss_dbm(room, ap: Array, points: np.ndarray, max_order: int = 2, tx_dbm: float = TX_DBM) -> np.ndarray:
    out = np.empty(len(points))
    for i, p in enumerate(points):
        paths = image_paths(room, ap, np.asarray(p, float), max_order=max_order)
        out[i] = tx_dbm + 10 * np.log10(sum(abs(q.gain) ** 2 for q in paths))
    return out


def grid(width: float, depth: float, step: float = 0.25, margin: float = 0.15) -> np.ndarray:
    xs = np.arange(margin, width - margin + 1e-9, step)
    ys = np.arange(margin, depth - margin + 1e-9, step)
    return np.array([(x, y) for y in ys for x in xs])


def plan_from_segments(segs: list[Segment], width: float, depth: float, loss_db: float = 5.0, gamma: float = -0.35, losses=None) -> FloorPlan:
    """A floor plan from inferred wall segments with one loss per wall (TGax uses 5 dB)."""
    s = [(f"w{i}", tuple(g.a), tuple(g.b)) for i, g in enumerate(segs)]
    return FloorPlan(s, width, depth, gamma_wall=gamma, transmission=10 ** (-loss_db / 20), losses=losses)


def free_space(width: float, depth: float) -> FloorPlan:
    return FloorPlan([], width, depth)
