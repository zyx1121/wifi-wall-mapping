"""Direct-path attenuation as wall evidence.

The estimator resolves the direct path, so its amplitude is free of
multipath fading. Its excess loss over free space is the sum of the losses
of the walls between the AP and the phone, plus one unknown offset per
walk (phone transmit power, antenna gains):

    excess_k = c + sum_w x_w [AP -> phone k crosses wall w]

A robust least-squares fit gives c and the loss x_w of every found wall.
Every direct link starts at the one AP, so the loss cannot place a wall
along the ray by itself. It tests the walls that reflections found:
a wall that many links cross without loss is a false wall.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .estimators import Paths
from .rooms import Segment, _cross


def los_amplitude(p: Paths, los_power_frac: float = 0.25) -> float:
    """Amplitude of the path that relative_params takes as the direct path."""
    if len(p) == 0:
        return float("nan")
    pw = np.abs(p.amp) ** 2
    ref = p.tau[np.argmax(pw)]
    dt = (p.tau - ref + p.period / 2) % p.period - p.period / 2
    strong = pw >= los_power_frac * pw.max()
    return float(np.abs(p.amp[int(np.argmin(np.where(strong, dt, np.inf)))]))


def direct_amplitude(p: Paths, d0: float, tol: float = 0.3) -> float:
    """Amplitude of the found path whose length is d0 (known LoS length).

    A wall can make the direct path weaker than a reflection, so the power
    rule of relative_params then picks the reflection. With d0 known the
    direct path is the one at that length. NaN when no path is there: the
    loss is too high to see.
    """
    if len(p) == 0:
        return float("nan")
    from .geometry import C

    diff = ((p.tau - d0 / C + p.period / 2) % p.period - p.period / 2) * C
    near = np.abs(diff) < tol
    if not np.any(near):
        return float("nan")
    return float(np.max(np.abs(p.amp[near])))


def excess_db(paths: list[Paths], d0s, wavelength: float) -> np.ndarray:
    a = np.array([direct_amplitude(p, d0) for p, d0 in zip(paths, d0s)])
    free = wavelength / (4 * np.pi * np.asarray(d0s, float))
    return 20 * np.log10(free / a)


def crossings(ap_pos, clients, pieces: list[Segment], n_lines: int) -> np.ndarray:
    """A[k, w] = 1 when AP -> client k crosses a piece of wall line w."""
    a = np.zeros((len(clients), n_lines))
    for k, c in enumerate(clients):
        for s in pieces:
            if _cross(ap_pos, c, s) is not None:
                a[k, s.line] = 1.0
    return a


def fit_losses(excess: np.ndarray, a: np.ndarray, f_scale: float = 2.0, max_db: float = 40.0):
    """Offset c and loss per wall line, robust to wrong direct paths."""
    ok = np.isfinite(excess)
    e, m = excess[ok], a[ok]
    n = m.shape[1]
    x0 = np.r_[np.median(e), np.full(n, 3.0)]

    def res(x):
        return x[0] + m @ x[1:] - e

    lo = np.r_[-np.inf, np.zeros(n)]
    hi = np.r_[np.inf, np.full(n, max_db)]
    sol = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=f_scale)
    return float(sol.x[0]), sol.x[1:], m.sum(axis=0).astype(int)


def false_lines(loss: np.ndarray, count: np.ndarray, min_links: int = 10, min_db: float = 1.7) -> list[int]:
    """Wall lines that enough links cross with almost no loss.

    min_db: on plan A the false wall fits 0.1 dB and the weakest true wall
    3.4 dB. 1.7 dB is about halfway.
    """
    return [int(w) for w in range(len(loss)) if count[w] >= min_links and loss[w] < min_db]
