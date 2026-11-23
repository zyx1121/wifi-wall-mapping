"""Phone dead reckoning (PDR) fused with the AP's direct-path measurements.

The phone counts steps and reads its heading. Each step has a length error
(one scale per walk plus noise per step) and a heading error (an unknown
start offset, a slow random-walk bias, and noise per step).

The AP measures the direct path of every packet: cos(angle) from its
linear array, and with FTM the range. An extended Kalman filter (EKF)
fuses both. The state is position, heading offset and step scale. The
start heading is unknown, so the filter runs one hypothesis per start
heading and keeps the one whose measurements fit best.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImuNoise:
    scale_sd: float = 0.04  # step length scale per walk
    length_sd: float = 0.02  # step length noise, fraction of the step
    heading_sd_deg: float = 3.0  # heading noise per step
    bias_sd_deg: float = 0.2  # random-walk step of the heading bias, per step


def simulate(track: np.ndarray, noise: ImuNoise, rng: np.random.Generator):
    """Measured step lengths and headings along a true track (N points, N-1 steps)."""
    d = np.diff(track, axis=0)
    length = np.linalg.norm(d, axis=1)
    head = np.arctan2(d[:, 1], d[:, 0])
    scale = 1 + noise.scale_sd * rng.standard_normal()
    l_m = length * scale * (1 + noise.length_sd * rng.standard_normal(len(length)))
    offset = rng.uniform(-np.pi, np.pi)  # phone frame vs AP frame
    bias = np.cumsum(np.deg2rad(noise.bias_sd_deg) * rng.standard_normal(len(length)))
    h_m = head + offset + bias + np.deg2rad(noise.heading_sd_deg) * rng.standard_normal(len(length))
    return l_m, h_m, offset


def dead_reckon(start, lengths, headings, offset: float = 0.0) -> np.ndarray:
    """PDR alone from a known start position and heading offset."""
    steps = lengths[:, None] * np.stack([np.cos(headings - offset), np.sin(headings - offset)], axis=1)
    return np.vstack([start, start + np.cumsum(steps, axis=0)])


def _ekf(ap, lengths, headings, cos_m, ranges, noise: ImuNoise, theta0: float, cos_sd: float, range_sd: float, gate: float):
    n = len(cos_m)
    # start: on the first measured bearing, at the FTM range or 2 m
    c0 = np.clip(cos_m[0], -0.999, 0.999) if np.isfinite(cos_m[0]) else 0.0
    r0 = ranges[0] if ranges is not None else 2.0
    u = np.array([c0, np.sqrt(1 - c0**2)])
    x = np.array([*(ap + r0 * u), theta0, 1.0])
    p = np.diag([1.0, 1.0, np.deg2rad(15) ** 2, noise.scale_sd**2])
    if ranges is None:
        # range unknown: wide along the ray
        p[:2, :2] = 0.05 * np.eye(2) + 4.0 * np.outer(u, u)
    out = [x[:2].copy()]
    cost, used = 0.0, 0
    sl, sh = noise.length_sd, np.deg2rad(noise.heading_sd_deg)
    qb = np.deg2rad(noise.bias_sd_deg) ** 2
    for k in range(n - 1):
        # predict one step: heading in the AP frame is measured heading minus offset
        h = headings[k] - x[2]
        l = lengths[k]
        s = x[3]
        dx = s * l * np.array([np.cos(h), np.sin(h)])
        f = np.eye(4)
        f[0, 2], f[1, 2] = s * l * np.sin(h), -s * l * np.cos(h)
        f[0, 3], f[1, 3] = l * np.cos(h), l * np.sin(h)
        x = x + np.r_[dx, 0, 0]
        g = np.array([[np.cos(h) * s * l, -np.sin(h) * s * l], [np.sin(h) * s * l, np.cos(h) * s * l]])
        q = np.zeros((4, 4))
        q[:2, :2] = g @ np.diag([sl**2, sh**2]) @ g.T + 1e-4 * np.eye(2)
        q[2, 2] = qb
        q[3, 3] = 1e-6
        p = f @ p @ f.T + q
        # update with the measurements of position k+1
        zs, hs, js, rs = [], [], [], []
        v = x[:2] - ap
        r = max(float(np.linalg.norm(v)), 0.1)
        if np.isfinite(cos_m[k + 1]):
            zs.append(cos_m[k + 1])
            hs.append(v[0] / r)
            js.append([v[1] ** 2 / r**3, -v[0] * v[1] / r**3, 0, 0])
            rs.append(cos_sd**2)
        if ranges is not None:
            zs.append(ranges[k + 1])
            hs.append(r)
            js.append([v[0] / r, v[1] / r, 0, 0])
            rs.append(range_sd**2)
        for z, hz, jz, rz in zip(zs, hs, js, rs):
            jz = np.array(jz)
            sv = float(jz @ p @ jz + rz)
            inn = z - hz
            nis = inn**2 / sv
            if nis > gate:
                cost += gate
                continue
            kg = p @ jz / sv
            x = x + kg * inn
            p = (np.eye(4) - np.outer(kg, jz)) @ p
            cost += nis
            used += 1
        x[1] = max(x[1], ap[1] + 0.05)  # the AP is on the south wall
        out.append(x[:2].copy())
    return np.array(out), cost


def track(ap, lengths, headings, cos_m, ranges=None, noise: ImuNoise | None = None, n_hyp: int = 12, cos_sd: float = 0.03, range_sd: float = 0.3, gate: float = 9.0) -> np.ndarray:
    """EKF track from PDR and the AP's direct-path angle (and range when given)."""
    noise = noise or ImuNoise()
    best, best_cost = None, np.inf
    for th in np.linspace(-np.pi, np.pi, n_hyp, endpoint=False):
        # offset hypothesis: the first heading points along th in the AP frame
        t, c = _ekf(np.asarray(ap, float), lengths, headings, cos_m, ranges, noise, headings[0] - th, cos_sd, range_sd, gate)
        if c < best_cost:
            best, best_cost = t, c
    return best
