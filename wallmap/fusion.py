"""Multi-link fusion: cluster candidate walls, check consistency, solve d0.

A link is one client position. It carries the LoS angle phi0, a LoS length d0
(known, noisy or unknown) and the measured non-LoS paths (phis, dls).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .geometry import client_position, walls_from_paths


@dataclass
class Link:
    phi0: float
    d0: float
    phis: np.ndarray
    dls: np.ndarray

    def candidates(self, ap_pos: np.ndarray, d0: float | None = None) -> np.ndarray:
        return walls_from_paths(ap_pos, self.phi0, self.d0 if d0 is None else d0, self.phis, self.dls)


def _foot(lines: np.ndarray) -> np.ndarray:
    return lines[:, 1:2] * np.stack([np.cos(lines[:, 0]), np.sin(lines[:, 0])], axis=1)


def cluster_walls(lines: np.ndarray, eps: float = 0.4, min_size: int = 2) -> list[tuple[float, float, int]]:
    """Average-linkage clustering of wall lines by their foot point.

    The foot point is the point of the line closest to the AP. Returns
    (theta, rho, size) for each cluster with at least min_size members.
    """
    if len(lines) == 0:
        return []
    if len(lines) == 1:
        return [(float(lines[0, 0]), float(lines[0, 1]), 1)] if min_size <= 1 else []
    lab = fcluster(linkage(_foot(lines), method="average"), t=eps, criterion="distance")
    out = []
    for c in np.unique(lab):
        mem = lines[lab == c]
        if len(mem) < min_size:
            continue
        th = float(np.arctan2(np.mean(np.sin(mem[:, 0])), np.mean(np.cos(mem[:, 0]))))
        out.append((th, float(np.median(mem[:, 1])), len(mem)))
    return out


def mirror_line(p: np.ndarray, ap_pos: np.ndarray, th: float, rho: float) -> np.ndarray:
    n = np.array([np.cos(th), np.sin(th)])
    return p - 2 * (np.dot(p - ap_pos, n) - rho) * n


def predict(ap_pos, link: Link, d0: float, th: float, rho: float) -> tuple[float, float]:
    """Angle and excess length of the first-order path off a wall line."""
    pc = client_position(ap_pos, link.phi0, d0)
    img = mirror_line(pc, ap_pos, th, rho)
    v = img - ap_pos
    L = float(np.linalg.norm(v))
    return float(np.arccos(np.clip(v[0] / L, -1, 1))), L - d0


def support(ap_pos, links: list[Link], d0s, th, rho, tol_phi_deg=5.0, tol_l=0.5) -> float:
    tp = np.deg2rad(tol_phi_deg)
    hit = 0
    for lk, d0 in zip(links, d0s):
        if len(lk.phis) == 0:
            continue
        ph, dl = predict(ap_pos, lk, d0, th, rho)
        if np.any((np.abs(lk.phis - ph) < tp) & (np.abs(lk.dls - dl) < tol_l)):
            hit += 1
    return hit / max(1, len(links))


def fuse(
    ap_pos: np.ndarray,
    links: list[Link],
    d0s: list[float] | None = None,
    min_frac: float = 0.3,
    eps: float = 0.4,
    check: bool = False,
    min_support: float = 0.3,
) -> np.ndarray:
    d0s = [lk.d0 for lk in links] if d0s is None else d0s
    cand = [lk.candidates(ap_pos, d0) for lk, d0 in zip(links, d0s)]
    lines = np.concatenate(cand) if cand else np.zeros((0, 2))
    if len(links) == 1:
        return lines
    min_size = max(2, int(np.ceil(min_frac * len(links))))
    walls = cluster_walls(lines, eps, min_size)
    if check:
        walls = [w for w in walls if support(ap_pos, links, d0s, w[0], w[1]) >= min_support]
    return np.array([(w[0], w[1]) for w in walls], dtype=float).reshape(-1, 2)


def solve_d0(
    ap_pos: np.ndarray,
    links: list[Link],
    d0_init: float = 3.0,
    rounds: int = 6,
    grid=np.arange(0.3, 10.0, 0.02),
    sig_l=0.3,
    sig_phi_deg=3.0,
    cap=9.0,
    **fuse_kw,
) -> tuple[np.ndarray, list[float]]:
    """Alternate between fusing walls and fitting each link's d0 to them."""
    d0s = [d0_init] * len(links)
    sp = np.deg2rad(sig_phi_deg)
    walls = np.zeros((0, 2))
    for _ in range(rounds):
        walls = fuse(ap_pos, links, d0s, **fuse_kw)
        if len(walls) == 0:
            break
        new = []
        for lk in links:
            if len(lk.phis) == 0:
                new.append(d0_init)
                continue
            cost = np.zeros(len(grid))
            for th, rho in walls:
                pc = ap_pos[None, :] + grid[:, None] * np.array([np.cos(lk.phi0), np.sin(lk.phi0)])[None, :]
                n = np.array([np.cos(th), np.sin(th)])
                img = pc - 2 * ((pc - ap_pos) @ n - rho)[:, None] * n[None, :]
                v = img - ap_pos
                L = np.linalg.norm(v, axis=1)
                ph = np.arccos(np.clip(v[:, 0] / L, -1, 1))
                dl = L - grid
                r = ((lk.dls[None, :] - dl[:, None]) / sig_l) ** 2 + ((lk.phis[None, :] - ph[:, None]) / sp) ** 2
                cost += np.minimum(r.min(axis=1), cap)
            new.append(float(grid[np.argmin(cost)]))
        d0s = new
    return fuse(ap_pos, links, d0s, **fuse_kw), d0s


def _curves(ap_pos, lk: Link, grid: np.ndarray) -> np.ndarray:
    """Candidate lines of one link for every d0 on the grid: (G, K, 2) foot points."""
    u0 = np.array([np.cos(lk.phi0), np.sin(lk.phi0)])
    uk = np.stack([np.cos(lk.phis), np.sin(lk.phis)], axis=1)  # (K, 2)
    pc = ap_pos + grid[:, None] * u0  # (G, 2)
    img = ap_pos + (grid[:, None] + lk.dls[None, :])[:, :, None] * uk[None, :, :]  # (G, K, 2)
    v = img - pc[:, None, :]
    nv = np.linalg.norm(v, axis=2, keepdims=True)
    n = v / np.maximum(nv, 1e-9)
    rho = np.sum(n * ((pc[:, None, :] + img) / 2 - ap_pos), axis=2, keepdims=True)
    return n * rho  # foot point; the sign flip of canonical() does not change it


def hough_d0(
    ap_pos: np.ndarray,
    links: list[Link],
    grid=np.arange(0.5, 10.0, 0.05),
    cell: float = 0.2,
    extent: float = 12.0,
    min_frac: float = 0.3,
    top_frac: float = 0.6,
    refine_rounds: int = 3,
    sig_l=0.3,
    sig_phi_deg=3.0,
    cap=9.0,
) -> list[float]:
    """Solve each link's LoS length without an initial guess.

    For every link and path, sweeping d0 traces a curve of candidate walls.
    Each link votes once per cell on its curve. A true wall collects one vote
    from every link that sees it, at that link's true d0. Peaks give the walls,
    then each link takes the d0 that best explains them.
    """
    from scipy.ndimage import maximum_filter

    nb = int(2 * extent / cell)
    acc = np.zeros((nb, nb), int)
    for lk in links:
        if len(lk.phis) == 0:
            continue
        f = _curves(ap_pos, lk, grid).reshape(-1, 2)
        ij = np.floor((f + extent) / cell).astype(int)
        ok = np.all((ij >= 0) & (ij < nb), axis=1)
        cells = np.unique(ij[ok], axis=0)
        acc[cells[:, 0], cells[:, 1]] += 1
    need = max(2, int(np.ceil(min_frac * len(links))), int(np.ceil(top_frac * acc.max())))
    pk = np.argwhere((acc == maximum_filter(acc, size=3)) & (acc >= need))
    walls = []
    for i, j in pk:
        foot = (np.array([i, j]) + 0.5) * cell - extent
        rho = float(np.linalg.norm(foot))
        if rho < 0.2:
            continue
        walls.append((float(np.arctan2(foot[1], foot[0])), rho))
    for _ in range(refine_rounds + 1):
        d0s = _fit_d0(ap_pos, links, walls, grid, sig_l, sig_phi_deg, cap)
        refined = fuse(ap_pos, links, d0s, min_frac=min_frac)
        if len(refined) == 0:
            break
        walls = [(float(t), float(r)) for t, r in refined]
    return d0s


def _fit_d0(ap_pos, links, walls, grid, sig_l, sig_phi_deg, cap) -> list[float]:
    d0s = []
    sp = np.deg2rad(sig_phi_deg)
    for lk in links:
        if len(lk.phis) == 0 or not walls:
            d0s.append(float(np.median(grid)))
            continue
        cost = np.zeros(len(grid))
        pc = ap_pos[None, :] + grid[:, None] * np.array([np.cos(lk.phi0), np.sin(lk.phi0)])[None, :]
        for th, rho in walls:
            n = np.array([np.cos(th), np.sin(th)])
            img = pc - 2 * ((pc - ap_pos) @ n - rho)[:, None] * n[None, :]
            v = img - ap_pos
            L = np.linalg.norm(v, axis=1)
            ph = np.arccos(np.clip(v[:, 0] / L, -1, 1))
            r = ((lk.dls[None, :] - (L - grid)[:, None]) / sig_l) ** 2 + ((lk.phis[None, :] - ph[:, None]) / sp) ** 2
            cost += np.minimum(r.min(axis=1), cap)
        d0s.append(float(grid[np.argmin(cost)]))
    return d0s
