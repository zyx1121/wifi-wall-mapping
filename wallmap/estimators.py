"""Multipath parameter estimators on one CSI packet.

Each estimator returns paths as (s, tau, amp): s = u_x = cos(phi) in [-1, 1],
tau in seconds (only differences between paths matter) and the complex
amplitude from a least-squares fit with wideband atoms.

Estimators:
  ifft   : 2D FFT (antennas x tones, Hann over tones), peaks above a noise gate
  omp    : greedy on the oversampled FFT grid, LS refit
  nomp   : OMP plus Newton refinement off the grid (Mamandipoor et al. 2016)
  music  : joint angle-delay MUSIC with antenna and frequency smoothing plus
           forward-backward averaging (SpotFi style)
  mpencil: matrix pencil on the delay axis, then beamforming per delay
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

from .csi import DF

GAMMA_DETECT = 15.0  # |corr|^2 / (MN sigma^2) gate, about P_fa 1e-2 over the map


@dataclass
class Paths:
    s: np.ndarray
    tau: np.ndarray
    amp: np.ndarray
    period: float  # delays are known modulo this period (s)

    def __len__(self):
        return len(self.s)


def _ctr(m: int) -> np.ndarray:
    return np.arange(m) - (m - 1) / 2


def atoms(s: np.ndarray, tau: np.ndarray, m: int, n_k: np.ndarray, fc: float) -> np.ndarray:
    """Wideband atoms, shape (M*N, K)."""
    s = np.atleast_1d(s)
    tau = np.atleast_1d(tau)
    fr = 1 + n_k * DF / fc  # (N,)
    sp = np.pi * _ctr(m)[:, None, None] * fr[None, :, None] * s[None, None, :]
    dl = -2 * np.pi * n_k[None, :, None] * DF * tau[None, None, :]
    return np.exp(1j * (sp + dl)).reshape(m * len(n_k), -1)


def ls_amplitudes(y: np.ndarray, s, tau, n_k, fc) -> np.ndarray:
    if len(np.atleast_1d(s)) == 0:
        return np.zeros(0, complex)
    a = atoms(s, tau, y.shape[0], n_k, fc)
    amp, *_ = np.linalg.lstsq(a, y.reshape(-1), rcond=None)
    return amp


# ---------------------------------------------------------------- FFT map


def _fft_sizes(m: int, n_k: np.ndarray, os_t: int = 4) -> tuple[int, int]:
    span = int(n_k.max() - n_k.min() + 1)
    pt = 1 << int(np.ceil(np.log2(os_t * span)))
    return 64, pt


def fft_map(y: np.ndarray, n_k: np.ndarray, win: np.ndarray | None = None):
    """Correlation of y with narrowband atoms on a grid.

    Returns (F, s_grid, tau_grid). F[p, q] = sum y * conj(atom(s_p, tau_q)),
    up to a unit-modulus factor.
    """
    m, _ = y.shape
    ps, pt = _fft_sizes(m, n_k)
    z = np.zeros((ps, pt), complex)
    yy = y if win is None else y * win[None, :]
    z[:m, n_k % pt] = yy
    f = np.fft.fft(np.fft.ifft(z, axis=1) * pt, axis=0)
    p = np.arange(ps)
    s = 2 * p / ps
    s[s >= 1] -= 2
    tau = np.arange(pt) / (pt * DF)
    return f, s, tau


def _peaks(p2: np.ndarray, k: int, size=(3, 5)) -> list[tuple[int, int]]:
    mx = maximum_filter(p2, size=size, mode="wrap")
    cand = np.argwhere((p2 == mx) & (p2 > 0))
    vals = p2[cand[:, 0], cand[:, 1]]
    order = np.argsort(vals)[::-1][:k]
    return [tuple(cand[i]) for i in order]


# ---------------------------------------------------------------- IFFT


def est_ifft(y, n_k, s2, fc, k_max=20, k_fixed=None, trace=None) -> Paths:
    m, n = y.shape
    w = np.hanning(n + 2)[1:-1]
    f, sg, tg = fft_map(y, n_k, w)
    p2 = np.abs(f) ** 2
    gate = -1.0 if k_fixed else GAMMA_DETECT * s2 * m * np.sum(w**2)
    pk = [(i, j) for i, j in _peaks(p2, k_fixed or k_max) if p2[i, j] > gate]
    s = np.array([sg[i] for i, _ in pk])
    tau = np.array([tg[j] for _, j in pk])
    amp = ls_amplitudes(y, s, tau, n_k, fc)
    if trace is not None:
        trace.append({"kind": "map", "map": p2, "s": sg, "tau": tg, "gate": gate})
        trace.append({"kind": "peaks", "map": p2, "s": sg, "tau": tg, "gate": gate, "picked": list(zip(s, tau))})
    return Paths(s, tau, amp, 1 / DF)


# ---------------------------------------------------------------- OMP / NOMP


def _newton_single(y, s, tau, n_k, fc, iters=6):
    """Refine one path (s, tau) to maximise |a^H y|^2 on residual y."""
    m = y.shape[0]
    fr = 1 + n_k * DF / fc
    dps = np.pi * _ctr(m)[:, None] * fr[None, :]  # d psi / d s, (M, N)
    dpt = np.broadcast_to(-2 * np.pi * n_k[None, :] * DF * 1e-9, dps.shape)  # d psi / d tau_ns
    t_ns = tau * 1e9

    def evalz(s_, t_):
        psi = dps * s_ - 2 * np.pi * n_k[None, :] * DF * t_ * 1e-9
        e = np.exp(-1j * psi) * y
        z = e.sum()
        zs = (-1j * dps * e).sum()
        zt = (-1j * dpt * e).sum()
        zss = -(dps * dps * e).sum()
        ztt = -(dpt * dpt * e).sum()
        zst = -(dps * dpt * e).sum()
        return z, np.array([zs, zt]), np.array([[zss, zst], [zst, ztt]])

    z, g1, h1 = evalz(s, t_ns)
    obj = abs(z) ** 2
    for _ in range(iters):
        g = 2 * np.real(np.conj(z) * g1)
        h = 2 * np.real(np.conj(g1)[:, None] * g1[None, :] + np.conj(z) * h1)
        if np.all(np.linalg.eigvalsh(h) < 0):
            step = -np.linalg.solve(h, g)
        else:
            break
        ok = False
        for _ls in range(4):
            s_new = float(np.clip(s + step[0], -1, 1))
            t_new = t_ns + step[1]
            z2, g2, h2 = evalz(s_new, t_new)
            if abs(z2) ** 2 >= obj:
                s, t_ns, z, g1, h1, obj = s_new, t_new, z2, g2, h2, abs(z2) ** 2
                ok = True
                break
            step = step / 2
        if not ok or np.max(np.abs(step)) < 1e-6:
            break
    return s, t_ns * 1e-9


def _greedy(y, n_k, s2, fc, k_max, refine: bool, cyclic_rounds=1, k_fixed=None, trace=None) -> Paths:
    m, n = y.shape
    s_l: list[float] = []
    t_l: list[float] = []
    res = y.copy()
    gate = -1.0 if k_fixed else GAMMA_DETECT * s2 * m * n
    for _ in range(k_fixed or k_max):
        f, sg, tg = fft_map(res, n_k)
        p2 = np.abs(f) ** 2
        i, j = np.unravel_index(np.argmax(p2), p2.shape)
        if p2[i, j] < gate:
            if trace is not None:
                trace.append({"kind": "stop", "map": p2, "s": sg, "tau": tg, "gate": gate, "found": list(zip(s_l, t_l))})
            break
        s_grid, t_grid = float(sg[i]), float(tg[j])
        s_new, t_new = s_grid, t_grid
        if refine:
            s_new, t_new = _newton_single(res, s_new, t_new, n_k, fc)
        if trace is not None:
            trace.append({"kind": "pick", "map": p2, "s": sg, "tau": tg, "gate": gate, "found": list(zip(s_l, t_l)), "grid": (s_grid, t_grid), "refined": (s_new, t_new) if refine else None})
        s_l.append(s_new)
        t_l.append(t_new)
        if refine:
            for _r in range(cyclic_rounds):
                amp = ls_amplitudes(y, s_l, t_l, n_k, fc)
                a_all = atoms(np.array(s_l), np.array(t_l), m, n_k, fc)
                full = (a_all @ amp).reshape(m, n)
                for q in range(len(s_l)):
                    own = (a_all[:, q] * amp[q]).reshape(m, n)
                    s_l[q], t_l[q] = _newton_single(y - full + own, s_l[q], t_l[q], n_k, fc, iters=3)
        amp = ls_amplitudes(y, s_l, t_l, n_k, fc)
        res = y - (atoms(np.array(s_l), np.array(t_l), m, n_k, fc) @ amp).reshape(m, n)
    s = np.array(s_l)
    t = np.array(t_l)
    return Paths(s, t, ls_amplitudes(y, s, t, n_k, fc), 1 / DF)


def est_omp(y, n_k, s2, fc, k_max=20, k_fixed=None, trace=None) -> Paths:
    return _greedy(y, n_k, s2, fc, k_max, refine=False, k_fixed=k_fixed, trace=trace)


def est_nomp(y, n_k, s2, fc, k_max=20, k_fixed=None, trace=None) -> Paths:
    return _greedy(y, n_k, s2, fc, k_max, refine=True, k_fixed=k_fixed, trace=trace)


# ---------------------------------------------------------------- subspace


def _smoothing_plan(n_k: np.ndarray, lf_max=64):
    span = int(n_k.max() - n_k.min() + 1)
    d = int(np.clip(span // (2 * (lf_max - 1)), 1, 32))
    lf = lf_max
    return d, lf


def _snapshots(y, n_k, la, lf, d):
    """Smoothed snapshots (la*lf, n_snap) over antenna and tone shifts."""
    m, _ = y.shape
    base = n_k.min()
    span = int(n_k.max() - base + 1)
    grid = np.full((m, span), np.nan + 0j)
    grid[:, n_k - base] = y
    present = ~np.isnan(grid[0].real)
    taps = d * np.arange(lf)
    starts = [j for j in range(span - taps[-1]) if present[j + taps].all()]
    cols = []
    for a0 in range(m - la + 1):
        blk = grid[a0 : a0 + la][:, np.array(starts)[:, None] + taps[None, :]]  # (la, S, lf)
        cols.append(blk.transpose(0, 2, 1).reshape(la * lf, -1))
    return np.concatenate(cols, axis=1)


def _fb_cov(x: np.ndarray) -> np.ndarray:
    r = x @ x.conj().T / x.shape[1]
    j = np.eye(r.shape[0])[::-1]
    return (r + j @ r.conj() @ j) / 2


def _order(ev: np.ndarray, s2: float, dim: int, n_snap: int, k_max: int) -> int:
    thr = s2 * (1 + np.sqrt(dim / n_snap)) ** 2 * 2.0
    return int(min(np.sum(ev > thr), k_max))


def _strongest_tau(y, n_k) -> float:
    f, _, tg = fft_map(y, n_k)
    return float(tg[np.unravel_index(np.argmax(np.abs(f)), f.shape)[1]])


def est_music(y, n_k, s2, fc, k_max=20, k_fixed=None, n_s=401, dt_ns=0.25, trace=None) -> Paths:
    m, _ = y.shape
    la = m if m <= 2 else m // 2 + 1
    d, lf = _smoothing_plan(n_k)
    x = _snapshots(y, n_k, la, lf, d)
    r = _fb_cov(x)
    ev, evec = np.linalg.eigh(r)
    ev, evec = ev[::-1], evec[:, ::-1]
    k = k_fixed or _order(ev, s2, la * lf, x.shape[1], k_max)
    period = 1 / (d * DF)
    if trace is not None:
        trace.append({"kind": "eig", "ev": ev[:40], "thr": s2 * (1 + np.sqrt(la * lf / x.shape[1])) ** 2 * 2.0, "k": k})
    if k == 0:
        return Paths(np.zeros(0), np.zeros(0), np.zeros(0, complex), period)
    es = evec[:, :k].reshape(la, lf, k)
    t0 = _strongest_tau(y, n_k)
    tg = t0 + np.arange(-50e-9, 250e-9, dt_ns * 1e-9)
    sg = np.linspace(-1, 1, n_s)
    a_s = np.exp(1j * np.pi * np.arange(la)[None, :] * sg[:, None])  # (S, la)
    a_t = np.exp(-2j * np.pi * d * DF * np.arange(lf)[None, :] * tg[:, None])  # (T, lf)
    bs = np.einsum("si,ijk->sjk", a_s, es.conj())  # (S, lf, k)
    # (T, lf) @ (lf, S*k) with BLAS, then sum |.|^2 over k
    q = a_t @ bs.transpose(1, 0, 2).reshape(lf, n_s * k)
    proj = np.sum(np.abs(q.reshape(len(tg), n_s, k)) ** 2, axis=2).T  # (S, T)
    pm = 1 / np.maximum(la * lf - proj, 1e-9)
    pk = _peaks(pm, k, size=(5, 5))
    s = np.array([sg[i] for i, _ in pk])
    t = np.array([tg[j] for _, j in pk])
    amp = ls_amplitudes(y, s, t, n_k, fc)
    keep = np.abs(amp) ** 2 * y.size > (0 if k_fixed else GAMMA_DETECT * s2)
    if trace is not None:
        trace.append({"kind": "spectrum", "map": pm, "s": sg, "tau": tg})
        trace.append({"kind": "peaks", "map": pm, "s": sg, "tau": tg, "picked": list(zip(s[keep], t[keep])), "dropped": list(zip(s[~keep], t[~keep]))})
    return Paths(s[keep], t[keep], amp[keep], period)


def est_mpencil(y, n_k, s2, fc, k_max=20, k_fixed=None, trace=None) -> Paths:
    m, _ = y.shape
    d, lf = _smoothing_plan(n_k)
    x = _snapshots(y, n_k, 1, lf, d)
    r = _fb_cov(x)
    ev, evec = np.linalg.eigh(r)
    ev, evec = ev[::-1], evec[:, ::-1]
    k = k_fixed or _order(ev, s2, lf, x.shape[1], k_max)
    period = 1 / (d * DF)
    if trace is not None:
        trace.append({"kind": "eig", "ev": ev[:40], "thr": s2 * (1 + np.sqrt(lf / x.shape[1])) ** 2 * 2.0, "k": k})
    if k == 0:
        return Paths(np.zeros(0), np.zeros(0), np.zeros(0, complex), period)
    es = evec[:, :k]
    phi = np.linalg.pinv(es[:-1]) @ es[1:]
    z = np.linalg.eigvals(phi)
    t0 = _strongest_tau(y, n_k)
    tau = (-np.angle(z) / (2 * np.pi * d * DF)) % period
    tau = t0 - 50e-9 + ((tau - (t0 - 50e-9)) % period)
    # amplitude per antenna for each delay, then beamform for the angle
    e = np.exp(-2j * np.pi * n_k[:, None] * DF * tau[None, :])  # (N, k)
    b, *_ = np.linalg.lstsq(e, y.T, rcond=None)  # (k, M)
    sg = np.linspace(-1, 1, 2001)
    steer = np.exp(1j * np.pi * _ctr(m)[None, :] * sg[:, None])  # (S, M)
    beams = np.abs(steer.conj() @ b.T)  # (S, k)
    s = sg[np.argmax(beams, axis=0)]
    amp = ls_amplitudes(y, s, tau, n_k, fc)
    keep = np.abs(amp) ** 2 * y.size > (0 if k_fixed else GAMMA_DETECT * s2)
    if trace is not None:
        trace.append({"kind": "delays", "tau_lines": tau})
        trace.append({"kind": "beams", "tau_lines": tau, "beams": beams, "s": sg, "picked": list(zip(s[keep], tau[keep])), "dropped": list(zip(s[~keep], tau[~keep]))})
    return Paths(s[keep], tau[keep], amp[keep], period)


ESTIMATORS = {
    "ifft": est_ifft,
    "omp": est_omp,
    "nomp": est_nomp,
    "music": est_music,
    "mpencil": est_mpencil,
}


# ---------------------------------------------------------------- to geometry


def relative_params(p: Paths, los_power_frac: float = 0.25):
    """LoS = earliest path with power >= frac * max power.

    Returns (phi0, phis, dls) where dls are excess path lengths in metres,
    or None when no path was found.
    """
    if len(p) == 0:
        return None
    pw = np.abs(p.amp) ** 2
    ref = p.tau[np.argmax(pw)]
    dt = (p.tau - ref + p.period / 2) % p.period - p.period / 2
    strong = pw >= los_power_frac * pw.max()
    i0 = int(np.argmin(np.where(strong, dt, np.inf)))
    phi = np.arccos(np.clip(p.s, -1, 1))
    rel = dt - dt[i0]
    others = np.array([i for i in range(len(p)) if i != i0 and rel[i] > 0])
    if len(others) == 0:
        return phi[i0], np.zeros(0), np.zeros(0)
    from .geometry import C

    return phi[i0], phi[others], rel[others] * C
