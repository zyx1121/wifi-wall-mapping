"""Stage 1: geometry with oracle path parameters (no CSI).

A. Oracle first-order parameters give the walls back exactly.
B. Sensitivity: noise on excess length, angle and d0, error per true wall.
C. Requirement table: the largest length noise that keeps the median wall
   error at or below 0.3 m, per angle noise.
D. 2.5D: AP and client heights plus floor and ceiling bounces, with the 2D
   inversion. Shows the bias and the false walls they cause.
"""

from __future__ import annotations

import common  # noqa: F401  (sets BLAS threads before numpy loads)
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import TARGETS, default_setup, match_row, out_dir, random_client, save_json, summarize
from wallmap.geometry import Room, image_paths, match_walls, oracle_params, wall_line, walls_from_paths

OUT = out_dir("stage1")
N_POS = 300


def per_wall_errors(ap_pos, room, phi0, d0, phis, dls, others):
    """Error of each first-order target wall, using the known association."""
    res = {}
    by = {w.name: w for w in room.walls()}
    for phi, dl, p in zip(phis, dls, others):
        if len(p.walls) != 1 or p.vertical or p.walls[0] not in TARGETS:
            continue
        th, rho = walls_from_paths(ap_pos, phi0, d0, np.array([phi]), np.array([dl]))[0]
        w = by[p.walls[0]]
        th_t, _ = wall_line(ap_pos, w)
        n = np.array([np.cos(th), np.sin(th)])
        res[p.walls[0]] = (
            abs(float(np.dot(w.mid - ap_pos, n)) - rho),
            abs(np.rad2deg((th - th_t + np.pi) % (2 * np.pi) - np.pi)),
        )
    return res


def part_a():
    rows_first, rows_all = [], []
    worst = 0.0
    for seed in range(N_POS):
        rng, room, ap = default_setup(seed)
        c = random_client(rng, room, ap.pos)
        paths = image_paths(room, ap, c, max_order=2)
        first = [p for p in paths if p.order <= 1 and p.walls != ("south",)]
        phi0, d0, phis, dls, _ = oracle_params(first)
        m = match_walls(walls_from_paths(ap.pos, phi0, d0, phis, dls), ap.pos, room)
        worst = max([worst] + list(m.dist_err.values()))
        rows_first.append(match_row(m))
        phi0, d0, phis, dls, _ = oracle_params(paths)
        rows_all.append(match_row(match_walls(walls_from_paths(ap.pos, phi0, d0, phis, dls), ap.pos, room)))
    return {
        "first_order_only": summarize(rows_first),
        "worst_dist_err_m": worst,
        "with_second_order": summarize(rows_all),
    }


def part_b():
    sig_l = [0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0]
    sig_phi = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    sig_d0 = [0.0, 0.1, 0.3, 1.0]
    rng0 = np.random.default_rng(123)
    geo = []
    for seed in range(N_POS):
        rng, room, ap = default_setup(seed)
        c = random_client(rng, room, ap.pos)
        paths = image_paths(room, ap, c, max_order=1)
        geo.append((room, ap, oracle_params(paths)))

    def run(sl, sp, sd):
        de, ae = [], []
        for room, ap, (phi0, d0, phis, dls, others) in geo:
            n = len(phis)
            r = per_wall_errors(
                ap.pos,
                room,
                phi0 + np.deg2rad(sp) * rng0.standard_normal(),
                max(0.1, d0 + sd * rng0.standard_normal()),
                phis + np.deg2rad(sp) * rng0.standard_normal(n),
                dls + sl * rng0.standard_normal(n),
                others,
            )
            for d, a in r.values():
                de.append(d)
                ae.append(a)
        return float(np.median(de)), float(np.percentile(de, 90)), float(np.median(ae))

    grid = np.zeros((len(sig_phi), len(sig_l), 3))
    for i, sp in enumerate(sig_phi):
        for j, sl in enumerate(sig_l):
            grid[i, j] = run(sl, sp, 0.0)
    d0_rows = {f"{sd}": run(0.0, 0.0, sd) for sd in sig_d0}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid[:, :, 0], origin="lower", cmap="viridis", aspect="auto", vmax=1.0)
    ax.set_xticks(range(len(sig_l)), [f"{v:g}" for v in sig_l])
    ax.set_yticks(range(len(sig_phi)), [f"{v:g}" for v in sig_phi])
    ax.set_xlabel("excess length noise sigma (m)")
    ax.set_ylabel("angle noise sigma (deg)")
    for i in range(len(sig_phi)):
        for j in range(len(sig_l)):
            v = grid[i, j, 0]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.6 else "k", fontsize=8)
    fig.colorbar(im, label="median wall distance error (m)")
    ax.set_title("Stage 1: wall error vs parameter noise (d0 exact)")
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity.png", dpi=150)

    req = {}
    for i, sp in enumerate(sig_phi):
        ok = [sl for j, sl in enumerate(sig_l) if grid[i, j, 0] <= 0.3]
        req[f"{sp:g}"] = max(ok) if ok else None
    return {
        "sig_l": sig_l,
        "sig_phi": sig_phi,
        "median": grid[:, :, 0].tolist(),
        "p90": grid[:, :, 1].tolist(),
        "ang_median": grid[:, :, 2].tolist(),
        "d0_noise": d0_rows,
        "max_sig_l_for_0p3m": req,
    }


def part_d():
    cases = {
        "2D": dict(fc=False, ap_h=1.5, c_h=1.5),
        "2.5D same height": dict(fc=True, ap_h=1.2, c_h=1.2),
        "2.5D AP 1.5 m, client 1.0 m": dict(fc=True, ap_h=1.5, c_h=1.0),
        "2.5D AP 2.0 m, client 0.8 m": dict(fc=True, ap_h=2.0, c_h=0.8),
    }
    out = {}
    for name, cfg in cases.items():
        rows, bias = [], []
        for seed in range(N_POS):
            rng, room, ap = default_setup(seed)
            ap.height = cfg["ap_h"]
            c = random_client(rng, room, ap.pos)
            paths = image_paths(room, ap, c, client_height=cfg["c_h"], max_order=2, floor_ceiling=cfg["fc"])
            phi0, d0, phis, dls, others = oracle_params(paths)
            rows.append(match_row(match_walls(walls_from_paths(ap.pos, phi0, d0, phis, dls), ap.pos, room)))
            bias += [d for d, _ in per_wall_errors(ap.pos, room, phi0, d0, phis, dls, others).values()]
        out[name] = summarize(rows) | {"first_order_bias_med": float(np.median(bias)), "first_order_bias_p90": float(np.percentile(bias, 90))}
    return out


if __name__ == "__main__":
    res = {"A_oracle": part_a(), "B_sensitivity": part_b(), "D_heights": part_d(), "positions": N_POS}
    save_json(OUT / "stage1.json", res)
    print("A", res["A_oracle"])
    print("B req (angle sigma deg -> max length sigma m for median <= 0.3 m):", res["B_sensitivity"]["max_sig_l_for_0p3m"])
    print("B d0 noise -> (median, p90, ang):", res["B_sensitivity"]["d0_noise"])
    for k, v in res["D_heights"].items():
        print("D", k, v)
