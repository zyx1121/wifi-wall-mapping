"""Stage 4: fuse many client positions into one wall map.

Each trial draws a random rectangular room and AP position on the south
wall, then 100 client positions. Fusion uses the first N positions.
d0 modes: known (true LoS length), ftm (known + 0.3 m noise), joint
(unknown, solved by Hough voting, see fusion.hough_d0). The consistency check keeps a cluster wall
only when its predicted first-order path shows up in at least 30% of links.
2.5D adds AP and client heights and floor and ceiling bounces.
"""

from __future__ import annotations

import itertools
import sys

import common  # noqa: F401  (BLAS threads)
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import match_row, out_dir, pool_map, random_client, run_link, save_json, summarize
from wallmap.csi import ALL_CALIBRATED
from wallmap.fusion import Link, fuse, hough_d0
from wallmap.geometry import Room, match_walls

OUT = out_dir("stage4")
# seeds 3000-3009 were used to tune the fusion parameters, the reported run uses 5000+
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N_MAX = 100
NS = [1, 5, 10, 20, 50, 100]
BWS = [160, 320]
ESTS = ["ifft", "nomp", "music"]
DIMS = ["2D", "2.5D"]
ANTS = 4
SNR = 30.0


def trial(args):
    seed, bw, est, dim = args
    rng = np.random.default_rng(seed)
    room = Room(width=rng.uniform(3.5, 7.0), depth=rng.uniform(3.0, 6.0))
    ap_pos = np.array([rng.uniform(1.0, room.width - 1.0), 0.1])
    fc = dim == "2.5D"
    links, truth_d0 = [], []
    for i in range(N_MAX):
        c = random_client(rng, room, ap_pos)
        ch = float(rng.uniform(0.8, 1.3)) if fc else 1.5
        rel, d0, *_ = run_link(seed * 1000 + i, bw, ANTS, SNR, est, ALL_CALIBRATED, floor_ceiling=fc, room=room, ap_pos=ap_pos, client=c, ap_h=1.5, client_h=ch)
        if rel is None:
            rel = (np.pi / 2, np.zeros(0), np.zeros(0))
        links.append(Link(rel[0], d0, rel[1], rel[2]))
        truth_d0.append(d0)
    out = []
    for n in NS:
        sub = links[:n]
        modes = {"known": [lk.d0 for lk in sub], "ftm": [lk.d0 + 0.3 * rng.standard_normal() for lk in sub]}
        for mode, d0s in modes.items():
            for check in (False, True):
                if n == 1 and check:
                    continue
                walls = fuse(ap_pos, sub, d0s, check=check)
                out.append(((n, mode, check), match_row(match_walls(walls, ap_pos, room))))
        if n >= 5:
            d0_est = hough_d0(ap_pos, sub)
            for check in (False, True):
                walls = fuse(ap_pos, sub, d0_est, check=check)
                r = match_row(match_walls(walls, ap_pos, room))
                r["d0_err_med"] = float(np.median(np.abs(np.array(d0_est) - np.array(truth_d0[:n]))))
                out.append(((n, "joint", check), r))
    return (bw, est, dim), out


def main():
    jobs = [(5000 + t, bw, est, dim) for t, bw, est, dim in itertools.product(range(TRIALS), BWS, ESTS, DIMS)]
    res = pool_map(trial, jobs)
    groups: dict = {}
    for key, rows in res:
        for sub, r in rows:
            groups.setdefault(key + sub, []).append(r)
    table = []
    for k, v in groups.items():
        row = dict(zip(["bw", "est", "dim", "n", "d0", "check"], k)) | summarize(v)
        if "d0_err_med" in v[0]:
            row["d0_err_med"] = float(np.median([x["d0_err_med"] for x in v]))
        table.append(row)
    save_json(OUT / "stage4.json", {"trials": TRIALS, "table": table})

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for col, dim in enumerate(DIMS):
        for est in ESTS:
            for mode, ls in (("known", "-"), ("joint", ":")):
                for bw, mk in ((160, "o"), (320, "s")):
                    sel = [r for r in table if r["bw"] == bw and r["est"] == est and r["dim"] == dim and r["d0"] == mode and r["check"] is True or (r["n"] == 1 and r["bw"] == bw and r["est"] == est and r["dim"] == dim and r["d0"] == mode)]
                    sel = sorted(sel, key=lambda r: r["n"])
                    if not sel:
                        continue
                    xs = [r["n"] for r in sel]
                    axes[0, col].plot(xs, [np.nan if r["dist_med"] is None else r["dist_med"] for r in sel], ls=ls, marker=mk, label=f"{est} {bw} {mode}")
                    axes[1, col].plot(xs, [r["false_per_trial"] for r in sel], ls=ls, marker=mk)
        axes[0, col].set_title(f"{dim}: median wall error (found walls)")
        axes[1, col].set_title(f"{dim}: false walls per map")
        axes[0, col].axhline(0.3, color="gray", ls="--", lw=1)
        for r_ in (0, 1):
            axes[r_, col].set_xscale("log")
        axes[1, col].set_xlabel("client positions fused")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fusion.png", dpi=150)

    for dim in DIMS:
        print(dim)
        for bw, est, mode in itertools.product(BWS, ESTS, ["known", "ftm", "joint"]):
            cells = []
            for n in NS:
                r = next((r for r in table if r["bw"] == bw and r["est"] == est and r["dim"] == dim and r["d0"] == mode and r["n"] == n and r["check"] == (n > 1)), None)
                if r is None:
                    cells.append("   -   ")
                    continue
                med = "-" if r["dist_med"] is None else f"{r['dist_med']:.2f}"
                cells.append(f"{med}/{r['detect']:.2f}/{r['false_per_trial']:.1f}")
            print(f"  {bw} {est:6s} {mode:6s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
