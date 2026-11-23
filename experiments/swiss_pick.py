"""Pick apartments from Swiss Dwellings and draw them.

Reads geometries.csv of Swiss Dwellings v3.0.0 (CC BY 4.0,
https://zenodo.org/records/7788422). Set SWISS_CSV to its path. The
default is ../data/swiss-dwellings/geometries.csv next to the repo.

Takes N apartments at random among those on 1 floor with a living room,
a kitchen, a bathroom and 1 or more bedrooms, and keeps the ones with 2 or
more AP sites. Seed 0 is the test set (results/raw/swiss_rows.pkl), which
stage 11 reports on. Seed 1 is a separate dev set
(results/raw/swiss_rows_dev.pkl), the only set used to choose settings.
Drawings go to results/swiss/.

    uv run python swiss_pick.py 12 0
    uv run python swiss_pick.py 12 1
"""

from __future__ import annotations

import os
import pickle
import sys

import common  # noqa: F401  (BLAS threads)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import ROOT, out_dir  # noqa: E402
from wallmap.floorplan import walk  # noqa: E402
from wallmap.swiss import ap_sites, build, candidates, read, waypoints  # noqa: E402

CSV = os.environ.get("SWISS_CSV", str(ROOT.parent / "data" / "swiss-dwellings" / "geometries.csv"))
OUT = out_dir("swiss")
RAW = {0: ROOT / "results" / "raw" / "swiss_rows.pkl", 1: ROOT / "results" / "raw" / "swiss_rows_dev.pkl"}


def draw(aid, apt, pos, sites, path):
    p = apt.plan
    fig, ax = plt.subplots(figsize=(7, 7 * p.depth / max(p.width, 1)))
    for _, g in apt.rooms:
        ax.fill(g[:, 0], g[:, 1], alpha=0.15)
    for name, a, b in p.segments:
        c = {"wall": "k", "window": "c", "door": "brown"}[apt.kind[name]]
        lw = max(1.0, apt.thick[name] * 40)
        ax.plot([a[0], b[0]], [a[1], b[1]], c, lw=lw, alpha=0.9 if name in apt.outer else 0.5, solid_capstyle="butt")
    for k, (_, (x, y)) in enumerate(p.rooms):
        ax.text(x, y, str(k), ha="center", fontsize=9)
    ax.plot(pos[:, 0], pos[:, 1], "-", c="tab:orange", lw=0.8)
    for s in sites:
        ax.plot(*s.pos, "s", c="r")
        ax.arrow(*s.pos, *(0.4 * s.normal), color="r", head_width=0.1)
    ax.set_aspect("equal")
    ax.set_title(f"{aid[:8]}  {p.width:.1f} x {p.depth:.1f} m, {len(pos)} positions, {len(sites)} AP sites")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ids = candidates(CSV)
    if seed != 0:  # never reuse a test apartment
        test = set(pickle.loads(RAW[0].read_bytes())) if RAW[0].exists() else set()
        ids = [i for i in ids if i not in test]
    rng = np.random.default_rng(seed)
    pick = [ids[i] for i in rng.permutation(len(ids))[: 3 * n]]
    rows = read(CSV, pick)
    keep = {}
    for aid in pick:
        apt = build(rows[aid])
        way = waypoints(apt)
        if len(way) < 2:
            continue
        pos = walk(waypoints=way, seed=0)
        sites = ap_sites(apt, pos)
        if len(sites) < 2:
            print(aid[:8], "skip:", len(sites), "AP sites")
            continue
        draw(aid, apt, pos, sites, OUT / f"s{seed}_{len(keep):02d}_{aid[:8]}.png")
        print(aid[:8], f"{apt.plan.width:.1f}x{apt.plan.depth:.1f} m, {len(apt.rooms)} areas, {len(apt.plan.boxes)} rooms, {len(apt.plan.segments)} segments, {len(pos)} positions, {len(sites)} sites")
        keep[aid] = rows[aid]
        if len(keep) == n:
            break
    print(f"{len(ids)} candidates, kept {len(keep)}")
    RAW[seed].parent.mkdir(parents=True, exist_ok=True)
    RAW[seed].write_bytes(pickle.dumps(keep))


if __name__ == "__main__":
    main()
