"""Stage 5: close fused walls into rooms.

Plan A is the demo apartment. fuse_walk was tuned on its walk (seed 0), so
plan A rows are not held out. Plan B (apartment_b) and walk seeds 7000+
are held out: no parameter was tuned on them. Report plan B.

Conditions per walk:
- walk: no walls, rooms from the walk alone (baseline)
- truth: the true wall segments (upper bound of the closing step)
- <est> <bw>: walls from fuse_walk, d0 known, 4 antennas, 30 dB, ideal hardware

    uv run python stage5.py
"""

from __future__ import annotations

import itertools
import pickle

import common  # noqa: F401  (BLAS threads)
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import out_dir, pool_map, save_json
from wallmap.csi import measure
from wallmap.estimators import ESTIMATORS, relative_params
from wallmap.floorplan import apartment, apartment_b, line_id, walk
from wallmap.fusion import Link, fuse_walk
from wallmap.geometry import Array, image_paths, wall_line
from wallmap.rooms import Segment, close_rooms, score_rooms

OUT = out_dir("stage5")
AP = np.array([2.5, 0.1])
ANTS = 4
SNR = 30.0
ESTS = ["music", "nomp"]
BWS = [160, 320]
PLANS = {"A": (apartment(), [0]), "B": (apartment_b(), [7000, 7001, 7002])}


def link(args):
    plan_id, seed, i, est, bw = args
    plan, _ = PLANS[plan_id]
    pos = walk(seed=seed, waypoints=plan.waypoints)
    ap = Array(AP, n_ant=ANTS)
    paths = image_paths(plan, ap, pos[i], max_order=2)
    los = [q for q in paths if q.is_los][0]
    # plan A seed 0 keeps the demo noise seeds
    y, n_k, s2 = measure(paths, ap, bw, SNR, np.random.default_rng(i if seed == 0 else seed * 1000 + i))
    rel = relative_params(ESTIMATORS[est](y, n_k, s2, ap.fc))
    if rel is None:
        rel = (np.pi / 2, np.zeros(0), np.zeros(0))
    return Link(rel[0], los.length, rel[1], rel[2])


def wall_score(plan, fused):
    targets = {}
    for w in plan.walls():
        targets.setdefault(line_id(w.name), wall_line(AP, w))
    hit, false = set(), 0
    for f in fused:
        lab = None
        for lid, (tt, tr) in targets.items():
            if abs((f["theta"] - tt + np.pi) % (2 * np.pi) - np.pi) < np.deg2rad(10) and abs(f["rho"] - tr) < 0.5:
                lab = lid
        if lab is None:
            false += 1
        elif lab != "south":
            hit.add(lab)
    return {"walls_found": len(hit), "walls_true": len(targets) - 1, "walls_false": false}


def main():
    jobs = []
    for pid, (plan, seeds) in PLANS.items():
        for seed in seeds:
            n = len(walk(seed=seed, waypoints=plan.waypoints))
            jobs += [(pid, seed, i, est, bw) for est, bw in itertools.product(ESTS, BWS) for i in range(n)]
    # the links take minutes, the room step takes seconds: cache the links
    raw = OUT.parent / "raw" / "stage5_links.pkl"
    if raw.exists():
        links = pickle.loads(raw.read_bytes())
    else:
        links = pool_map(link, jobs)
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(pickle.dumps(links))
    by: dict[tuple, list] = {}
    for (pid, seed, i, est, bw), lk in zip(jobs, links):
        by.setdefault((pid, seed, est, bw), []).append(lk)

    rows, maps = [], {}
    for pid, (plan, seeds) in PLANS.items():
        truth = [Segment(w.p0, w.p1, 10) for w in plan.walls()]
        for seed in seeds:
            pos = walk(seed=seed, waypoints=plan.waypoints)
            conds = {"walk": [], "truth": truth}
            for est, bw in itertools.product(ESTS, BWS):
                lks = by[(pid, seed, est, bw)]
                conds[f"{est} {bw}"] = fuse_walk(AP, lks, [lk.d0 for lk in lks], pos)
            for name, walls in conds.items():
                rm = close_rooms(walls, pos)
                row = {"plan": pid, "seed": seed, "cond": name, **score_rooms(rm, plan.boxes, plan.doors)}
                if name not in ("walk", "truth"):
                    row.update(wall_score(plan, walls))
                rows.append(row)
                if seed == seeds[0]:
                    maps[(pid, name)] = (rm, pos)

    save_json(OUT / "stage5.json", rows)
    print(f"{'plan':4s} {'cond':10s} {'IoU':>5s} {'rooms':>7s} {'found':>5s} {'doors R/P':>10s} {'walls':>8s}")
    for pid in PLANS:
        for cond in dict.fromkeys(r["cond"] for r in rows if r["plan"] == pid):
            rs = [r for r in rows if r["plan"] == pid and r["cond"] == cond]
            m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
            walls = f"{m('walls_found'):.1f}/{rs[0]['walls_true']} {m('walls_false'):.1f}F" if "walls_found" in rs[0] else ""
            print(f"{pid:4s} {cond:10s} {m('mean_iou'):5.2f} {m('rooms_ok'):4.1f}/{rs[0]['rooms_true']} {m('rooms_found'):5.1f} {m('door_recall'):4.2f}/{m('door_precision'):4.2f} {walls:>8s}")
    plot(maps)


def plot(maps):
    for pid, (plan, _) in PLANS.items():
        conds = [c for (p, c) in maps if p == pid]
        fig, axs = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 3.2))
        for ax, cond in zip(axs, conds):
            rm, pos = maps[(pid, cond)]
            lab = np.ma.masked_equal(rm.labels.T, 0)
            ext = [rm.x0, rm.x0 + rm.labels.shape[0] * rm.res, rm.y0, rm.y0 + rm.labels.shape[1] * rm.res]
            ax.imshow(lab, origin="lower", extent=ext, cmap="tab10", alpha=0.45, interpolation="nearest")
            for w in plan.walls():
                ax.plot([w.p0[0], w.p1[0]], [w.p0[1], w.p1[1]], color="0.75", lw=3, zorder=1)
            for s in rm.walls:
                ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], color="k", lw=1.2)
            for s in rm.removed:
                ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], color="r", lw=1.2, ls="--")
            for d in rm.doors:
                ax.plot(*d, "o", ms=4, color="tab:green")
            ax.plot(pos[:, 0], pos[:, 1], color="0.4", lw=0.5)
            ax.plot(*AP, "ks", ms=4)
            ax.set_title(cond, fontsize=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / f"rooms_{pid}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
