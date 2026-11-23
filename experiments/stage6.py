"""Stage 6: real wall materials, a fixed noise floor and a radio map.

Part 1 reruns stage 5 with walls from ITU-R P.2040 (concrete outside,
brick and drywall inside) and a fixed noise floor: a 1 m free-space link
has 51 dB SNR at 160 MHz and 48 dB at 320 MHz (15 dBm phone, 7 dB noise
figure). Walls and distance now lower the SNR of a link. Stage 5 kept
30 dB on every link.

Part 2 predicts the received power over the apartment from the walls and
compares it with the true power:
- free space: no walls
- true walls: true wall positions, 5 dB per wall (TGax residential)
- <est> walls: the closed walls from close_rooms, 5 dB per wall

Plan B-rc is plan B with 10 cm concrete in place of the brick walls.

Plan B and walk seeds 7000+ are held out. Plan A rows are not.

    uv run python stage6.py
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
from wallmap.floorplan import MATERIALS_A, MATERIALS_B, MATERIALS_B_RC, apartment, apartment_b, line_id, walk
from wallmap.fusion import Link, fuse_walk
from wallmap.geometry import Array, image_paths, wall_line
from wallmap.radiomap import free_space, grid, plan_from_segments, rss_dbm
from wallmap.rooms import close_rooms, score_rooms

OUT = out_dir("stage6")
AP = np.array([2.5, 0.1])
ANTS = 4
SNR_1M = {160: 51.0, 320: 48.0}
ESTS = ["music", "nomp"]
BWS = [160, 320]
PLANS = {
    "A": (apartment().with_materials(MATERIALS_A), [0]),
    "B": (apartment_b().with_materials(MATERIALS_B), [7000, 7001, 7002]),
    "B-rc": (apartment_b().with_materials(MATERIALS_B_RC), [7000, 7001, 7002]),
}


def link(args):
    plan_id, seed, i, est, bw = args
    plan, _ = PLANS[plan_id]
    pos = walk(seed=seed, waypoints=plan.waypoints)
    ap = Array(AP, n_ant=ANTS)
    paths = image_paths(plan, ap, pos[i], max_order=2)
    los = [q for q in paths if q.is_los][0]
    y, n_k, s2 = measure(paths, ap, bw, 0.0, np.random.default_rng(seed * 1000 + i + 1), snr_1m_db=SNR_1M[bw])
    p = ESTIMATORS[est](y, n_k, s2, ap.fc)
    rel = relative_params(p)
    if rel is None:
        rel = (np.pi / 2, np.zeros(0), np.zeros(0))
    phi_true = float(np.arccos(np.clip(los.u[0], -1, 1)))
    snr = 10 * np.log10(max(float(np.mean(np.abs(y) ** 2)) / s2 - 1, 1e-3))
    return {
        "link": Link(rel[0], los.length, rel[1], rel[2]),
        "wrong_los": bool(abs(rel[0] - phi_true) > np.deg2rad(5)),
        "snr": float(snr),
        "paths": p,
        "los_gain": complex(los.gain),
    }


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


def rss_chunk(args):
    kind, key, pts = args
    ap = Array(AP, n_ant=ANTS)
    return rss_dbm(ROOMS[kind][key], ap, pts)


ROOMS: dict = {}


def load_links(jobs):
    """Links take minutes: cache them by job."""
    raw = OUT.parent / "raw" / "stage6_links.pkl"
    cache = pickle.loads(raw.read_bytes()) if raw.exists() else {}
    if not isinstance(cache, dict):
        cache = {}
    todo = [j for j in jobs if j not in cache]
    if todo:
        cache.update(zip(todo, pool_map(link, todo)))
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(pickle.dumps(cache))
    return [cache[j] for j in jobs]


def main():
    jobs = []
    for pid, (plan, seeds) in PLANS.items():
        for seed in seeds:
            n = len(walk(seed=seed, waypoints=plan.waypoints))
            jobs += [(pid, seed, i, est, bw) for est, bw in itertools.product(ESTS, BWS) for i in range(n)]
    res = load_links(jobs)
    by: dict[tuple, list] = {}
    for (pid, seed, i, est, bw), r in zip(jobs, res):
        by.setdefault((pid, seed, est, bw), []).append(r)

    rows, closed = [], {}
    for pid, (plan, seeds) in PLANS.items():
        for seed in seeds:
            pos = walk(seed=seed, waypoints=plan.waypoints)
            for est, bw in itertools.product(ESTS, BWS):
                rs = by[(pid, seed, est, bw)]
                lks = [r["link"] for r in rs]
                fw = fuse_walk(AP, lks, [lk.d0 for lk in lks], pos)
                rm = close_rooms(fw, pos)
                rows.append({
                    "plan": pid, "seed": seed, "cond": f"{est} {bw}",
                    **score_rooms(rm, plan.boxes, plan.doors), **wall_score(plan, fw),
                    "wrong_los": float(np.mean([r["wrong_los"] for r in rs])),
                    "snr_p10": float(np.percentile([r["snr"] for r in rs], 10)),
                    "snr_p50": float(np.median([r["snr"] for r in rs])),
                })
                closed[(pid, seed, est, bw)] = rm.closed
    save_json(OUT / "stage6.json", rows)
    print(f"{'plan':4s} {'cond':10s} {'IoU':>5s} {'rooms':>7s} {'walls':>9s} {'wrongLoS':>8s} {'SNR p10/p50':>12s}")
    for pid in PLANS:
        for cond in dict.fromkeys(r["cond"] for r in rows if r["plan"] == pid):
            rs = [r for r in rows if r["plan"] == pid and r["cond"] == cond]
            m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
            print(f"{pid:4s} {cond:10s} {m('mean_iou'):5.2f} {m('rooms_ok'):4.1f}/5 {m('walls_found'):4.1f}/7 {m('walls_false'):4.1f}F {m('wrong_los'):8.2f} {m('snr_p10'):5.1f}/{m('snr_p50'):5.1f}")

    # part 2: radio map on plan B, walls found by MUSIC and NOMP at 160 MHz
    rmap = []
    for pid in ("B", "B-rc"):
        plan, seeds = PLANS[pid]
        pts = grid(plan.width, plan.depth)
        x0, y0, x1, y1 = plan.boxes[0][1:]
        same = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
        ROOMS.clear()
        ROOMS["truth"] = {0: plan}
        ROOMS["free"] = {0: free_space(plan.width, plan.depth)}
        ROOMS["true walls"] = {0: plan_from_segments([_seg(w) for w in plan.walls()], plan.width, plan.depth)}
        for est in ESTS:
            ROOMS[f"{est} walls"] = {s: plan_from_segments(closed[(pid, s, est, 160)], plan.width, plan.depth) for s in seeds}
        chunks = np.array_split(np.arange(len(pts)), 32)
        maps = {}
        for kind, d in ROOMS.items():
            for key in d:
                maps[(kind, key)] = np.concatenate(pool_map(rss_chunk, [(kind, key, pts[c]) for c in chunks]))
        truth = maps[("truth", 0)]
        for kind in [k for k in ROOMS if k != "truth"]:
            e = np.stack([maps[(kind, k)] - truth for k in ROOMS[kind]])
            row = {"plan": pid, "pred": kind}
            for area, m in (("same room", same), ("other rooms", ~same)):
                a = np.abs(e[:, m])
                row[area] = {"median_abs_db": float(np.median(a)), "p90_abs_db": float(np.percentile(a, 90)), "bias_db": float(np.mean(e[:, m]))}
            rmap.append(row)
            o = row["other rooms"]
            print(f"radio map {pid:4s} {kind:12s}: other rooms median {o['median_abs_db']:4.1f} dB, p90 {o['p90_abs_db']:4.1f}, bias {o['bias_db']:+5.1f}; same room median {row['same room']['median_abs_db']:4.1f}")
        plot_maps(pid, plan, pts, maps, closed[(pid, seeds[0], "nomp", 160)])
    save_json(OUT / "radiomap.json", rmap)


def _seg(w):
    from wallmap.rooms import Segment

    return Segment(w.p0, w.p1)


def plot_maps(pid, plan, pts, maps, found):
    xs, ys = np.unique(pts[:, 0]), np.unique(pts[:, 1])
    panels = [("truth", 0, "true power"), ("free", 0, "free space"), ("true walls", 0, "true walls, 5 dB"), ("nomp walls", 7000, "NOMP walls, 5 dB")]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (kind, key, title) in zip(axs, panels):
        z = maps[(kind, key)].reshape(len(ys), len(xs))
        im = ax.imshow(z, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], vmin=-75, vmax=-25, cmap="viridis")
        for w in plan.walls():
            ax.plot([w.p0[0], w.p1[0]], [w.p0[1], w.p1[1]], color="w", lw=1.5)
        if kind == "nomp walls":
            for s in found:
                ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], color="r", lw=1)
        ax.plot(*AP, "ws", ms=4)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.8, label="dBm")
    fig.savefig(OUT / f"radiomap_{pid}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
