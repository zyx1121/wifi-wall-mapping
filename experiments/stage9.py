"""Stage 9: 1, 2 or 3 APs, in different rooms.

One AP sees a wall only where its mirror reflection lands on the wall.
On plan B, no position of the walk has a first-order reflection off the
wall between bedroom A and the bathroom. More APs see more walls.

Each room has one AP site: the middle of its longest outer wall
(wallmap.multi_ap). Every AP measures the same walk. Each AP fuses its own
links in its own frame. The rooms close over the walls of all APs.
Settings as in stage 6: P.2040 walls, fixed noise floor, 160 MHz, 4
antennas, known LoS length.

2 modes:
- nomp: NOMP on simulated CSI
- oracle: the true paths within 25 dB of the strongest path, no estimator error

We set nothing on plan B. Plan A is in the table too.

    uv run python stage9.py
"""

from __future__ import annotations

import itertools
import pickle

import common  # noqa: F401  (BLAS threads)
import numpy as np

from common import ROOT, out_dir, pool_map, save_json
from wallmap.csi import measure
from wallmap.estimators import ESTIMATORS, relative_params
from wallmap.floorplan import MATERIALS_A, MATERIALS_B, apartment, apartment_b, line_id, walk
from wallmap.fusion import Link, fuse_walk
from wallmap.geometry import Array, image_paths
from wallmap.multi_ap import ap_sites, frame, label_segment, local_plan
from wallmap.rooms import Segment, close_rooms, score_rooms

OUT = out_dir("stage9")
CACHE = ROOT / "results" / "raw" / "stage9_links.pkl"
ANTS = 4
BW = 160
SNR_1M = 51.0
ORACLE_DB = -25.0
MODES = ["nomp", "oracle"]
PLANS = {
    "A": (apartment().with_materials(MATERIALS_A), [0]),
    "B": (apartment_b().with_materials(MATERIALS_B), [7000, 7001, 7002]),
}
SITES = {pid: ap_sites(plan) for pid, (plan, _) in PLANS.items()}


def interior(plan) -> list[str]:
    outer = {"south", "east", "north", "west"}
    return sorted({line_id(w.name) for w in plan.walls()} - outer)


def link(args):
    pid, seed, k, i, mode = args
    plan, _ = PLANS[pid]
    site = SITES[pid][k]
    f = frame(plan, site)
    lp = local_plan(plan, f)
    ap = Array(f.to_local(site.pos), n_ant=ANTS)
    client = f.to_local(walk(seed=seed, waypoints=plan.waypoints)[i])
    paths = image_paths(lp, ap, client, max_order=2)
    los = [q for q in paths if q.is_los][0]

    def phi(q):
        return float(np.arccos(np.clip(q.u[0], -1, 1)))

    if mode == "oracle":
        g = max(abs(q.gain) for q in paths)
        keep = [q for q in paths if not q.is_los and 20 * np.log10(abs(q.gain) / g) > ORACLE_DB]
        return Link(phi(los), los.length, np.array([phi(q) for q in keep]), np.array([q.length - los.length for q in keep]))
    y, n_k, s2 = measure(paths, ap, BW, 0.0, np.random.default_rng(seed * 1000 + i + 1 + 100_000 * k), snr_1m_db=SNR_1M)
    rel = relative_params(ESTIMATORS["nomp"](y, n_k, s2, ap.fc))
    if rel is None:
        rel = (np.pi / 2, np.zeros(0), np.zeros(0))
    return Link(rel[0], los.length, rel[1], rel[2])


def fuse(args):
    """Walls that one AP finds, in the global frame."""
    pid, seed, k, links = args
    plan, _ = PLANS[pid]
    site = SITES[pid][k]
    f = frame(plan, site)
    pos = f.to_local(walk(seed=seed, waypoints=plan.waypoints))
    fw = fuse_walk(f.to_local(site.pos), links, [lk.d0 for lk in links], pos)
    return [Segment(f.to_global(w["p0"]), f.to_global(w["p1"]), int(w["size"])) for w in fw]


def score(args):
    pid, seed, combo, segs = args
    plan, _ = PLANS[pid]
    pos = walk(seed=seed, waypoints=plan.waypoints)
    labels = [label_segment(plan, s.a, s.b) for s in segs]
    inner = interior(plan)
    rm = close_rooms(segs, pos) if segs else None
    sc = score_rooms(rm, plan.boxes, plan.doors) if rm else {"rooms_ok": 0, "mean_iou": 0.0}
    return {
        "walls_found": len({lab for lab in labels if lab in inner}),
        "walls_true": len(inner),
        "walls_false": sum(lab is None for lab in labels),
        "rooms_ok": sc["rooms_ok"],
        "mean_iou": float(sc["mean_iou"]),
    }


def load_links(jobs):
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    todo = [j for j in jobs if j not in cache]
    if todo:
        for j, r in zip(todo, pool_map(link, todo)):
            cache[j] = r
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(cache))
    return [cache[j] for j in jobs]


def main():
    jobs = []
    for pid, (plan, seeds) in PLANS.items():
        for seed in seeds:
            n = len(walk(seed=seed, waypoints=plan.waypoints))
            jobs += [(pid, seed, k, i, mode) for k in range(len(SITES[pid])) for i in range(n) for mode in MODES]
    by: dict[tuple, list] = {}
    for (pid, seed, k, i, mode), lk in zip(jobs, load_links(jobs)):
        by.setdefault((pid, seed, k, mode), []).append(lk)
    keys = list(by)
    walls = dict(zip(keys, pool_map(fuse, [(pid, seed, k, by[(pid, seed, k, mode)]) for pid, seed, k, mode in keys])))

    runs = []
    for pid, (plan, seeds) in PLANS.items():
        n_sites = len(SITES[pid])
        for n_ap in (1, 2, 3):
            for combo in itertools.combinations(range(n_sites), n_ap):
                for seed in seeds:
                    for mode in MODES:
                        segs = [s for k in combo for s in walls[(pid, seed, k, mode)]]
                        runs.append(((pid, seed, combo, mode), (pid, seed, combo, segs)))
    scores = pool_map(score, [r[1] for r in runs])
    rows = []
    for ((pid, seed, combo, mode), _), sc in zip(runs, scores):
        rooms = [SITES[pid][k].room + str(k) for k in combo]
        rows.append({"plan": pid, "seed": seed, "aps": rooms, "n_ap": len(combo), "mode": mode, **sc})
    save_json(OUT / "stage9.json", rows)

    for pid in PLANS:
        print(f"plan {pid}: AP sites " + ", ".join(f"{k} {s.room} ({s.pos[0]:.1f}, {s.pos[1]:.1f})" for k, s in enumerate(SITES[pid])))
        print(f"{'APs':24s} {'mode':6s} {'walls':>6s} {'false':>5s} {'rooms':>6s} {'IoU':>5s}")
        groups = {}
        for r in rows:
            if r["plan"] == pid:
                groups.setdefault((tuple(r["aps"]), r["mode"]), []).append(r)
        for (aps, mode), rs in sorted(groups.items(), key=lambda g: (len(g[0][0]), g[0][1], -np.mean([r["rooms_ok"] for r in g[1]]), -np.mean([r["mean_iou"] for r in g[1]]))):
            m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
            print(f"{' + '.join(aps):24s} {mode:6s} {m('walls_found'):3.1f}/{rs[0]['walls_true']} {m('walls_false'):5.1f} {m('rooms_ok'):4.1f}/5 {m('mean_iou'):5.2f}")


if __name__ == "__main__":
    main()
