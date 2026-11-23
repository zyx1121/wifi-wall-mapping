"""Stage 10: after one walk, where should the AP go?

The AP starts at the site of one room (stage 9). The phone walks the
house once. From the walk we fuse the walls, close the rooms and fit the
loss of each wall (stages 5 to 7). With this map we predict the received
power over the house for every candidate AP position, and pick the one
with the best 10th percentile (the weakest spots). The true plan then
scores the pick.

3 ways to pick:
- map: the walls and losses from the walk
- free space: no walls, the distance only
- truth: the true plan, the best possible pick

Plan B, NOMP at 160 MHz, P.2040 walls, fixed noise floor. Candidates on a
0.5 m grid, at least 0.2 m from every wall. We score the house on a
second 0.5 m grid, shifted by 0.25 m.

    uv run python stage10.py
"""

from __future__ import annotations

import pickle

import common  # noqa: F401  (BLAS threads)
import numpy as np

from common import ROOT, out_dir, pool_map, save_json
from wallmap.attenuation import crossings, excess_db, false_lines, fit_losses
from wallmap.csi import measure
from wallmap.estimators import ESTIMATORS, relative_params
from wallmap.floorplan import MATERIALS_B, apartment_b, walk
from wallmap.fusion import Link, fuse_walk
from wallmap.geometry import Array, image_paths
from wallmap.multi_ap import ap_sites, frame, local_plan
from wallmap.radiomap import free_space, plan_from_segments, rss_dbm
from wallmap.rooms import Segment, close_rooms, score_rooms

OUT = out_dir("stage10")
CACHE = ROOT / "results" / "raw" / "stage10_links.pkl"
ANTS = 4
BW = 160
SNR_1M = 51.0
SEEDS = [7000, 7001, 7002]
STEP = 0.5
PLAN = apartment_b().with_materials(MATERIALS_B)
SITES = ap_sites(PLAN)


def inside(step: float, clear: float, start: float) -> np.ndarray:
    xs = np.arange(start, PLAN.width, step)
    ys = np.arange(start, PLAN.depth, step)
    pts = np.array([(x, y) for y in ys for x in xs])
    ok = np.ones(len(pts), bool)
    for w in PLAN.walls():
        t = (w.p1 - w.p0) / np.linalg.norm(w.p1 - w.p0)
        n = np.array([-t[1], t[0]])
        s = (pts - w.p0) @ t
        ok &= ~((np.abs((pts - w.p0) @ n) < clear) & (s > -clear) & (s < np.linalg.norm(w.p1 - w.p0) + clear))
    return pts[ok]


CANDS = inside(STEP, 0.2, STEP / 2)
EVAL = inside(STEP, 0.1, STEP)  # off the candidate grid, so no point sits on an AP


def link(args):
    """One link from the AP of site k, in its frame. Same noise seeds as stage 9."""
    k, seed, i = args
    site = SITES[k]
    f = frame(PLAN, site)
    ap = Array(f.to_local(site.pos), n_ant=ANTS)
    client = f.to_local(walk(seed=seed, waypoints=PLAN.waypoints)[i])
    paths = image_paths(local_plan(PLAN, f), ap, client, max_order=2)
    los = [q for q in paths if q.is_los][0]
    y, n_k, s2 = measure(paths, ap, BW, 0.0, np.random.default_rng(seed * 1000 + i + 1 + 100_000 * k), snr_1m_db=SNR_1M)
    p = ESTIMATORS["nomp"](y, n_k, s2, ap.fc)
    rel = relative_params(p) or (np.pi / 2, np.zeros(0), np.zeros(0))
    return Link(rel[0], los.length, rel[1], rel[2]), float(excess_db([p], [los.length], ap.wavelength)[0])


def load_links(jobs):
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    todo = [j for j in jobs if j not in cache]
    if todo:
        for j, r in zip(todo, pool_map(link, todo)):
            cache[j] = r
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(cache))
    return [cache[j] for j in jobs]


def build_map(args):
    """Walls and losses from one walk with the AP of site k (stages 5 to 7)."""
    k, seed, res = args
    site = SITES[k]
    f = frame(PLAN, site)
    pos = walk(seed=seed, waypoints=PLAN.waypoints)
    links = [r[0] for r in res]
    exc = np.array([r[1] for r in res])
    fw = fuse_walk(f.to_local(site.pos), links, [lk.d0 for lk in links], f.to_local(pos))
    segs = [Segment(f.to_global(w["p0"]), f.to_global(w["p1"]), int(w["size"])) for w in fw]
    if not segs:
        return free_space(PLAN.width, PLAN.depth), 0
    rm = close_rooms(segs, pos)
    if rm.closed:
        _, loss, count = fit_losses(exc, crossings(site.pos, pos, rm.closed, len(rm.walls)))
        drop = set(false_lines(loss, count))
        keep = [s for i, s in enumerate(rm.walls) if i not in drop]
        if drop and keep:
            rm = close_rooms(keep, pos)
    losses = {}
    if rm.closed:
        _, loss, _ = fit_losses(exc, crossings(site.pos, pos, rm.closed, len(rm.walls)))
        losses = {f"w{i}": float(loss[s.line]) for i, s in enumerate(rm.closed)}
    rooms = score_rooms(rm, PLAN.boxes, PLAN.doors)["rooms_ok"]
    return plan_from_segments(rm.closed, PLAN.width, PLAN.depth, losses=losses), rooms


def p10(args):
    """10th percentile of the received power over the house, for each candidate AP."""
    room, order, cands = args
    return [float(np.percentile(rss_dbm(room, Array(c, n_ant=ANTS), EVAL, max_order=order), 10)) for c in cands]


def score_all(room, order) -> np.ndarray:
    chunks = np.array_split(CANDS, 32)
    return np.concatenate(pool_map(p10, [(room, order, c) for c in chunks]))


def main():
    jobs = [(k, seed, i) for k in range(len(SITES)) for seed in SEEDS for i in range(len(walk(seed=seed, waypoints=PLAN.waypoints)))]
    res = load_links(jobs)
    by: dict[tuple, list] = {}
    for (k, seed, i), r in zip(jobs, res):
        by.setdefault((k, seed), []).append(r)
    keys = list(by)
    maps = pool_map(build_map, [(k, seed, by[(k, seed)]) for k, seed in keys])

    true_p10 = score_all(PLAN, 2)
    free_pick = int(np.argmax(score_all(free_space(PLAN.width, PLAN.depth), 1)))
    best = float(true_p10.max())
    site_p10 = [float(np.percentile(rss_dbm(PLAN, Array(s.pos, n_ant=ANTS), EVAL, max_order=2), 10)) for s in SITES]
    rows = []
    for (k, seed), (room, rooms) in zip(keys, maps):
        pick = int(np.argmax(score_all(room, 1)))
        rows.append({
            "site": SITES[k].room + str(k), "seed": seed, "rooms_ok": rooms,
            "pick": CANDS[pick].tolist(), "now_db": site_p10[k],
            "map_db": float(true_p10[pick]), "free_db": float(true_p10[free_pick]), "best_db": best,
        })
    save_json(OUT / "stage10.json", {"rows": rows, "best_pos": CANDS[int(np.argmax(true_p10))].tolist(), "free_pos": CANDS[free_pick].tolist(), "candidates": len(CANDS)})

    print(f"{len(CANDS)} candidates, true best {best:.1f} dBm at {CANDS[int(np.argmax(true_p10))]}, free space picks {CANDS[free_pick]} ({true_p10[free_pick]:.1f} dBm)")
    print(f"{'AP now':10s} {'rooms':>5s} {'p10 now':>8s} {'map pick':>9s} {'free pick':>9s} {'best':>6s}   picks")
    for k in range(len(SITES)):
        rs = [r for r in rows if r["site"] == SITES[k].room + str(k)]
        m = lambda key: float(np.mean([r[key] for r in rs]))  # noqa: E731
        picks = " ".join(f"({r['pick'][0]:.2f},{r['pick'][1]:.2f})" for r in rs)
        print(f"{SITES[k].room + str(k):10s} {m('rooms_ok'):5.1f} {m('now_db'):8.1f} {m('map_db'):9.1f} {m('free_db'):9.1f} {best:6.1f}   {picks}")


if __name__ == "__main__":
    main()
