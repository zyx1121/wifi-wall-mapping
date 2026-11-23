"""Stage 11: stages 9 and 10 on 12 real apartments (Swiss Dwellings).

swiss_pick.py picks the apartments (seed 0, the test set). Each gets a
route through every room and 1 AP site per room that has a long outer wall
(wallmap.swiss). The dataset has no materials, so 2 rules give them:
"swiss" and "taiwan" (wallmap.swiss.STYLES). Settings as in stage 9: NOMP
at 160 MHz, P.2040 walls, fixed noise floor, 4 antennas, known LoS length.
"oracle" uses the true paths within 25 dB of the strongest one. "truth"
closes the rooms over the true walls: the best the room step can do.

The array cannot tell front from back. Each AP uses only the walk
positions in front of it (wallmap.swiss.in_front).

close_rooms cuts along whole wall lines (extend=None, as in stage 9). Real
plans have many short walls, so stage 11 also extends each wall by
EXTEND metres only. EXTEND comes from the dev set (swiss_pick.py seed 1),
not from these apartments.

Part 1 (stage 9): rooms from 1, 2 or 3 APs.
Part 2 (stage 10): from the walk of 1 AP, pick the AP position with the
best 10th percentile of received power. Candidates on a 1 m grid in the
rooms, the house scored on a 0.5 m grid. All radio maps here use first
order paths (stage 10 found the same best spot with order 1 and 2).

    uv run python swiss_pick.py 12 0
    uv run python stage11.py
"""

from __future__ import annotations

import itertools
import pickle

import common  # noqa: F401  (BLAS threads)
import numpy as np
from matplotlib.path import Path as MPath

from common import ROOT, out_dir, pool_map, save_json
from wallmap.attenuation import crossings, excess_db, false_lines, fit_losses
from wallmap.csi import measure
from wallmap.estimators import ESTIMATORS, relative_params
from wallmap.floorplan import line_id, walk
from wallmap.fusion import Link, fuse_walk
from wallmap.geometry import Array, image_paths
from wallmap.multi_ap import frame, label_segment, local_plan
from wallmap.radiomap import free_space, plan_from_segments, rss_dbm
from wallmap.rooms import Segment, close_rooms, score_rooms
from wallmap.swiss import STYLES, ap_sites, build, in_front, materials, waypoints

OUT = out_dir("stage11")
CACHE = ROOT / "results" / "raw" / "stage11_links.pkl"
ROWS = ROOT / "results" / "raw" / "swiss_rows.pkl"
ANTS = 4
BW = 160
SNR_1M = 51.0
ORACLE_DB = -25.0
MODES = ["nomp", "oracle"]
SEED = 0
EXTEND = 1.5  # metres, chosen on the dev set: true-wall rooms flat from 1.0 to 2.5
EXTENDS = {"lines": None, "1.5 m": EXTEND}


def _load():
    out = {}
    for aid, rows in pickle.loads(ROWS.read_bytes()).items():
        apt = build(rows)
        pos = walk(waypoints=waypoints(apt), seed=SEED)
        sites = ap_sites(apt, pos)
        out[aid] = (apt, pos, sites, [np.flatnonzero(in_front(s, pos)) for s in sites])
    return out


APTS = _load()


def plan_of(aid, style):
    apt = APTS[aid][0]
    return apt.plan.with_materials(materials(apt, style))


def link(args):
    aid, style, k, i, mode = args
    _, pos, sites, _ = APTS[aid]
    plan = plan_of(aid, style)
    site = sites[k]
    f = frame(plan, site)
    ap = Array(f.to_local(site.pos), n_ant=ANTS)
    paths = image_paths(local_plan(plan, f), ap, f.to_local(pos[i]), max_order=2)
    los = [q for q in paths if q.is_los][0]

    def phi(q):
        return float(np.arccos(np.clip(q.u[0], -1, 1)))

    if mode == "oracle":
        g = max(abs(q.gain) for q in paths)
        keep = [q for q in paths if not q.is_los and 20 * np.log10(abs(q.gain) / g) > ORACLE_DB]
        return Link(phi(los), los.length, np.array([phi(q) for q in keep]), np.array([q.length - los.length for q in keep])), float("nan")
    y, n_k, s2 = measure(paths, ap, BW, 0.0, np.random.default_rng(i + 1 + 100_000 * k), snr_1m_db=SNR_1M)
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


def fuse(args):
    aid, style, k, links = args
    _, pos, sites, front = APTS[aid]
    plan = plan_of(aid, style)
    f = frame(plan, sites[k])
    fw = fuse_walk(f.to_local(sites[k].pos), links, [lk.d0 for lk in links], f.to_local(pos[front[k]]))
    return [Segment(f.to_global(w["p0"]), f.to_global(w["p1"]), int(w["size"])) for w in fw]


def inner_lines(aid) -> set[str]:
    apt = APTS[aid][0]
    return {line_id(n) for n, *_ in apt.plan.segments if n not in apt.outer and apt.kind[n] == "wall"}


def score(args):
    aid, segs, extend = args
    apt, pos, _, _ = APTS[aid]
    inner = inner_lines(aid)
    labels = [label_segment(apt.plan, s.a, s.b) for s in segs]
    if segs:
        sc = score_rooms(close_rooms(segs, pos, extend=extend), apt.plan.boxes, apt.plan.doors)
    else:
        sc = {"rooms_ok": 0, "mean_iou": 0.0}
    return {
        "walls_found": len({lab for lab in labels if lab in inner}), "walls_true": len(inner),
        "walls_false": sum(lab is None for lab in labels),
        "rooms_ok": int(sc["rooms_ok"]), "rooms_true": len(apt.plan.boxes), "mean_iou": float(sc["mean_iou"]),
    }


# ---------------------------------------------------------------- part 2


def grids(aid):
    apt = APTS[aid][0]
    p = apt.plan

    def inside(step, start, clear):
        xs, ys = np.arange(start, p.width, step), np.arange(start, p.depth, step)
        pts = np.array([(x, y) for y in ys for x in xs])
        ok = np.zeros(len(pts), bool)
        for _, g in apt.rooms:
            ok |= MPath(g).contains_points(pts)
        for name, a, b in p.segments:
            a, b = np.array(a), np.array(b)
            t = (b - a) / np.linalg.norm(b - a)
            s = (pts - a) @ t
            h = np.abs((pts - a) @ np.array([-t[1], t[0]]))
            ok &= ~((h < apt.thick[name] / 2 + clear) & (s > -clear) & (s < np.linalg.norm(b - a) + clear))
        return pts[ok]

    return inside(1.0, 0.5, 0.2), inside(0.5, 0.25, 0.1)


def p10_all(args):
    room, cands, evals = args
    return [float(np.percentile(rss_dbm(room, Array(c, n_ant=ANTS), evals, max_order=1), 10)) for c in cands]


def build_map(args):
    """Walls and losses from the walk of 1 AP (stages 5 to 7)."""
    aid, style, k, res = args
    apt, pos, sites, front = APTS[aid]
    site = sites[k]
    pos = pos[front[k]]
    plan = plan_of(aid, style)
    f = frame(plan, site)
    links = [r[0] for r in res]
    exc = np.array([r[1] for r in res])
    fw = fuse_walk(f.to_local(site.pos), links, [lk.d0 for lk in links], f.to_local(pos))
    segs = [Segment(f.to_global(w["p0"]), f.to_global(w["p1"]), int(w["size"])) for w in fw]
    if not segs:
        return free_space(plan.width, plan.depth)
    rm = close_rooms(segs, pos, extend=EXTEND)
    if rm.closed:
        _, loss, count = fit_losses(exc, crossings(site.pos, pos, rm.closed, len(rm.walls)))
        drop = set(false_lines(loss, count))
        keep = [s for i, s in enumerate(rm.walls) if i not in drop]
        if drop and keep:
            rm = close_rooms(keep, pos, extend=EXTEND)
    losses = {}
    if rm.closed:
        _, loss, _ = fit_losses(exc, crossings(site.pos, pos, rm.closed, len(rm.walls)))
        losses = {f"w{i}": float(loss[s.line]) for i, s in enumerate(rm.closed)}
    return plan_from_segments(rm.closed, plan.width, plan.depth, losses=losses)


def per_apartment(rows, key):
    """Mean over the AP sets of each apartment, then over apartments, so each apartment counts once."""
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["apt"], []).append(r[key])
    return float(np.mean([np.mean(v) for v in by.values()]))


def main():
    jobs = []
    for aid, (apt, pos, sites, front) in APTS.items():
        for style in STYLES:
            jobs += [(aid, style, k, int(i), mode) for k in range(len(sites)) for i in front[k] for mode in MODES]
    n_all = sum(len(pos) * len(sites) for _, pos, sites, _ in APTS.values())
    n_front = sum(len(f) for *_, front in APTS.values() for f in front)
    print(f"{len(APTS)} apartments, {sum(len(v[2]) for v in APTS.values())} AP sites, {len(jobs)} links. Each AP drops the {1 - n_front / n_all:.0%} of positions behind it")
    by: dict[tuple, list] = {}
    for (aid, style, k, i, mode), r in zip(jobs, load_links(jobs)):
        by.setdefault((aid, style, k, mode), []).append(r)
    keys = list(by)
    walls = dict(zip(keys, pool_map(fuse, [(aid, style, k, [r[0] for r in by[(aid, style, k, mode)]]) for aid, style, k, mode in keys])))

    # part 1
    runs = []
    for aid, (apt, pos, sites, _) in APTS.items():
        true_segs = [Segment(np.array(a), np.array(b)) for _, a, b in apt.plan.segments]
        for ext_name, ext in EXTENDS.items():
            runs.append(((aid, "-", "truth", ext_name, ()), (aid, true_segs, ext)))
            for style in STYLES:
                for mode in MODES:
                    for n_ap in (1, 2, 3):
                        for combo in itertools.combinations(range(len(sites)), n_ap):
                            segs = [s for k in combo for s in walls[(aid, style, k, mode)]]
                            runs.append(((aid, style, mode, ext_name, combo), (aid, segs, ext)))
    scores = pool_map(score, [r[1] for r in runs])
    rows = [{"apt": aid, "style": style, "mode": mode, "extend": ext_name, "aps": [APTS[aid][2][k].room for k in combo], "n_ap": len(combo), **sc}
            for ((aid, style, mode, ext_name, combo), _), sc in zip(runs, scores)]
    save_json(OUT / "rooms.json", rows)

    # part 2
    mk = [(aid, style, k) for aid, style, k, mode in keys if mode == "nomp"]
    maps = pool_map(build_map, [(aid, style, k, by[(aid, style, k, "nomp")]) for aid, style, k in mk])
    picks = []
    for aid in APTS:
        apt, _, sites, _ = APTS[aid]
        cands, evals = grids(aid)
        chunks = np.array_split(np.arange(len(cands)), 16)
        free = np.concatenate(pool_map(p10_all, [(free_space(apt.plan.width, apt.plan.depth), cands[c], evals) for c in chunks]))
        for style in STYLES:
            truth = np.concatenate(pool_map(p10_all, [(plan_of(aid, style), cands[c], evals) for c in chunks]))
            now = pool_map(p10_all, [(plan_of(aid, style), [s.pos], evals) for s in sites])
            assert np.all(np.isfinite(truth)) and np.all(np.isfinite(free)), aid
            for (a2, s2, k), room in zip(mk, maps):
                if a2 != aid or s2 != style:
                    continue
                mp = np.concatenate(pool_map(p10_all, [(room, cands[c], evals) for c in chunks]))
                picks.append({"apt": aid, "style": style, "site": sites[k].room, "now_db": now[k][0], "map_db": float(truth[int(np.argmax(mp))]),
                              "free_db": float(truth[int(np.argmax(free))]), "best_db": float(truth.max())})
    save_json(OUT / "placement.json", picks)

    inner = [n for apt, *_ in APTS.values() for n in apt.kind if apt.kind[n] == "wall" and n not in apt.outer]
    concrete = np.mean([apt.thick[n] >= 0.15 for apt, *_ in APTS.values() for n in apt.kind if apt.kind[n] == "wall" and n not in apt.outer])
    print(f"{len(inner)} inner walls. Under the swiss rule {concrete:.0%} are concrete (15 cm or more)")
    print("part 1: rooms, mean per apartment, then over apartments")
    print(f"{'extend':6s} {'style':7s} {'mode':6s} {'APs':>3s} {'walls':>11s} {'false':>6s} {'rooms':>11s} {'IoU':>5s}")
    for ext_name in EXTENDS:
        for style, mode, n_ap in [("-", "truth", 0)] + list(itertools.product(STYLES, MODES, (1, 2, 3))):
            rs = [r for r in rows if r["extend"] == ext_name and r["style"] == style and r["mode"] == mode and r["n_ap"] == n_ap]
            m = lambda k: per_apartment(rs, k)  # noqa: E731
            print(f"{ext_name:6s} {style:7s} {mode:6s} {n_ap:3d} {m('walls_found'):4.1f}/{m('walls_true'):4.1f} {m('walls_false'):6.1f} {m('rooms_ok'):4.1f}/{m('rooms_true'):4.1f} {m('mean_iou'):5.2f}")
    print("part 2: p10 (dBm), mean per apartment, then over apartments")
    for style in STYLES:
        rs = [r for r in picks if r["style"] == style]
        m = lambda k: per_apartment(rs, k)  # noqa: E731
        win = np.mean([r["map_db"] > r["free_db"] + 0.5 for r in rs])
        lose = np.mean([r["map_db"] < r["free_db"] - 0.5 for r in rs])
        print(f"{style:7s} now {m('now_db'):6.1f}  map pick {m('map_db'):6.1f}  free pick {m('free_db'):6.1f}  best {m('best_db'):6.1f}   map beats free by 0.5 dB in {win:.0%} of {len(rs)} runs, loses in {lose:.0%}")


if __name__ == "__main__":
    main()
