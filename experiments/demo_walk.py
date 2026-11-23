"""Export a walk through the held-out apartment B for the web demo.

3 APs: in the living room, the kitchen and bedroom A, each on the middle of
the room's longest outer wall (stage 9). A client walks through every room,
one position every 0.3 m. Each position gives one link to every AP. The
demo plays the links in order and shows the walls that the APs find.
Walls follow ITU-R P.2040 with a fixed noise floor (stage 6). 4 antennas,
ideal hardware, known LoS length, NOMP at 160 MHz.

Every SNAP positions it also closes the rooms over the walls of all APs
and fits the loss of each wall (stage 7, one offset per AP). It predicts the received power of the
strongest AP over a grid.

The APs and the walk are the best plan B run of stage 9 (5 of 5 rooms).
We tuned nothing on plan B, but the export is the best run, not a typical
one. Stage 9 has the typical numbers.

Writes demo/public/wall-mapping-assets/walk.json.

    uv run python demo_walk.py
"""

from __future__ import annotations

import json
import pickle

import common  # noqa: F401  (BLAS threads)

import numpy as np
from scipy import ndimage

from common import ROOT, pool_map
from wallmap.attenuation import crossings, excess_db, false_lines, fit_losses
from wallmap.csi import measure
from wallmap.estimators import ESTIMATORS, relative_params
from wallmap.floorplan import MATERIALS_B, apartment_b, line_id, walk
from wallmap.fusion import Link, fuse_walk
from wallmap.geometry import Array, image_paths, walls_from_paths
from wallmap.materials import WALLS
from wallmap.multi_ap import ap_sites, frame, label_segment, local_plan
from wallmap.radiomap import plan_from_segments, rss_dbm
from wallmap.rooms import Segment, close_rooms, score_rooms

ANTS = 4
BW = 160
SNR_1M = 51.0
SEED = 7002
COMBO = (0, 1, 3)  # site index: living room, kitchen, bedroom A
SNAP = 10
CELL = 0.2  # room grid (m)
HEAT = 0.3  # radio map grid (m), divides the plan size
# The loss test of stage 7 drops false walls. Its 1.7 dB threshold comes from 1 AP on plan A.
# With these 3 APs it also drops true walls (5 rooms become 2), so the demo keeps every wall.
DROP = False
OUT = ROOT / "demo" / "public" / "wall-mapping-assets" / "walk.json"

PLAN = apartment_b().with_materials(MATERIALS_B)
POS = walk(seed=SEED, waypoints=PLAN.waypoints)
SITES = [ap_sites(PLAN)[k] for k in COMBO]
FRAMES = [frame(PLAN, s) for s in SITES]
INNER = sorted({line_id(w.name) for w in PLAN.walls()} - {"south", "east", "north", "west"})


def r2(v):
    return round(float(v), 2)


def estimate(args):
    """One link in the frame of AP j. Same noise seeds as stage 9."""
    j, i = args
    f = FRAMES[j]
    ap = Array(f.to_local(SITES[j].pos), n_ant=ANTS)
    paths = image_paths(local_plan(PLAN, f), ap, f.to_local(POS[i]), max_order=2)
    los = [q for q in paths if q.is_los][0]
    k = COMBO[j]
    y, n_k, s2 = measure(paths, ap, BW, 0.0, np.random.default_rng(SEED * 1000 + i + 1 + 100_000 * k), snr_1m_db=SNR_1M)
    p = ESTIMATORS["nomp"](y, n_k, s2, ap.fc)
    rel = relative_params(p)
    if rel is None:
        rel = (np.pi / 2, np.zeros(0), np.zeros(0))
    found = [[r2(s), r2(t * 299_792_458.0)] for s, t in zip(p.s, p.tau)]
    exc = float(excess_db([p], [los.length], ap.wavelength)[0])
    return Link(rel[0], los.length, rel[1], rel[2]), found, exc


def heat_grid():
    xs = (np.arange(round(PLAN.width / HEAT)) + 0.5) * HEAT
    ys = (np.arange(round(PLAN.depth / HEAT)) + 0.5) * HEAT
    return xs, ys, np.array([(x, y) for y in ys for x in xs])


def room_grid():
    xs = np.arange(-0.5 + CELL / 2, PLAN.width + 0.5, CELL)
    ys = np.arange(-0.5 + CELL / 2, PLAN.depth + 0.5, CELL)
    return xs, ys


def rle(values) -> list[int]:
    out: list[int] = []
    for v in values:
        if out and out[-2] == v:
            out[-1] += 1
        else:
            out += [int(v), 1]
    return out


def strongest(room, pts, max_order):
    return np.max([rss_dbm(room, Array(s.pos, n_ant=ANTS), pts, max_order=max_order) for s in SITES], axis=0)


def fit(rm, pos, excs):
    """Loss per wall line from the links of all APs, one offset per AP."""
    a = np.vstack([crossings(s.pos, pos, rm.closed, len(rm.walls)) for s in SITES])
    groups = np.repeat(np.arange(len(SITES)), len(pos))
    return fit_losses(np.concatenate(excs), a, groups=groups)


def snapshot(segs, n, excs, truth_heat):
    """Rooms, wall losses and the radio map after the first n positions."""
    pos = POS[:n]
    excs = [e[:n] for e in excs]
    rm = close_rooms(segs, pos)
    if rm.closed and DROP:
        # stage 7: drop the wall lines that many links cross without loss
        _, loss, count = fit(rm, pos, excs)
        drop = set(false_lines(loss, count))
        keep = [s for i, s in enumerate(rm.walls) if i not in drop]
        if drop and keep:
            rm = close_rooms(keep, pos)
    score = score_rooms(rm, PLAN.boxes, PLAN.doors)
    # for drawing only: paint the empty cells inside rooms and the thin strips along walls
    lab = rm.labels.copy()
    holes = ndimage.binary_fill_holes(lab > 0) & (lab == 0)
    while holes.any():
        grown = ndimage.grey_dilation(lab, size=3)
        fill = holes & (grown > 0)
        if not fill.any():
            break
        lab[fill] = grown[fill]
        holes &= lab == 0
    for _ in range(3):
        grown = ndimage.grey_dilation(lab, size=3)
        fill = (lab == 0) & (grown > 0)
        lab[fill] = grown[fill]
    xs, ys = room_grid()
    labels = []
    for y in ys:
        for x in xs:
            i, j = rm.cell((x, y))
            ok = 0 <= i < lab.shape[0] and 0 <= j < lab.shape[1]
            labels.append(int(lab[i, j]) if ok else 0)
    losses = {}
    if rm.closed:
        _, loss, _ = fit(rm, pos, excs)
        losses = {f"w{i}": float(loss[s.line]) for i, s in enumerate(rm.closed)}
    room = plan_from_segments(rm.closed, PLAN.width, PLAN.depth, losses=losses)
    _, _, pts = heat_grid()
    heat = strongest(room, pts, 1)
    return {
        "n": n,
        "rooms": rle(labels),
        "rooms_ok": score["rooms_ok"],
        "iou": r2(score["mean_iou"]),
        "walls": [[r2(s.a[0]), r2(s.a[1]), r2(s.b[0]), r2(s.b[1])] for s in rm.closed],
        "heat": [int(round(v)) for v in heat],
        "heat_err": r2(np.median(np.abs(heat - truth_heat))),
    }


def fuse_ap(args):
    """Walls that AP j finds after each position, in the global frame."""
    j, links = args
    f = FRAMES[j]
    ap = f.to_local(SITES[j].pos)
    pos = f.to_local(POS)
    d0s = [lk.d0 for lk in links]
    out = []
    for n in range(1, len(links) + 1):
        fw = fuse_walk(ap, links[:n], d0s[:n], pos[:n]) if n >= 8 else []
        out.append([Segment(f.to_global(w["p0"]), f.to_global(w["p1"]), int(w["size"])) for w in fw])
    return out


def snap_job(args):
    return snapshot(*args)


def candidates(j, lk):
    """Candidate wall lines of one link as (theta, rho, AP) in the global frame."""
    f = FRAMES[j]
    ap = f.to_local(SITES[j].pos)
    if not len(lk.phis):
        return []
    out = []
    for th, rho in walls_from_paths(ap, lk.phi0, lk.d0, lk.phis, lk.dls):
        n = np.array([np.cos(th), np.sin(th)]) @ f.r  # local normal to global
        out.append([r2(np.arctan2(n[1], n[0])), r2(rho), j])
    return out


def thickness(lid: str) -> float:
    return round(sum(t for _, t in WALLS[PLAN.materials[lid]]), 3)


def main():
    n = len(POS)
    jobs = [(j, i) for j in range(len(SITES)) for i in range(n)]
    cache = ROOT / "results" / "raw" / "demo_links.pkl"
    key = (SEED, COMBO, BW)
    old = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    if key not in old:
        old = {key: pool_map(estimate, jobs)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(old))
    res = old[key]
    links = [[res[j * n + i][0] for i in range(n)] for j in range(len(SITES))]
    excs = [np.array([res[j * n + i][2] for i in range(n)]) for j in range(len(SITES))]
    per_ap = pool_map(fuse_ap, [(j, links[j]) for j in range(len(SITES))])
    steps_segs = [[s for j in range(len(SITES)) for s in per_ap[j][k]] for k in range(n)]

    hx, hy, hpts = heat_grid()
    truth_heat = strongest(PLAN, hpts, 2)
    at = [k for k in range(n) if steps_segs[k] and ((k + 1) % SNAP == 0 or k == n - 1)]
    snaps = pool_map(snap_job, [(steps_segs[k], k + 1, excs, truth_heat) for k in at])

    fused = [[[r2(s.a[0]), r2(s.a[1]), r2(s.b[0]), r2(s.b[1]), label_segment(PLAN, s.a, s.b)] for s in segs] for segs in steps_segs]
    cand = [[c for j in range(len(SITES)) for c in candidates(j, links[j][i])] for i in range(n)]

    # the right-hand panel shows the paths at the living room AP (frame = plan frame)
    ap0 = Array(SITES[0].pos, n_ant=ANTS)
    truth = []
    for c in POS:
        paths = image_paths(PLAN, ap0, c, max_order=2)
        gmax = max(abs(q.gain) for q in paths)
        truth.append([[r2(q.u[0]), r2(q.length), q.order] for q in paths if q.order <= 1 and 20 * np.log10(abs(q.gain) / gmax) > -30])

    rx, ry = room_grid()
    out = {
        "plan": {
            "width": PLAN.width,
            "depth": PLAN.depth,
            # x0, y0, x1, y1, thickness (m), material
            "walls": [[w.p0[0], w.p0[1], w.p1[0], w.p1[1], thickness(line_id(w.name)), PLAN.materials[line_id(w.name)]] for w in PLAN.walls()],
            "rooms": [[name, x, y] for name, (x, y) in PLAN.rooms],
            "rooms_true": len(PLAN.boxes),
        },
        "aps": [[r2(s.pos[0]), r2(s.pos[1])] for s in SITES],
        "bw": BW,
        "targets": INNER,
        "positions": [[r2(x), r2(y)] for x, y in POS],
        "truth": truth,
        "run": {"fused": fused, "cand": cand, "found": [res[i][1] for i in range(n)], "snaps": snaps},
        "rooms_grid": {"x0": r2(rx[0] - CELL / 2), "y0": r2(ry[0] - CELL / 2), "cell": CELL, "nx": len(rx), "ny": len(ry)},
        "heat_grid": {"x0": 0.0, "y0": 0.0, "cell": HEAT, "nx": len(hx), "ny": len(hy)},
        "heat_truth": [int(round(v)) for v in truth_heat],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    last, sn = fused[-1], snaps[-1]
    print(f"walls {len({w[4] for w in last if w[4] in INNER})}/{len(INNER)}, false {sum(w[4] is None for w in last)}, rooms {sn['rooms_ok']}/5 IoU {sn['iou']}, radio map {sn['heat_err']} dB")
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
