"""Export a walk through the apartment for the web demo.

One AP on the south wall of the living room. A client walks through every
room, one position every 0.3 m. Each position gives one link. The demo plays
the links in order and shows the walls that fuse_walk finds after each one.
Walls follow ITU-R P.2040 and the noise floor is fixed (stage 6). 4
antennas, ideal hardware, known LoS length.

Every SNAP positions it also closes the rooms (stage 5), fits the loss of
each wall (stage 7) and predicts the received power over a grid.

Writes demo/public/wall-mapping-assets/walk.json.

    uv run python demo_walk.py
"""

from __future__ import annotations

import itertools
import json

import common  # noqa: F401  (BLAS threads)

import numpy as np
from scipy import ndimage

from common import ROOT, pool_map
from wallmap.attenuation import crossings, excess_db, fit_losses
from wallmap.csi import measure
from wallmap.estimators import ESTIMATORS, relative_params
from wallmap.floorplan import MATERIALS_A, apartment, line_id, walk
from wallmap.fusion import Link, fuse_walk
from wallmap.geometry import Array, image_paths, wall_line, walls_from_paths
from wallmap.radiomap import plan_from_segments, rss_dbm
from wallmap.rooms import close_rooms, score_rooms

AP = np.array([2.5, 0.1])
BWS = [160, 320]
ANTS = 4
SNR_1M = {160: 51.0, 320: 48.0}
SNAP = 10
CELL = 0.2  # room grid (m)
HEAT = 0.4  # radio map grid (m)
OUT = ROOT / "demo" / "public" / "wall-mapping-assets" / "walk.json"

PLAN = apartment().with_materials(MATERIALS_A)
POS = walk()


def r2(v):
    return round(float(v), 2)


def targets():
    out = {}
    for w in PLAN.walls():
        out.setdefault(line_id(w.name), wall_line(AP, w))
    return out


TARGETS = targets()


def label(th, rho):
    for lid, (tt, tr) in TARGETS.items():
        if abs((th - tt + np.pi) % (2 * np.pi) - np.pi) < np.deg2rad(10) and abs(rho - tr) < 0.5:
            return lid
    return None


def estimate(args):
    i, est, bw = args
    ap = Array(AP, n_ant=ANTS)
    paths = image_paths(PLAN, ap, POS[i], max_order=2)
    los = [q for q in paths if q.is_los][0]
    y, n_k, s2 = measure(paths, ap, bw, 0.0, np.random.default_rng(i), snr_1m_db=SNR_1M[bw])
    p = ESTIMATORS[est](y, n_k, s2, ap.fc)
    rel = relative_params(p)
    # path lengths: ideal hardware keeps delays absolute
    found = [[r2(s), r2(t * 299_792_458.0)] for s, t in zip(p.s, p.tau)]
    exc = float(excess_db([p], [los.length], ap.wavelength)[0])
    return rel, los.length, found, exc


def heat_grid():
    xs = np.arange(HEAT / 2, PLAN.width, HEAT)
    ys = np.arange(HEAT / 2, PLAN.depth, HEAT)
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


def snapshot(fw, n, exc, truth_heat):
    """Rooms, wall losses and the radio map after the first n positions."""
    pos = POS[:n]
    rm = close_rooms(fw, pos)
    # paint the small empty cells inside rooms (between close parallel lines) with the room around them
    lab = rm.labels
    holes = ndimage.binary_fill_holes(lab > 0) & (lab == 0)
    while holes.any():
        grown = ndimage.grey_dilation(lab, size=3)
        fill = holes & (grown > 0)
        if not fill.any():
            break
        lab[fill] = grown[fill]
        holes &= lab == 0
    xs, ys = room_grid()
    labels = []
    for y in ys:
        for x in xs:
            i, j = rm.cell((x, y))
            ok = 0 <= i < rm.labels.shape[0] and 0 <= j < rm.labels.shape[1]
            labels.append(int(rm.labels[i, j]) if ok else 0)
    score = score_rooms(rm, PLAN.boxes, PLAN.doors)
    losses = {}
    if rm.closed:
        a = crossings(AP, pos, rm.closed, len(rm.walls))
        _, loss, _ = fit_losses(exc[:n], a)
        losses = {f"w{i}": float(loss[s.line]) for i, s in enumerate(rm.closed)}
    room = plan_from_segments(rm.closed, PLAN.width, PLAN.depth, losses=losses)
    _, _, pts = heat_grid()
    heat = rss_dbm(room, Array(AP, n_ant=ANTS), pts, max_order=1)
    x0, y0, x1, y1 = PLAN.boxes[0][1:]
    other = ~((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1))
    return {
        "n": n,
        "rooms": rle(labels),
        "rooms_ok": score["rooms_ok"],
        "walls": [[r2(s.a[0]), r2(s.a[1]), r2(s.b[0]), r2(s.b[1])] for s in rm.closed],
        "heat": [int(round(v)) for v in heat],
        "heat_err": r2(np.median(np.abs(heat - truth_heat)[other])),
    }


def fuse_all(args):
    est, bw, rels, truth_heat = args
    exc = np.array([e for *_, e in rels])
    links, d0s = [], []
    for rel, d0, _, _ in rels:
        rel = rel if rel is not None else (np.pi / 2, np.zeros(0), np.zeros(0))
        links.append(Link(rel[0], d0, rel[1], rel[2]))
        d0s.append(d0)
    steps, snaps = [], []
    for n in range(1, len(links) + 1):
        fw = fuse_walk(AP, links[:n], d0s[:n], POS[:n]) if n >= 8 else []
        steps.append([[r2(f["p0"][0]), r2(f["p0"][1]), r2(f["p1"][0]), r2(f["p1"][1]), label(f["theta"], f["rho"])] for f in fw])
        if fw and (n % SNAP == 0 or n == len(links)):
            snaps.append(snapshot(fw, n, exc, truth_heat))
    cand = []
    for lk, d0 in zip(links, d0s):
        lines = walls_from_paths(AP, lk.phi0, d0, lk.phis, lk.dls) if len(lk.phis) else np.zeros((0, 2))
        cand.append([[r2(t), r2(r)] for t, r in lines])
    return est, bw, steps, cand, snaps


def main():
    jobs = [(i, est, bw) for est, bw in itertools.product(ESTIMATORS, BWS) for i in range(len(POS))]
    res = pool_map(estimate, jobs)
    by = {}
    for (i, est, bw), r in zip(jobs, res):
        by.setdefault((est, bw), []).append(r)
    hx, hy, hpts = heat_grid()
    truth_heat = rss_dbm(PLAN, Array(AP, n_ant=ANTS), hpts, max_order=2)
    fused = pool_map(fuse_all, [(est, bw, by[(est, bw)], truth_heat) for est, bw in itertools.product(ESTIMATORS, BWS)])

    ap = Array(AP, n_ant=ANTS)
    truth = []
    for c in POS:
        paths = image_paths(PLAN, ap, c, max_order=2)
        gmax = max(abs(q.gain) for q in paths)
        truth.append([[r2(q.u[0]), r2(q.length), q.order] for q in paths if q.order <= 1 and 20 * np.log10(abs(q.gain) / gmax) > -30])

    runs = {}
    for est, bw, steps, cand, snaps in fused:
        runs.setdefault(est, {})[str(bw)] = {
            "fused": steps,
            "cand": cand,
            "found": [r[2] for r in by[(est, bw)]],
            "snaps": snaps,
        }
    rx, ry = room_grid()
    out = {
        "plan": {
            "width": PLAN.width,
            "depth": PLAN.depth,
            "walls": [[w.p0[0], w.p0[1], w.p1[0], w.p1[1]] for w in PLAN.walls()],
            "rooms": [[name, x, y] for name, (x, y) in PLAN.rooms],
        },
        "ap": AP.tolist(),
        "targets": [k for k in TARGETS if k != "south"],
        "positions": [[r2(x), r2(y)] for x, y in POS],
        "truth": truth,
        "bws": BWS,
        "runs": runs,
        "rooms_grid": {"x0": r2(rx[0] - CELL / 2), "y0": r2(ry[0] - CELL / 2), "cell": CELL, "nx": len(rx), "ny": len(ry)},
        "heat_grid": {"x0": 0.0, "y0": 0.0, "cell": HEAT, "nx": len(hx), "ny": len(hy)},
        "heat_truth": [int(round(v)) for v in truth_heat],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    for est, bw, steps, _, snaps in fused:
        last = steps[-1]
        hit = {w[4] for w in last if w[4] not in (None, "south")}
        sn = snaps[-1] if snaps else {"rooms_ok": 0, "heat_err": float("nan")}
        print(f"{est:8s} {bw}: found {len(hit)}/7, false {sum(1 for w in last if w[4] is None)}, rooms {sn['rooms_ok']}/5, radio map {sn['heat_err']} dB")


if __name__ == "__main__":
    main()
