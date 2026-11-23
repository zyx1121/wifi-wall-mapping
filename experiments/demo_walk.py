"""Export a walk through the apartment for the web demo.

One AP on the south wall of the living room. A client walks through every
room, one position every 0.3 m. Each position gives one link. The demo plays
the links in order and shows the walls that fuse_walk finds after each one.
4 antennas, 30 dB, ideal hardware, known LoS length.

Writes demo/public/wall-mapping-assets/walk.json.

    uv run python demo_walk.py
"""

from __future__ import annotations

import itertools
import json

import common  # noqa: F401  (BLAS threads)
import numpy as np

from common import ROOT, pool_map
from wallmap.csi import measure
from wallmap.estimators import ESTIMATORS, relative_params
from wallmap.floorplan import apartment, line_id, walk
from wallmap.fusion import Link, fuse_walk
from wallmap.geometry import Array, image_paths, wall_line, walls_from_paths

AP = np.array([2.5, 0.1])
BWS = [160, 320]
ANTS = 4
SNR = 30.0
OUT = ROOT / "demo" / "public" / "wall-mapping-assets" / "walk.json"

PLAN = apartment()
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
    y, n_k, s2 = measure(paths, ap, bw, SNR, np.random.default_rng(i))
    p = ESTIMATORS[est](y, n_k, s2, ap.fc)
    rel = relative_params(p)
    # path lengths: ideal hardware keeps delays absolute
    found = [[r2(s), r2(t * 299_792_458.0)] for s, t in zip(p.s, p.tau)]
    return rel, los.length, found


def fuse_all(args):
    est, bw, rels = args
    links, d0s = [], []
    for rel, d0, _ in rels:
        rel = rel if rel is not None else (np.pi / 2, np.zeros(0), np.zeros(0))
        links.append(Link(rel[0], d0, rel[1], rel[2]))
        d0s.append(d0)
    steps = []
    for n in range(1, len(links) + 1):
        fw = fuse_walk(AP, links[:n], d0s[:n], POS[:n]) if n >= 8 else []
        steps.append([[r2(f["p0"][0]), r2(f["p0"][1]), r2(f["p1"][0]), r2(f["p1"][1]), label(f["theta"], f["rho"])] for f in fw])
    cand = []
    for lk, d0 in zip(links, d0s):
        lines = walls_from_paths(AP, lk.phi0, d0, lk.phis, lk.dls) if len(lk.phis) else np.zeros((0, 2))
        cand.append([[r2(t), r2(r)] for t, r in lines])
    return est, bw, steps, cand


def main():
    jobs = [(i, est, bw) for est, bw in itertools.product(ESTIMATORS, BWS) for i in range(len(POS))]
    res = pool_map(estimate, jobs)
    by = {}
    for (i, est, bw), r in zip(jobs, res):
        by.setdefault((est, bw), []).append(r)
    fused = pool_map(fuse_all, [(est, bw, by[(est, bw)]) for est, bw in itertools.product(ESTIMATORS, BWS)])

    ap = Array(AP, n_ant=ANTS)
    truth = []
    for c in POS:
        paths = image_paths(PLAN, ap, c, max_order=2)
        gmax = max(abs(q.gain) for q in paths)
        truth.append([[r2(q.u[0]), r2(q.length), q.order] for q in paths if q.order <= 1 and 20 * np.log10(abs(q.gain) / gmax) > -30])

    runs = {}
    for est, bw, steps, cand in fused:
        runs.setdefault(est, {})[str(bw)] = {
            "fused": steps,
            "cand": cand,
            "found": [f for _, _, f in by[(est, bw)]],
        }
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
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    for est, bw, steps, _ in fused:
        last = steps[-1]
        hit = {w[4] for w in last if w[4] not in (None, "south")}
        print(f"{est:8s} {bw}: found {len(hit)}/7, false {sum(1 for w in last if w[4] is None)}")


if __name__ == "__main__":
    main()
