"""Stage 7: direct-path attenuation tests the found walls and gives their loss.

Uses the links of stage 6 (materials and a fixed noise floor). Per walk:
1. Fuse the walls and close the rooms (stages 5 and 6).
2. Find the direct path at the known LoS length and fit its excess loss: one offset plus one loss per wall line
   (wallmap.attenuation).
3. Drop wall lines that at least 10 links cross with less than 1.7 dB.
   Close the rooms again and refit.
4. Predict the radio map with the fitted losses, not 5 dB per wall.

We set the 2 thresholds on plan A. Plan B and B-rc are held out.

    uv run python stage7.py
"""

from __future__ import annotations

import common  # noqa: F401  (BLAS threads)
import numpy as np

import stage6 as S6
from common import out_dir, pool_map, save_json
from wallmap.attenuation import crossings, excess_db, false_lines, fit_losses
from wallmap.floorplan import line_id, walk
from wallmap.fusion import fuse_walk
from wallmap.geometry import Array, canonical, wall_line
from wallmap.materials import loss_db
from wallmap.radiomap import grid, plan_from_segments, rss_dbm
from wallmap.rooms import close_rooms, score_rooms

OUT = out_dir("stage7")
AP = S6.AP
BW = 160
ESTS = ["nomp", "music"]


def label(plan, seg) -> str | None:
    """True line id of a found wall line, or None for a false wall."""
    n = np.array([-seg.t[1], seg.t[0]])
    th, rho = canonical(n, float(np.dot(seg.a - AP, n)))
    for w in plan.walls():
        tt, tr = wall_line(AP, w)
        if abs((th - tt + np.pi) % (2 * np.pi) - np.pi) < np.deg2rad(10) and abs(rho - tr) < 0.5:
            return line_id(w.name)
    return None


DEBUG = False


def labels_dbg(plan, s):
    return label(plan, s) or f"false ({s.a[0]:.1f},{s.a[1]:.1f})-({s.b[0]:.1f},{s.b[1]:.1f})"


def fit(rm, pos, exc):
    a = crossings(AP, pos, rm.closed, len(rm.walls))
    return fit_losses(exc, a)


def run(args):
    pid, seed, est, rs = args
    plan, _ = S6.PLANS[pid]
    pos = walk(seed=seed, waypoints=plan.waypoints)
    lks = [r["link"] for r in rs]
    fw = fuse_walk(AP, lks, [lk.d0 for lk in lks], pos)
    rm = close_rooms(fw, pos)
    lam = Array(AP).wavelength
    exc = excess_db([r["paths"] for r in rs], [lk.d0 for lk in lks], lam)
    c, loss, count = fit(rm, pos, exc)
    drop = false_lines(loss, count)
    keep = [s for i, s in enumerate(rm.walls) if i not in drop]
    rm2 = close_rooms(keep, pos) if keep else rm
    c2, loss2, count2 = fit(rm2, pos, exc)
    if DEBUG:
        for i, s in enumerate(rm.walls):
            print(pid, seed, est, labels_dbg(plan, s), f"fit {loss[i]:.1f} dB", int(count[i]), "dropped" if i in drop else "")

    labels = [label(plan, s) for s in rm.walls]
    labels2 = [label(plan, s) for s in rm2.walls]
    walls = []
    for i, lab in enumerate(labels2):
        true_db = loss_db(plan.materials[lab], Array(AP).fc) if lab else None
        walls.append({"line": lab, "fit_db": float(loss2[i]), "links": int(count2[i]), "true_db": true_db})
    row = {
        "plan": pid, "seed": seed, "est": est,
        "before": {**score_rooms(rm, plan.boxes, plan.doors), "false_walls": sum(lab is None for lab in labels)},
        "after": {**score_rooms(rm2, plan.boxes, plan.doors), "false_walls": sum(lab is None for lab in labels2)},
        "dropped": [labels[i] or "false" for i in drop],
        "offset_db": c2,
        "walls": walls,
    }
    pieces = rm2.closed
    losses = {f"w{i}": float(loss2[s.line]) for i, s in enumerate(pieces)}
    return row, (pieces, losses)


def rss_job(args):
    room, pts = args
    return rss_dbm(room, Array(AP), pts)


def main():
    jobs = []
    for pid, (plan, seeds) in S6.PLANS.items():
        for seed in seeds:
            n = len(walk(seed=seed, waypoints=plan.waypoints))
            jobs += [(pid, seed, i, est, bw) for est in S6.ESTS for bw in S6.BWS for i in range(n)]
    res = S6.load_links(jobs)
    by: dict[tuple, list] = {}
    for (pid, seed, i, est, bw), r in zip(jobs, res):
        if bw == BW:
            by.setdefault((pid, seed, est), []).append(r)
    runs = [(pid, seed, est, by[(pid, seed, est)]) for pid, (_, seeds) in S6.PLANS.items() for seed in seeds for est in ESTS]
    out = pool_map(run, runs)
    rows = [o[0] for o in out]
    save_json(OUT / "stage7.json", rows)

    print(f"{'plan':5s} {'est':6s} {'IoU before/after':>17s} {'rooms':>9s} {'false walls':>12s}  dropped")
    for pid in S6.PLANS:
        for est in ESTS:
            rs = [r for r in rows if r["plan"] == pid and r["est"] == est]
            m = lambda ph, k: float(np.mean([r[ph][k] for r in rs]))  # noqa: E731
            drops = [d for r in rs for d in r["dropped"]]
            print(f"{pid:5s} {est:6s} {m('before', 'mean_iou'):8.2f}/{m('after', 'mean_iou'):.2f} {m('before', 'rooms_ok'):4.1f}/{m('after', 'rooms_ok'):.1f} {m('before', 'false_walls'):6.1f}/{m('after', 'false_walls'):.1f}  {drops}")
    print("wall loss fits (NOMP, true wall lines crossed by 10+ links):")
    for pid in S6.PLANS:
        errs = [w["fit_db"] - w["true_db"] for r in rows if r["plan"] == pid and r["est"] == "nomp" for w in r["walls"] if w["line"] and w["links"] >= 10]
        per = {}
        for r in rows:
            if r["plan"] == pid and r["est"] == "nomp":
                for w in r["walls"]:
                    if w["line"] and w["links"] >= 10:
                        per.setdefault(w["line"], []).append((round(w["fit_db"], 1), round(w["true_db"], 1)))
        if errs:
            print(f"  {pid:5s} median |err| {np.median(np.abs(errs)):.1f} dB over {len(errs)} walls: {per}")

    # radio map with fitted losses, plan B and B-rc, NOMP
    rmap = []
    for pid in ("B", "B-rc"):
        plan, seeds = S6.PLANS[pid]
        pts = grid(plan.width, plan.depth)
        x0, y0, x1, y1 = plan.boxes[0][1:]
        other = ~((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1))
        chunks = np.array_split(np.arange(len(pts)), 32)
        truth = np.concatenate(pool_map(rss_job, [(plan, pts[c]) for c in chunks]))
        errs = []
        for (rp, seed, est, _), (_, (pieces, losses)) in zip(runs, out):
            if rp != pid or est != "nomp":
                continue
            room = plan_from_segments(pieces, plan.width, plan.depth, losses=losses)
            pred = np.concatenate(pool_map(rss_job, [(room, pts[c]) for c in chunks]))
            errs.append(pred - truth)
        e = np.stack(errs)[:, other]
        row = {"plan": pid, "pred": "nomp walls, fitted loss", "median_abs_db": float(np.median(np.abs(e))), "p90_abs_db": float(np.percentile(np.abs(e), 90)), "bias_db": float(np.mean(e))}
        rmap.append(row)
        print(f"radio map {pid:5s} fitted loss, other rooms: median {row['median_abs_db']:.1f} dB, p90 {row['p90_abs_db']:.1f}, bias {row['bias_db']:+.1f}")
    save_json(OUT / "radiomap.json", rmap)


if __name__ == "__main__":
    main()
