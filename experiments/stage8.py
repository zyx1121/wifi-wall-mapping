"""Stage 8: phone positions unknown. Where do the LoS length and the walk come from?

Uses the links of stage 6 (plan B, P.2040 walls, fixed noise floor,
160 MHz). Only the positions and the LoS length d0 change:
- truth: true positions and d0 (stages 5 to 7)
- ftm: d0 from FTM with 0.3 m noise, position from d0 and the measured angle, no IMU
- pdr: dead reckoning alone, from a known start position and heading
- imu+angle: EKF of PDR and the measured cos(angle), start heading unknown
- imu+angle+ftm: the same EKF plus FTM ranges

Two IMUs: "phone" (4% step scale, 3 degrees per step, bias walk 0.2
degrees per step) and "poor" (9%, 5 degrees, 0.5 degrees per step).
We set the EKF noise from these numbers, not from the results.

    uv run python stage8.py
"""

from __future__ import annotations

import itertools

import common  # noqa: F401  (BLAS threads)
import numpy as np

import stage6 as S6
from common import out_dir, pool_map, save_json
from wallmap.floorplan import walk
from wallmap.fusion import Link, fuse_walk
from wallmap.pdr import ImuNoise, dead_reckon, simulate, track
from wallmap.rooms import close_rooms, score_rooms

OUT = out_dir("stage8")
AP = S6.AP
BW = 160
ESTS = ["nomp", "music"]
IMUS = {"phone": ImuNoise(), "poor": ImuNoise(0.09, 0.03, 5.0, 0.5)}
MODES = ["truth", "ftm", "pdr", "imu+angle", "imu+angle+ftm"]
FTM_SD = 0.3


def run(args):
    pid, seed, est, imu, rs = args
    plan, _ = S6.PLANS[pid]
    pos = walk(seed=seed, waypoints=plan.waypoints)
    smooth = walk(seed=seed, jitter=0.0, waypoints=plan.waypoints)
    rng = np.random.default_rng(seed + 100)
    noise = IMUS[imu]
    lengths, headings, offset = simulate(smooth, noise, rng)
    lks = [r["link"] for r in rs]
    d0_true = np.array([lk.d0 for lk in lks])
    ftm = d0_true + FTM_SD * rng.standard_normal(len(lks))
    cos_m = np.cos([lk.phi0 for lk in lks])
    rows = []
    for mode in MODES:
        if mode == "truth":
            p, d0 = pos, d0_true
        elif mode == "ftm":
            d0 = ftm
            p = AP + d0[:, None] * np.stack([cos_m, np.sqrt(1 - np.clip(cos_m, -1, 1) ** 2)], axis=1)
        elif mode == "pdr":
            p = dead_reckon(pos[0], lengths, headings, offset)
            d0 = np.linalg.norm(p - AP, axis=1)
        else:
            p = track(AP, lengths, headings, cos_m, ftm if mode.endswith("ftm") else None, noise)
            d0 = np.linalg.norm(p - AP, axis=1)
        links = [Link(lk.phi0, float(d), lk.phis, lk.dls) for lk, d in zip(lks, d0)]
        fw = fuse_walk(AP, links, list(d0), p)
        rm = close_rooms(fw, p)
        perr = np.linalg.norm(p - pos, axis=1)
        rows.append({
            "plan": pid, "seed": seed, "est": est, "imu": imu, "mode": mode,
            "pos_p50": float(np.median(perr)), "pos_p90": float(np.percentile(perr, 90)),
            "d0_p50": float(np.median(np.abs(d0 - d0_true))),
            **score_rooms(rm, plan.boxes, plan.doors), **S6.wall_score(plan, fw),
        })
    return rows


def main():
    jobs = []
    for pid, (plan, seeds) in S6.PLANS.items():
        for seed in seeds:
            n = len(walk(seed=seed, waypoints=plan.waypoints))
            jobs += [(pid, seed, i, est, bw) for est in S6.ESTS for bw in S6.BWS for i in range(n)]
    res = S6.load_links(jobs)
    by: dict[tuple, list] = {}
    for (pid, seed, i, est, bw), r in zip(jobs, res):
        if bw == BW and pid == "B":
            by.setdefault((pid, seed, est), []).append(r)
    runs = [(pid, seed, est, imu, rs) for (pid, seed, est), rs in by.items() if est in ESTS for imu in IMUS]
    rows = [r for out in pool_map(run, runs) for r in out]
    save_json(OUT / "stage8.json", rows)
    print(f"{'est':6s} {'imu':6s} {'mode':14s} {'pos p50/p90':>12s} {'d0 p50':>7s} {'walls':>10s} {'rooms':>6s} {'IoU':>5s}")
    for est, imu, mode in itertools.product(ESTS, IMUS, MODES):
        rs = [r for r in rows if r["est"] == est and r["imu"] == imu and r["mode"] == mode]
        if imu == "poor" and mode in ("truth", "ftm"):
            continue
        m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
        print(f"{est:6s} {imu:6s} {mode:14s} {m('pos_p50'):5.2f}/{m('pos_p90'):5.2f} {m('d0_p50'):7.2f} {m('walls_found'):4.1f}/7 {m('walls_false'):3.1f}F {m('rooms_ok'):4.1f}/5 {m('mean_iou'):5.2f}")


if __name__ == "__main__":
    main()
