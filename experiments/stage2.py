"""Stage 2: ideal CSI, 5 estimators, sweep bandwidth x antennas x SNR.

One link per trial (one client position), d0 known, no hardware impairments.
Also runs the oracle model order (true path count) for NOMP and MUSIC.
"""

from __future__ import annotations

import itertools
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import default_setup, out_dir, pool_map, random_client, save_json, single_link_trial, summarize
from wallmap.geometry import image_paths

OUT = out_dir("stage2")
BWS = [80, 160, 320]
ANTS = [2, 4, 8]
SNRS = [10, 20, 30, 40]
ESTS = ["ifft", "omp", "nomp", "music", "mpencil"]
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def true_count(seed):
    rng, room, ap = default_setup(seed)
    c = random_client(rng, room, ap.pos)
    return min(20, len(image_paths(room, ap, c, max_order=2)))


def main():
    jobs, keys = [], []
    for bw, m, snr, est in itertools.product(BWS, ANTS, SNRS, ESTS):
        for t in range(TRIALS):
            jobs.append((1000 + t, bw, m, snr, est, None, None))
            keys.append(("auto", bw, m, snr, est))
    # oracle model order on the reference SNR
    for bw, m, est in itertools.product(BWS, ANTS, ["nomp", "music"]):
        for t in range(TRIALS):
            jobs.append((1000 + t, bw, m, 30.0, est, None, true_count(1000 + t)))
            keys.append(("oracle_k", bw, m, 30.0, est))
    rows = pool_map(single_link_trial, jobs)
    groups: dict = {}
    for k, r in zip(keys, rows):
        groups.setdefault(k, []).append(r)
    table = [dict(zip(["k_mode", "bw", "ants", "snr", "est"], k)) | summarize(v) for k, v in groups.items()]
    save_json(OUT / "stage2.json", {"trials": TRIALS, "table": table})

    # figure: median wall error vs bandwidth at 30 dB, one panel per antenna count
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, m in zip(axes, ANTS):
        for est in ESTS:
            ys = [next(r["dist_med"] for r in table if r["k_mode"] == "auto" and r["bw"] == bw and r["ants"] == m and r["snr"] == 30 and r["est"] == est) for bw in BWS]
            ys = [np.nan if v is None else v for v in ys]
            ax.plot(BWS, ys, marker="o", label=est)
        ax.axhline(0.3, color="gray", ls="--", lw=1)
        ax.set_title(f"{m} antennas, 30 dB")
        ax.set_xticks(BWS)
        ax.set_xlabel("bandwidth (MHz)")
    axes[0].set_ylabel("median wall distance error (m)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_bw.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, m in zip(axes, ANTS):
        for est in ESTS:
            ys = [next(r["detect"] for r in table if r["k_mode"] == "auto" and r["bw"] == bw and r["ants"] == m and r["snr"] == 30 and r["est"] == est) for bw in BWS]
            ax.plot(BWS, ys, marker="o", label=est)
        ax.set_title(f"{m} antennas, 30 dB")
        ax.set_xticks(BWS)
        ax.set_xlabel("bandwidth (MHz)")
    axes[0].set_ylabel("wall detection rate (3 walls)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "detect_vs_bw.png", dpi=150)

    ref = [r for r in table if r["k_mode"] == "auto" and r["bw"] == 160 and r["ants"] == 4 and r["snr"] == 30]
    print("reference 160 MHz, 4 antennas, 30 dB:")
    for r in ref:
        print(f"  {r['est']:8s} detect={r['detect']:.2f} med={r['dist_med']} p90={r['dist_p90']} false={r['false_per_trial']:.1f} secs={r['secs_med']:.2f}")


if __name__ == "__main__":
    main()
