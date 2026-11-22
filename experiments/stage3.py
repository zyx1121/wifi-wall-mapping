"""Stage 3: hardware impairments, one at a time, then all together.

One link per trial, d0 known. Each impairment is drawn fresh per packet.
"""

from __future__ import annotations

import itertools
import sys

import common  # noqa: F401  (BLAS threads)
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import out_dir, pool_map, save_json, single_link_trial, summarize
from wallmap.csi import ALL_CALIBRATED, Impairments

OUT = out_dir("stage3")
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CONFIGS = [(160, 4, 30.0), (320, 4, 30.0)]
ESTS = ["ifft", "omp", "nomp", "music", "mpencil"]
IMPS = {
    "ideal": Impairments(),
    "1 random phase": Impairments(random_phase=True),
    "2 timing offset 100 ns": Impairments(sto_ns=100.0),
    "3a antenna phase, calibrated 5 deg": Impairments(ant_phase="calibrated"),
    "3b antenna phase, uncalibrated": Impairments(ant_phase="uncalibrated"),
    "4 AGC 3 dB": Impairments(agc_db=3.0),
    "5 quantization 8 bit": Impairments(quant_bits=8),
    "6 RF chain ripple": Impairments(rf_ripple=True),
    "all, calibrated": ALL_CALIBRATED,
    "all, uncalibrated": Impairments(True, 100.0, "uncalibrated", 5.0, 3.0, True, 8),
}


def main():
    jobs, keys = [], []
    for (bw, m, snr), est, (iname, imp) in itertools.product(CONFIGS, ESTS, IMPS.items()):
        for t in range(TRIALS):
            jobs.append((2000 + t, bw, m, snr, est, imp, None))
            keys.append((bw, m, snr, est, iname))
    rows = pool_map(single_link_trial, jobs)
    groups: dict = {}
    for k, r in zip(keys, rows):
        groups.setdefault(k, []).append(r)
    table = [dict(zip(["bw", "ants", "snr", "est", "impairment"], k)) | summarize(v) for k, v in groups.items()]
    save_json(OUT / "stage3.json", {"trials": TRIALS, "table": table})

    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(13, 4.8), sharey=True)
    names = list(IMPS)
    for ax, (bw, m, snr) in zip(axes, CONFIGS):
        x = np.arange(len(names))
        wdt = 0.16
        for i, est in enumerate(ESTS):
            ys = []
            for n in names:
                r = next(r for r in table if r["bw"] == bw and r["ants"] == m and r["est"] == est and r["impairment"] == n)
                ys.append(np.nan if r["dist_med"] is None else r["dist_med"])
            ax.bar(x + (i - 2) * wdt, ys, wdt, label=est)
        ax.axhline(0.3, color="gray", ls="--", lw=1)
        ax.set_xticks(x, names, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{bw} MHz, {m} antennas, {snr:g} dB")
    axes[0].set_ylabel("median wall distance error (m)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "impairments.png", dpi=150)

    for bw, m, snr in CONFIGS:
        print(f"{bw} MHz, {m} antennas, {snr:g} dB")
        for n in names:
            cells = []
            for est in ESTS:
                r = next(r for r in table if r["bw"] == bw and r["est"] == est and r["impairment"] == n)
                med = "-" if r["dist_med"] is None else f"{r['dist_med']:.2f}"
                cells.append(f"{est}={med}/{r['detect']:.2f}")
            print(f"  {n:36s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
