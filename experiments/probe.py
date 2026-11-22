"""Time each estimator once per bandwidth at 8 antennas (worst case)."""

import sys

from common import single_link_trial

for bw in (80, 160, 320):
    for est in ("ifft", "omp", "nomp", "music", "mpencil"):
        r = single_link_trial((1, bw, 8, 30.0, est, None, None))
        print(f"{bw:3d} MHz {est:8s} {r['secs']:6.2f} s  paths={r['n_paths']:2d} found={sum(r['found'].values())} "
              f"false={r['n_false']} err={ {k: round(v, 3) for k, v in r['dist_err'].items()} }", flush=True)
sys.exit(0)
