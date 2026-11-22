"""Shared setup for the stage scripts."""

from __future__ import annotations

import os

# OpenBLAS with many threads makes a 320x320 eigh 800x slower on the sandbox VM.
# Use 1 BLAS thread per process and parallelise with processes instead.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path as FsPath

import numpy as np

from wallmap.csi import Impairments, measure
from wallmap.estimators import ESTIMATORS, relative_params
from wallmap.geometry import Array, Room, image_paths, match_walls, oracle_params, walls_from_paths

ROOT = FsPath(__file__).resolve().parents[1]
TARGETS = ("east", "north", "west")
AP_X, AP_Y = 2.0, 0.1


def out_dir(stage: str) -> FsPath:
    d = ROOT / "results" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def random_client(rng, room: Room, ap_pos, margin=0.5, min_dist=1.0) -> np.ndarray:
    while True:
        p = np.array([rng.uniform(margin, room.width - margin), rng.uniform(margin, room.depth - margin)])
        if np.linalg.norm(p - ap_pos) >= min_dist:
            return p


def default_setup(seed: int, n_ant=4):
    rng = np.random.default_rng(seed)
    room = Room()
    ap = Array(np.array([AP_X, AP_Y]), n_ant=n_ant)
    return rng, room, ap


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-trial match results."""
    found = [r["found"][w] for r in rows for w in TARGETS]
    derr = [r["dist_err"][w] for r in rows for w in TARGETS if r["found"][w]]
    aerr = [r["ang_err"][w] for r in rows for w in TARGETS if r["found"][w]]
    out = {
        "trials": len(rows),
        "detect": float(np.mean(found)) if found else 0.0,
        "dist_med": float(np.median(derr)) if derr else None,
        "dist_p90": float(np.percentile(derr, 90)) if derr else None,
        "ang_med": float(np.median(aerr)) if aerr else None,
        "false_per_trial": float(np.mean([r["n_false"] for r in rows])),
    }
    if rows and "secs" in rows[0]:
        out["secs_med"] = float(np.median([r["secs"] for r in rows]))
    return out


def match_row(m) -> dict:
    return {"found": m.found, "dist_err": m.dist_err, "ang_err": m.ang_err, "n_false": m.n_false, "n_est": m.n_est}


def run_link(
    seed: int,
    bw: int,
    n_ant: int,
    snr: float,
    est: str,
    imp: Impairments | None = None,
    floor_ceiling=False,
    k_fixed=None,
    room: Room | None = None,
    ap_pos=None,
    client=None,
    ap_h=1.5,
    client_h=None,
):
    """Simulate one link and estimate its paths. Returns the relative params."""
    rng = np.random.default_rng(seed)
    room = room or Room()
    ap = Array(np.array([AP_X, AP_Y]) if ap_pos is None else ap_pos, n_ant=n_ant, height=ap_h)
    client = random_client(rng, room, ap.pos) if client is None else client
    paths = image_paths(room, ap, client, client_height=client_h, max_order=2, floor_ceiling=floor_ceiling)
    y, n_k, s2 = measure(paths, ap, bw, snr, rng, imp)
    t0 = time.perf_counter()
    kw = {"k_fixed": k_fixed} if k_fixed else {}
    p = ESTIMATORS[est](y, n_k, s2, ap.fc, **kw)
    secs = time.perf_counter() - t0
    los = [q for q in paths if q.is_los][0]
    return relative_params(p), los.length, client, ap, room, secs, len(p)


def single_link_trial(args) -> dict:
    seed, bw, n_ant, snr, est, imp, k_fixed = args
    rel, d0, client, ap, room, secs, n_p = run_link(seed, bw, n_ant, snr, est, imp, k_fixed=k_fixed)
    if rel is None:
        lines = np.zeros((0, 2))
    else:
        phi0, phis, dls = rel
        lines = walls_from_paths(ap.pos, phi0, d0, phis, dls)
    r = match_row(match_walls(lines, ap.pos, room))
    r["secs"] = secs
    r["n_paths"] = n_p
    return r


def pool_map(fn, jobs, workers=None):
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs, chunksize=4))


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=1, default=float))
