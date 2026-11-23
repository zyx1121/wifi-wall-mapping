"""Export the search steps of every estimator for the web demo.

Writes demo/public/wall-mapping-assets/trace.json: a few random rooms, one client position each,
3 bandwidths, 4 antennas, 30 dB, ideal hardware. Ideal hardware keeps the
delays absolute, so each peak maps to one image point in the room view.

    uv run python demo_trace.py [scenes]
"""

from __future__ import annotations

import base64
import sys

import common  # noqa: F401  (BLAS threads)
import numpy as np

from common import ROOT, random_client
from wallmap.csi import measure
from wallmap.estimators import ESTIMATORS, relative_params
from wallmap.geometry import C, Array, Room, image_paths, wall_line, walls_from_paths

SCENES = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BWS = [80, 160, 320]
ANTS = 4
SNR = 30.0
DB_RANGE = 60.0
OUT = ROOT / "demo" / "public" / "wall-mapping-assets" / "trace.json"
NAMES = {"east": "E", "north": "N", "west": "W", "south": "S"}


def r3(v):
    return round(float(v), 3)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def to_db8(p: np.ndarray, ref: float) -> np.ndarray:
    db = 10 * np.log10(np.maximum(p, 1e-30) / ref)
    return np.clip((db + DB_RANGE) / DB_RANGE * 255, 0, 255).round()


def level(v: float, ref: float) -> float:
    return r3((10 * np.log10(max(v, 1e-30) / ref) + DB_RANGE) / DB_RANGE * 255)


def encode_map(p, sg, tg, lmax, ref, s_step=1):
    """Crop a (S, T) map to path lengths 0..lmax and sort it by s."""
    order = np.argsort(sg)
    sg = sg[order][::s_step]
    p = p[order][::s_step]
    L = tg * C
    cols = np.where((L >= 0) & (L <= lmax))[0]
    cols = cols[np.argsort(L[cols])]
    m = p[:, cols]
    return {
        "ns": len(sg),
        "nx": len(cols),
        "s0": r3(sg[0]),
        "s1": r3(sg[-1]),
        "x0": r3(L[cols[0]]),
        "x1": r3(L[cols[-1]]),
        "data": b64(to_db8(m, ref)),
    }


def pts(pairs):
    return [[r3(s), r3(t * C)] for s, t in pairs]


def step_json(st, lmax, ref):
    k = st["kind"]
    out = {"kind": k}
    if "map" in st:
        r = ref if ref is not None else float(st["map"].max())
        out["map"] = encode_map(st["map"], np.asarray(st["s"]), np.asarray(st["tau"]), lmax, r, s_step=2 if len(st["s"]) > 200 else 1)
        if st.get("gate") is not None and st["gate"] > 0:
            out["gate"] = level(st["gate"], r)
    for key in ("picked", "found", "dropped"):
        if key in st:
            out[key] = pts(st[key])
    if st.get("grid") is not None:
        out["grid"] = pts([st["grid"]])[0]
    if st.get("refined") is not None:
        out["refined"] = pts([st["refined"]])[0]
    if k == "eig":
        out["ev"] = [r3(10 * np.log10(max(v, 1e-30))) for v in st["ev"]]
        out["thr"] = r3(10 * np.log10(st["thr"]))
        out["k"] = int(st["k"])
    if "tau_lines" in st:
        out["lines"] = [r3(t * C) for t in st["tau_lines"]]
    if k == "beams":
        b = st["beams"][::10]
        b = b / np.maximum(b.max(axis=0, keepdims=True), 1e-30)
        out["beams"] = {"ns": b.shape[0], "k": b.shape[1], "s0": r3(st["s"][0]), "s1": r3(st["s"][::10][-1]), "data": b64((b**2 * 255).round())}
    return out


def final_json(p, ap, room, d0):
    rel = relative_params(p)
    if rel is None:
        return {"los": None, "images": [], "lines": []}
    phi0, phis, dls = rel
    lines = walls_from_paths(ap.pos, phi0, d0, phis, dls)
    walls = room.walls()
    out_lines, used = [], set()
    for th, rho in lines:
        n = np.array([np.cos(th), np.sin(th)])
        best, best_c, err = None, np.inf, None
        for w in walls:
            th_t, _ = wall_line(ap.pos, w)
            de = abs(float(np.dot(w.mid - ap.pos, n)) - rho)
            ae = abs((th - th_t + np.pi) % (2 * np.pi) - np.pi)
            c = de / 0.5 + ae / np.deg2rad(10)
            if de <= 0.5 and ae <= np.deg2rad(10) and c < best_c:
                best, best_c, err = w.name, c, de
        tag = best
        if best is not None and (best in used or best == "south"):
            tag = "dup" if best != "south" else "south"
        elif best is not None:
            used.add(best)
        out_lines.append({"th": r3(th), "rho": r3(rho), "wall": tag, "err": r3(err) if err is not None and tag == best else None})
    imgs = [[r3(ap.pos[0] + (d0 + dl) * np.cos(ph)), r3(ap.pos[1] + (d0 + dl) * np.sin(ph))] for ph, dl in zip(phis, dls)]
    return {"los": [r3(ap.pos[0] + d0 * np.cos(phi0)), r3(ap.pos[1] + d0 * np.sin(phi0))], "images": imgs, "lines": out_lines}


def scene(seed):
    rng = np.random.default_rng(seed)
    room = Room(width=float(rng.uniform(3.5, 7.0)), depth=float(rng.uniform(3.0, 6.0)))
    ap = Array(np.array([float(rng.uniform(1.0, room.width - 1.0)), 0.1]), n_ant=ANTS)
    client = random_client(rng, room, ap.pos)
    paths = image_paths(room, ap, client, max_order=2)
    los = [q for q in paths if q.is_los][0]
    lmax = float(np.ceil(max(q.length for q in paths) + 2))
    sc = {
        "seed": seed,
        "room": [r3(room.width), r3(room.depth)],
        "ap": [r3(v) for v in ap.pos],
        "client": [r3(v) for v in client],
        "d0": r3(los.length),
        "lmax": lmax,
        "walls": [{"name": w.name, "p0": [r3(v) for v in w.p0], "p1": [r3(v) for v in w.p1]} for w in room.walls()],
        "paths": [
            {
                "label": "LoS" if q.is_los else "".join(NAMES[w] for w in q.walls),
                "order": q.order,
                "s": r3(q.u[0]),
                "L": r3(q.length),
                "img": [r3(q.image[0]), r3(q.image[1])],
                "power": r3(20 * np.log10(abs(q.gain) / abs(los.gain))),
            }
            for q in paths
        ],
        "runs": {},
    }
    for bw in BWS:
        y, n_k, s2 = measure(paths, ap, bw, SNR, np.random.default_rng(seed * 10 + bw))
        sc["runs"][str(bw)] = {}
        for name, fn in ESTIMATORS.items():
            tr: list = []
            p = fn(y, n_k, s2, ap.fc, trace=tr)
            greedy = name in ("omp", "nomp")
            ref = float(tr[0]["map"].max()) if greedy and tr and "map" in tr[0] else None
            sc["runs"][str(bw)][name] = {
                "steps": [step_json(st, lmax, ref) for st in tr],
                "final": final_json(p, ap, room, los.length),
            }
    return sc


def main():
    import json

    scenes = [scene(7000 + i) for i in range(SCENES)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"bws": BWS, "ants": ANTS, "snr": SNR, "dbRange": DB_RANGE, "scenes": scenes}, separators=(",", ":")))
    print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
