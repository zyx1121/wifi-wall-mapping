import numpy as np
import pytest

from wallmap.csi import measure, tone_offsets
from wallmap.estimators import ESTIMATORS, relative_params
from wallmap.geometry import Array, Room, image_paths, match_walls, oracle_params, walls_from_paths


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    room = Room()
    ap = Array(np.array([2.0, 0.1]), n_ant=4)
    client = np.array([rng.uniform(0.6, 4.4), rng.uniform(1.5, 3.4)])
    return rng, room, ap, client


def test_first_order_oracle_is_exact():
    """Oracle first-order parameters give each wall back to < 1 mm."""
    for seed in range(50):
        rng, room, ap, client = _setup(seed)
        paths = [p for p in image_paths(room, ap, client, max_order=1) if p.walls != ("south",)]
        phi0, d0, phis, dls, _ = oracle_params(paths)
        est = walls_from_paths(ap.pos, phi0, d0, phis, dls)
        m = match_walls(est, ap.pos, room)
        assert all(m.found.values())
        assert max(m.dist_err.values()) < 1e-3
        assert max(m.ang_err.values()) < 1e-3
        assert m.n_false == 0


def test_image_paths_counts():
    _, room, ap, client = _setup(1)
    p1 = image_paths(room, ap, client, max_order=1)
    assert len(p1) == 5  # LoS + 4 walls in a convex room
    p2 = image_paths(room, ap, client, max_order=2)
    assert len(p2) > 5
    p3 = image_paths(room, ap, client, max_order=1, floor_ceiling=True)
    assert len(p3) == 5 + 2  # plus floor and ceiling copies of the LoS


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_estimators_two_paths(name):
    """Two well separated paths at 40 dB come back within 1 deg and 5 cm."""
    rng, room, ap, client = _setup(3)
    ap = Array(np.array([2.0, 0.1]), n_ant=8)
    los = image_paths(room, ap, client, max_order=1)
    paths = [los[0]] + [p for p in los if p.walls == ("north",)]
    y, n_k, s2 = measure(paths, ap, 160, 40.0, rng)
    p = ESTIMATORS[name](y, n_k, s2, ap.fc)
    rel = relative_params(p)
    assert rel is not None
    phi0, phis, dls = rel
    t_phi0, _, t_phis, t_dls, _ = oracle_params(paths)
    assert abs(np.rad2deg(phi0 - t_phi0)) < 1.0
    j = int(np.argmin(np.abs(dls - t_dls[0])))
    on_grid = name in ("ifft", "omp")  # grid step is about 0.47 m and 2 deg at 160 MHz
    assert abs(dls[j] - t_dls[0]) < (0.3 if on_grid else 0.05)
    assert abs(np.rad2deg(phis[j] - t_phis[0])) < (3.0 if on_grid else 1.0)


def test_tone_counts():
    assert len(tone_offsets(80)) == 996
    assert len(tone_offsets(160)) == 1992
    assert len(tone_offsets(320)) == 3984


def test_rooms_from_true_walls():
    """The true walls of plan B close into its 5 rooms and 4 doors."""
    from wallmap.floorplan import apartment_b, walk
    from wallmap.rooms import Segment, close_rooms, score_rooms

    plan = apartment_b()
    pos = walk(waypoints=plan.waypoints)
    rm = close_rooms([Segment(w.p0, w.p1, 10) for w in plan.walls()], pos)
    s = score_rooms(rm, plan.boxes, plan.doors)
    assert s["rooms_ok"] == 5 and s["rooms_found"] == 5
    assert s["mean_iou"] > 0.9
    assert s["door_recall"] == 1.0
    # without walls the walk alone gives 1 region
    assert close_rooms([], pos).n_rooms == 1


def test_wall_materials_match_measurements():
    """P.2040 slabs land near measured losses and never gain energy."""
    from wallmap.materials import WALLS, _slab, coefficients, loss_db

    def slab_db(layers, f):
        return -20 * np.log10(abs(_slab(layers, f, 1.0)[1]))

    # Lott and Forkel, 5.2 GHz: 10 cm concrete 16 dB, 20 cm 29 dB
    assert abs(slab_db((("concrete", 0.10),), 5.2e9) - 16) < 4
    assert abs(slab_db((("concrete", 0.20),), 5.2e9) - 29) < 4
    # NYU, 6.75 GHz: plasterboard stud wall 2.1 to 3.0 dB
    assert abs(loss_db("drywall", 6.75e9) - 2.55) < 1.5
    # air is transparent
    r, t = _slab((("air", 0.1),), 6e9, 0.7)
    assert abs(r) < 1e-9 and abs(abs(t) - 1) < 1e-9
    for w in WALLS:
        for c in (1.0, 0.7, 0.3, 0.05):
            r, t = coefficients(w, 6.425e9, c)
            assert abs(r) ** 2 + abs(t) ** 2 <= 1 + 1e-9


def test_wall_loss_fit_recovers_losses():
    """The robust fit finds the offset and per-wall losses despite outliers."""
    from wallmap.attenuation import false_lines, fit_losses

    rng = np.random.default_rng(0)
    a = (rng.random((200, 3)) < 0.3).astype(float)
    true = np.array([4.0, 15.0, 0.0])
    e = -3.0 + a @ true + 0.5 * rng.standard_normal(200)
    e[:10] -= 12  # wrong direct paths
    c, loss, count = fit_losses(e, a)
    assert abs(c + 3.0) < 0.5
    assert np.all(np.abs(loss - true) < 0.7)
    assert false_lines(loss, count) == [2]
    # one offset per AP
    g = np.repeat([0, 1], 100)
    e2 = np.where(g == 0, -3.0, 5.0) + a @ true + 0.5 * rng.standard_normal(200)
    _, loss2, _ = fit_losses(e2, a, groups=g)
    assert np.all(np.abs(loss2 - true) < 0.7)


def test_ap_frame_keeps_paths():
    """An AP on the north wall sees the same path lengths in its own frame."""
    from wallmap.floorplan import MATERIALS_B, apartment_b
    from wallmap.multi_ap import ap_sites, frame, local_plan

    plan = apartment_b().with_materials(MATERIALS_B)
    site = [s for s in ap_sites(plan) if s.normal[1] < 0][0]
    f = frame(plan, site)
    client = np.array([4.4, 5.5])
    assert np.allclose(f.to_global(f.to_local(client)), client)
    ap_l = f.to_local(site.pos)
    assert ap_l[1] < f.to_local(client)[1]  # client in front of the AP
    g = sorted(round(q.length, 6) for q in image_paths(plan, Array(site.pos), client, max_order=1))
    lo = sorted(round(q.length, 6) for q in image_paths(local_plan(plan, f), Array(ap_l), f.to_local(client), max_order=1))
    assert g == lo


def test_ekf_beats_ftm_alone():
    """PDR plus angle and FTM tracks a walk better than FTM alone, start heading unknown."""
    from wallmap.floorplan import apartment_b, walk
    from wallmap.pdr import ImuNoise, simulate, track

    plan = apartment_b()
    ap = np.array([2.5, 0.1])
    true = walk(seed=7000, jitter=0.0, waypoints=plan.waypoints)
    rng = np.random.default_rng(1)
    lengths, headings, _ = simulate(true, ImuNoise(), rng)
    v = true - ap
    r = np.linalg.norm(v, axis=1)
    cos_m = v[:, 0] / r + 0.01 * rng.standard_normal(len(r))
    ftm = r + 0.3 * rng.standard_normal(len(r))
    est = track(ap, lengths, headings, cos_m, ftm)
    alone = ap + ftm[:, None] * np.stack([cos_m, np.sqrt(1 - np.clip(cos_m, -1, 1) ** 2)], axis=1)
    e_ekf = np.median(np.linalg.norm(est - true, axis=1))
    e_ftm = np.median(np.linalg.norm(alone - true, axis=1))
    assert e_ekf < 0.2
    assert e_ekf < 0.7 * e_ftm


def test_swiss_apartment_from_rows():
    """2 rooms, an inner wall with a door, a window: walls cut at openings, a route through both rooms."""
    from wallmap.floorplan import walk
    from wallmap.swiss import ap_sites, build, in_front, materials, waypoints

    def rect(x0, y0, x1, y1):
        return f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"

    rows = [
        {"entity_type": "separator", "entity_subtype": "WALL", "geometry": rect(-0.2, -0.2, 8.2, 0)},
        {"entity_type": "separator", "entity_subtype": "WALL", "geometry": rect(-0.2, 4, 8.2, 4.2)},
        {"entity_type": "separator", "entity_subtype": "WALL", "geometry": rect(-0.2, 0, 0, 4)},
        {"entity_type": "separator", "entity_subtype": "WALL", "geometry": rect(8, 0, 8.2, 4)},
        {"entity_type": "separator", "entity_subtype": "WALL", "geometry": rect(3.95, 0, 4.05, 4)},
        {"entity_type": "opening", "entity_subtype": "DOOR", "geometry": rect(3.95, 1.5, 4.05, 2.4)},
        {"entity_type": "opening", "entity_subtype": "WINDOW", "geometry": rect(1, -0.2, 2.5, 0)},
        {"entity_type": "area", "entity_subtype": "LIVING_ROOM", "geometry": rect(0, 0, 3.95, 4)},
        {"entity_type": "area", "entity_subtype": "BEDROOM", "geometry": rect(4.05, 0, 8, 4)},
    ]
    apt = build(rows)
    kinds = sorted(apt.kind.values())
    assert kinds.count("window") == 1 and "door" not in kinds
    inner = [n for n in apt.kind if n not in apt.outer]
    assert len(inner) == 2  # the inner wall, cut in 2 at the door
    mats = materials(apt, "swiss")
    assert all(m[0][0] == "plasterboard" for n, m in mats.items() if n in inner)
    pos = walk(waypoints=waypoints(apt), seed=0)
    assert len(apt.plan.boxes) == 2  # a wall with a door keeps 2 rooms
    for _, g in apt.rooms:
        lo, hi = g.min(axis=0), g.max(axis=0)
        assert np.any(np.all((pos > lo) & (pos < hi), axis=1))  # the route enters both rooms
    sites = ap_sites(apt, pos)
    assert len(sites) >= 1 and all(in_front(st, pos).mean() >= 0.8 for st in sites)
