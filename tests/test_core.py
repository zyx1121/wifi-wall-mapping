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
