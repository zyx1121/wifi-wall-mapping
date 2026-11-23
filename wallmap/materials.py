"""Wall reflection and transmission from ITU-R P.2040-2.

A wall is a stack of flat layers (material, thickness in metres) in air.
Each material has a complex relative permittivity

    eta = a f^b - j 17.98 c f^d / f        (f in GHz, Table 3, eq. 9b and 12)

The multi-layer method of section 2.2.2.1 (eq. 39 to 42) gives the
E-field reflection and transmission coefficients at any incidence angle.

The AP and the phone use vertical antennas and all walls are vertical, so
the E-field is normal to the plane of incidence: TE polarisation.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Table 3: a, b, c, d
TABLE3 = {
    "air": (1.0, 0.0, 0.0, 0.0),
    "concrete": (5.24, 0.0, 0.0462, 0.7822),
    "brick": (3.91, 0.0, 0.0238, 0.16),
    "plasterboard": (2.73, 0.0, 0.0085, 0.9395),
    "wood": (1.99, 0.0, 0.0047, 1.0718),
    "glass": (6.31, 0.0, 0.0036, 1.3394),
}

# wall types as layer stacks
WALLS: dict[str, tuple[tuple[str, float], ...]] = {
    "concrete": (("concrete", 0.15),),  # reinforced concrete outer wall
    "brick": (("brick", 0.12),),  # half-brick partition
    "rc10": (("concrete", 0.10),),  # reinforced concrete partition
    "drywall": (("plasterboard", 0.0125), ("air", 0.075), ("plasterboard", 0.0125)),  # stud partition
    "wood_door": (("wood", 0.045),),
    "glass": (("glass", 0.006),),
}


def permittivity(material: str, f_hz: float) -> complex:
    a, b, c, d = TABLE3[material]
    f = f_hz / 1e9
    return complex(a * f**b, -17.98 * c * f**d / f)


@lru_cache(maxsize=4096)
def _slab(layers: tuple[tuple[str, float], ...], f_hz: float, cos_i: float) -> tuple[complex, complex]:
    lam = 299_792_458.0 / f_hz
    eta = [1.0 + 0j] + [permittivity(m, f_hz) for m, _ in layers] + [1.0 + 0j]
    d = [0.0] + [t for _, t in layers] + [0.0]
    s0 = np.sqrt(max(0.0, 1.0 - cos_i**2))
    cos = [np.sqrt(1 - s0**2 / e + 0j) for e in eta]
    n = len(layers)
    a, b = 1.0 + 0j, 0.0 + 0j  # A_{N+1}, B_{N+1}
    for k in range(n, -1, -1):
        y = (cos[k + 1] / cos[k]) * np.sqrt(eta[k + 1] / eta[k])  # eq. 41b
        ph = 2 * np.pi / lam * np.sqrt(eta[k]) * d[k] * cos[k]
        a, b = (
            0.5 * np.exp(1j * ph) * (a * (1 + y) + b * (1 - y)),
            0.5 * np.exp(-1j * ph) * (a * (1 - y) + b * (1 + y)),
        )
    return complex(b / a), complex(1 / a)


def coefficients(wall, f_hz: float, cos_i: float) -> tuple[complex, complex]:
    """TE reflection and transmission of a wall type at cos(incidence angle).

    wall: a WALLS key, or a tuple of (material, thickness) layers.
    """
    layers = WALLS[wall] if isinstance(wall, str) else tuple(wall)
    # a path along the wall (cos 0) divides by 0 in eq. 41b
    return _slab(layers, float(f_hz), max(round(float(abs(cos_i)), 3), 1e-3))


def loss_db(wall, f_hz: float, cos_i: float = 1.0) -> float:
    """Transmission loss in dB (positive)."""
    return float(-20 * np.log10(abs(coefficients(wall, f_hz, cos_i)[1])))
