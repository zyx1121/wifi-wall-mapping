"""Wi-Fi CSI synthesis and hardware impairments.

Tone plan: HE/EHT subcarrier spacing 78.125 kHz with a full-band RU
(996, 2x996 and 4x996 tones for 80, 160 and 320 MHz). We place the tones as
one contiguous block around a 5-tone DC null. Real plans have extra nulls
between the 80 MHz segments. This is an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import C, Array, Path

DF = 78.125e3
USABLE_TONES = {80: 996, 160: 1992, 320: 3984}


def tone_offsets(bw_mhz: int) -> np.ndarray:
    half = USABLE_TONES[bw_mhz] // 2
    neg = np.arange(-half - 2, -2)
    pos = np.arange(3, half + 3)
    return np.concatenate([neg, pos])


def synth_csi(
    paths: list[Path],
    ap: Array,
    n_k: np.ndarray,
    snr_db: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """CSI matrix (n_ant x n_tones) with exact wideband array response.

    SNR is the mean |H|^2 over the noise power per CSI entry.
    Returns the noisy CSI and the noise power.
    """
    f = ap.fc + n_k * DF  # (N,)
    x = ap.element_x()  # (M,)
    tau = np.array([p.delay for p in paths])  # (P,)
    ux = np.array([p.u[0] for p in paths])
    g = np.array([p.gain for p in paths])
    # effective delay per path and element
    teff = tau[:, None] - x[None, :] * ux[:, None] / C  # (P, M)
    h = np.einsum("p,pmk->mk", g, np.exp(-2j * np.pi * teff[:, :, None] * f[None, None, :]))
    if snr_db is None:
        return h, 0.0
    p_sig = float(np.mean(np.abs(h) ** 2))
    if snr_1m_db is None:
        s2 = p_sig / 10 ** (snr_db / 10)
    else:
        s2 = (ap.wavelength / (4 * np.pi)) ** 2 / 10 ** (snr_1m_db / 10)
    noise = np.sqrt(s2 / 2) * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    return h + noise, s2


@dataclass
class Impairments:
    random_phase: bool = False  # common phase per packet (CFO, PLL)
    sto_ns: float = 0.0  # packet detection delay, uniform in [-sto, sto]
    ant_phase: str = "none"  # "none", "calibrated" (residual) or "uncalibrated"
    ant_phase_resid_deg: float = 5.0
    agc_db: float = 0.0  # random gain, uniform in [-agc, agc] dB
    rf_ripple: bool = False  # per-chain filter ripple, different per antenna
    quant_bits: int = 0  # 0 = no quantization

    def label(self) -> str:
        parts = []
        if self.random_phase:
            parts.append("phase")
        if self.sto_ns:
            parts.append(f"sto{self.sto_ns:g}ns")
        if self.ant_phase != "none":
            parts.append(f"ant-{self.ant_phase}")
        if self.agc_db:
            parts.append(f"agc{self.agc_db:g}dB")
        if self.rf_ripple:
            parts.append("ripple")
        if self.quant_bits:
            parts.append(f"q{self.quant_bits}")
        return "+".join(parts) or "ideal"


ALL_CALIBRATED = Impairments(True, 100.0, "calibrated", 5.0, 3.0, True, 8)


def impair_hardware(h: np.ndarray, n_k: np.ndarray, imp: Impairments, rng: np.random.Generator) -> np.ndarray:
    """Impairments that act before the noise is added (RF chain)."""
    m, n = h.shape
    out = h.copy()
    if imp.rf_ripple:
        k = (n_k - n_k.min()) / max(1, n_k.max() - n_k.min())
        for a in range(m):
            amp = np.ones(n)
            ph = np.zeros(n)
            for r in (1, 2, 3):
                amp += 0.03 * rng.standard_normal() * np.cos(2 * np.pi * r * k + rng.uniform(0, 2 * np.pi))
                ph += np.deg2rad(2.0) * rng.standard_normal() * np.sin(2 * np.pi * r * k + rng.uniform(0, 2 * np.pi))
            out[a] *= amp * np.exp(1j * ph)
    if imp.ant_phase == "uncalibrated":
        out *= np.exp(1j * rng.uniform(0, 2 * np.pi, m))[:, None]
    elif imp.ant_phase == "calibrated":
        out *= np.exp(1j * np.deg2rad(imp.ant_phase_resid_deg) * rng.standard_normal(m))[:, None]
    return out


def impair_receiver(y: np.ndarray, n_k: np.ndarray, imp: Impairments, rng: np.random.Generator) -> np.ndarray:
    """Impairments that act after the noise (timing, AGC, CSI report)."""
    out = y.copy()
    if imp.sto_ns:
        t = rng.uniform(-imp.sto_ns, imp.sto_ns) * 1e-9
        out *= np.exp(-2j * np.pi * n_k * DF * t)[None, :]
    if imp.random_phase:
        out *= np.exp(1j * rng.uniform(0, 2 * np.pi))
    if imp.agc_db:
        out *= 10 ** (rng.uniform(-imp.agc_db, imp.agc_db) / 20)
    if imp.quant_bits:
        levels = 2 ** (imp.quant_bits - 1) - 1
        scale = max(np.abs(out.real).max(), np.abs(out.imag).max())
        out = (np.round(out.real / scale * levels) + 1j * np.round(out.imag / scale * levels)) / levels * scale
    return out


def measure(
    paths: list[Path],
    ap: Array,
    bw_mhz: int,
    snr_db: float,
    rng: np.random.Generator,
    imp: Impairments | None = None,
    snr_1m_db: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One packet: CSI, tone offsets and noise power (before AGC).

    By default the noise sits snr_db below this link's own signal. With
    snr_1m_db the noise floor is fixed instead: a free-space link of 1 m
    has that SNR, so walls and distance lower the SNR of a link.
    """
    n_k = tone_offsets(bw_mhz)
    imp = imp or Impairments()
    h, _ = synth_csi(paths, ap, n_k, None, rng)
    h = impair_hardware(h, n_k, imp, rng)
    p_sig = float(np.mean(np.abs(h) ** 2))
    if snr_1m_db is None:
        s2 = p_sig / 10 ** (snr_db / 10)
    else:
        s2 = (ap.wavelength / (4 * np.pi)) ** 2 / 10 ** (snr_1m_db / 10)
    y = h + np.sqrt(s2 / 2) * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    y2 = impair_receiver(y, n_k, imp, rng)
    # the receiver knows its noise floor in the same units as the report
    gain2 = float(np.mean(np.abs(y2) ** 2) / np.mean(np.abs(y) ** 2))
    return y2, n_k, s2 * gain2
