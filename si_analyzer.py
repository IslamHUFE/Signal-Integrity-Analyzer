#!/usr/bin/env python3
"""SI Analyzer for Touchstone .s2p channels (dark UI).

The S-parameter data drives the plotted channel analysis. A synthetic eye is
calculated from S21 using the user-selected signal type, data rate and swing;
it is not a replacement for a standards-compliant transmitter/receiver or BER
simulator.

The eye is accumulated as a 2-D hit-density histogram (time x voltage) over
hundreds of thousands to millions of symbols and drawn like a spectrogram.
"""
from __future__ import annotations

import math
import re
import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import matplotlib as mpl
import numpy as np
import skrf as rf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.special import erfc

APP_TITLE = "SI Analyzer — Touchstone"

# ----------------------------------------------------------------------------
# Dark palette
# ----------------------------------------------------------------------------
BG = "#0b0f19"
PANEL = "#111827"
CARD = "#151c2c"
CARD_HI = "#1c2540"
BORDER = "#263149"
FG = "#e6eaf2"
MUTED = "#8b97ad"
ACCENT = "#4cc2ff"
ACCENT2 = "#ff9f5a"
GREEN = "#7ee787"
PURPLE = "#d2a8ff"
RED = "#ff6b6b"
PLOT_BG = "#0d1320"
AX_BG = "#0a0e17"
GRID = "#243049"

# Smith-chart trace: blue at the lowest frequency -> red at the highest.
FREQ_CMAP = LinearSegmentedColormap.from_list("freq_blue_red", ["#2f7bff", "#ff3b3b"])

mpl.rcParams.update({
    "figure.facecolor": PLOT_BG,
    "axes.facecolor": AX_BG,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": FG,
    "axes.titlecolor": FG,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": FG,
    "grid.color": GRID,
    "grid.alpha": 0.6,
    "legend.facecolor": CARD,
    "legend.edgecolor": BORDER,
    "legend.labelcolor": FG,
    "savefig.facecolor": PLOT_BG,
    "lines.linewidth": 1.6,
})


@dataclass
class ChannelData:
    """`s` / `z0` always hold the active 2-port (for a 4-port file: the differential Sdd block)."""
    path: Path
    frequency_hz: np.ndarray
    s: np.ndarray
    z0: np.ndarray
    ports: int = 2
    s_full: np.ndarray | None = None
    z0_full: np.ndarray | None = None
    noise: dict[str, Any] | None = None
    pairing: str = "1-2 / 3-4"
    note: str = ""


class NeedPortMap(Exception):
    """Raised when a file has a port count that needs a user-chosen mapping."""

    def __init__(self, nports: int) -> None:
        super().__init__(f"{nports}-port file needs a port mapping")
        self.nports = nports


def read_touchstone(path: str | Path, pairing: str = "1-2 / 3-4", port_map: tuple | None = None) -> ChannelData:
    """Load an N-port Touchstone file through scikit-rf.

    2-port files load directly; 4-port files default to the differential Sdd block of `pairing`.
    Any other file (or an explicit `port_map`) selects the channel by mapping:
    ("se", in_port, out_port) or ("diff", in+, in-, out+, out-), 0-based.
    """
    file_path = Path(path)
    network = rf.Network(str(file_path))
    n = network.nports
    if n < 2:
        raise ValueError("A Touchstone file with at least 2 ports is required.")
    f = np.asarray(network.f, dtype=float)
    s = np.asarray(network.s, dtype=complex)
    z0 = np.asarray(network.z0, dtype=complex)
    if len(f) < 3 or not np.all(np.isfinite(f)) or np.any(np.diff(f) <= 0):
        raise ValueError("The Touchstone file must contain at least three increasing frequency points.")
    if not np.all(np.isfinite(s)):
        raise ValueError("The S-parameter file contains non-finite values.")
    if z0.ndim == 1:
        z0 = np.repeat(z0[:, None], n, axis=1)
    if port_map is None and n not in (2, 4):
        raise NeedPortMap(n)
    noise = None
    if n == 2:
        try:
            noise = parse_touchstone_noise(file_path)
        except Exception:
            noise = None
    if port_map is not None:
        if port_map[0] == "se":
            i, j = int(port_map[1]), int(port_map[2])
            if i == j or max(i, j) >= n or min(i, j) < 0:
                raise ValueError("Invalid single-ended port selection.")
            sel = [i, j]
            return ChannelData(file_path, f, s[:, sel][:, :, sel], z0[:, sel], ports=n, z0_full=z0, noise=noise,
                               note=f"single-ended, port {i + 1} → port {j + 1}")
        order = [int(q) for q in port_map[1:5]]
        if len(set(order)) != 4 or max(order) >= n or min(order) < 0:
            raise ValueError("Invalid differential port selection.")
        sub = s[:, order][:, :, order]
        z_d = 2.0 * np.column_stack([z0[:, order[0]], z0[:, order[2]]])
        o = [q + 1 for q in order]
        return ChannelData(file_path, f, mixed_mode(sub, "1-2 / 3-4")["sdd"], z_d, ports=n, s_full=sub, z0_full=z0[:, order],
                           pairing="1-2 / 3-4", note=f"differential Sdd, ports ({o[0]},{o[1]}) → ({o[2]},{o[3]})")
    if n == 4:
        mm = mixed_mode(s, pairing)
        z0_diff = 2.0 * np.column_stack([z0[:, 0], z0[:, 2]])
        return ChannelData(file_path, f, mm["sdd"], z0_diff, ports=4, s_full=s, z0_full=z0, pairing=pairing)
    return ChannelData(file_path, f, s, z0, ports=2, noise=noise)


def impedance_from_reflection(gamma: np.ndarray, reference_z: np.ndarray) -> np.ndarray:
    """Convert a one-port reflection coefficient to impedance."""
    denominator = 1.0 - gamma
    with np.errstate(divide="ignore", invalid="ignore"):
        z = reference_z * (1.0 + gamma) / denominator
    z[np.abs(denominator) < 1e-12] = np.nan + 1j * np.nan
    return z


# ----------------------------------------------------------------------------
# Signal types for the eye diagram
# ----------------------------------------------------------------------------
# bps : information bits carried per symbol (UI = bps / data_rate)
# thr : decision thresholds, normalised so the outer levels are at +/-1
def _pam(levels: int) -> dict[str, Any]:
    lv = np.linspace(-1.0, 1.0, levels)
    return {"kind": "pam", "levels": levels, "bps": math.log2(levels), "thr": list(0.5 * (lv[:-1] + lv[1:]))}


SIGNALS: dict[str, dict[str, Any]] = {
    "NRZ (PAM2)": _pam(2),
    "PAM3": _pam(3),
    "PAM4": _pam(4),
    "PAM8": _pam(8),
    "RZ (50 % duty)": {"kind": "rz", "bps": 1.0, "thr": [-0.5, 0.5]},
    "Manchester": {"kind": "manchester", "bps": 0.5, "thr": [0.0]},
    "MLT-3": {"kind": "mlt3", "bps": 1.0, "thr": [-0.5, 0.5]},
    "Duobinary": {"kind": "duobinary", "bps": 1.0, "thr": [-0.5, 0.5]},
    "Clock (1010…)": {"kind": "clock", "bps": 1.0, "thr": [0.0]},
    "PRBS7 (NRZ)": {"kind": "prbs", "order": 7, "bps": 1.0, "thr": [0.0]},
    "PRBS9 (NRZ)": {"kind": "prbs", "order": 9, "bps": 1.0, "thr": [0.0]},
    "PRBS15 (NRZ)": {"kind": "prbs", "order": 15, "bps": 1.0, "thr": [0.0]},
}
SIGNAL_CHOICES = list(SIGNALS.keys())
DEFAULT_SIGNAL = SIGNAL_CHOICES[0]

_PRBS_TAPS = {7: (7, 6), 9: (9, 5), 15: (15, 14)}


@lru_cache(maxsize=None)
def _prbs_sequence(order: int) -> np.ndarray:
    """One full period of a maximal-length LFSR sequence (x^n + x^k + 1)."""
    t1, t2 = _PRBS_TAPS[order]
    length = (1 << order) - 1
    mask = length
    state = mask
    out = np.empty(length, dtype=np.int8)
    for i in range(length):
        new = ((state >> (t1 - 1)) ^ (state >> (t2 - 1))) & 1
        out[i] = new
        state = ((state << 1) | new) & mask
    return out


def _generate_symbols(spec: dict[str, Any], rng: np.random.Generator, n: int) -> np.ndarray:
    """Return n symbol levels in [-1, 1] for the selected signal type."""
    kind = spec["kind"]
    if kind == "pam":
        lv = np.linspace(-1.0, 1.0, spec["levels"])
        return lv[rng.integers(0, spec["levels"], size=n)]
    if kind == "rz":
        return 2.0 * rng.integers(0, 2, size=n) - 1.0
    if kind == "manchester":
        b = 2.0 * rng.integers(0, 2, size=n // 2) - 1.0
        return np.column_stack([b, -b]).ravel()
    if kind == "mlt3":
        bits = rng.integers(0, 2, size=n)
        return np.array([0.0, 1.0, 0.0, -1.0])[np.cumsum(bits) % 4]
    if kind == "duobinary":
        b = 2.0 * rng.integers(0, 2, size=n) - 1.0
        return 0.5 * (b + np.roll(b, 1))
    if kind == "clock":
        return np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    if kind == "prbs":
        seq = _prbs_sequence(spec["order"])
        start = int(rng.integers(len(seq)))
        return 2.0 * seq[(start + np.arange(n)) % len(seq)] - 1.0
    raise ValueError(f"Unknown signal kind: {kind}")


# ----------------------------------------------------------------------------
# Eye / jitter engine (block-wise, constant memory, any number of symbols)
# ----------------------------------------------------------------------------
def _zero_crossings(rx: np.ndarray, oversample: int) -> np.ndarray:
    """Return crossing positions (in samples, modulo one UI) of the zero threshold."""
    y0, y1 = rx[:-1], rx[1:]
    idx = np.flatnonzero(((y0 < 0.0) & (y1 >= 0.0)) | ((y0 > 0.0) & (y1 <= 0.0)))
    if len(idx) == 0:
        return np.empty(0)
    denom = rx[idx + 1] - rx[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.abs(denom) > 1e-15, -rx[idx] / denom, 0.0)
    return np.mod(idx + frac, oversample)


def _eye_openings(dens: np.ndarray, centers: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Per-sub-eye, per-column opening (0.1 % tails): array of shape (n_eyes, columns)."""
    edges = np.concatenate([[-np.inf], np.sort(thresholds), [np.inf]])
    rows = []
    for k in range(1, len(edges) - 1):
        lo = (centers > edges[k - 1]) & (centers <= edges[k])
        up = (centers > edges[k]) & (centers <= edges[k + 1])
        if not lo.any() or not up.any():
            continue
        d_lo, d_up = dens[:, lo], dens[:, up]
        cum_up = np.cumsum(d_up, axis=1) / np.maximum(d_up.sum(axis=1, keepdims=True), 1.0)
        cum_lo = np.cumsum(d_lo[:, ::-1], axis=1) / np.maximum(d_lo.sum(axis=1, keepdims=True), 1.0)
        up_edge = centers[up][(cum_up >= 1e-3).argmax(axis=1)]
        lo_edge = centers[lo][::-1][(cum_lo >= 1e-3).argmax(axis=1)]
        rows.append(up_edge - lo_edge)
    return np.array(rows)


def _statistical_eye_height(dens: np.ndarray, centers: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Per-column eye opening; for multi-level signals the smallest sub-eye wins."""
    ops = _eye_openings(dens, centers, thresholds)
    return ops.min(axis=0) if len(ops) else np.zeros(dens.shape[0])


def ctle_response(freq: np.ndarray, boost_db: float, peak_hz: float) -> np.ndarray:
    """One-zero / two-pole CTLE normalised to unity gain at its peak frequency (low frequencies are attenuated)."""
    if boost_db <= 0 or peak_hz <= 0:
        return np.ones(len(freq), dtype=complex)
    g = 10.0 ** (boost_db / 20.0)
    fz, fp1 = peak_hz / math.sqrt(g), peak_hz * math.sqrt(g)
    fp2 = 2.5 * fp1

    def raw(x: Any) -> Any:
        return (1.0 + 1j * x / fz) / ((1.0 + 1j * x / fp1) * (1.0 + 1j * x / fp2))

    return raw(freq) / np.abs(raw(np.asarray(peak_hz, dtype=float)))


def apply_ffe(wave: np.ndarray, os_: int, pre: float, post: float) -> np.ndarray:
    """3-tap TX FFE [-pre, 1-|pre|-|post|, -post] applied on a symbol-rate waveform (circular)."""
    if pre == 0.0 and post == 0.0:
        return wave
    return (1.0 - abs(pre) - abs(post)) * wave - pre * np.roll(wave, -os_) - post * np.roll(wave, os_)


def _pulse_response(h_i: np.ndarray, n: int, os_: int, amp: float, kind: str, ffe: tuple[float, float],
                    pre_ui: int = 8) -> tuple[np.ndarray, int]:
    """Single-symbol response of TX FFE + channel + CTLE. Returns (response rolled by pre_ui UI, peak index)."""
    pulse = np.zeros(n)
    pulse[:os_] = amp
    if kind == "rz":
        pulse[os_ // 2:os_] = 0.0
    pulse = apply_ffe(pulse, os_, *ffe)
    p = np.roll(np.fft.irfft(np.fft.rfft(pulse) * h_i, n=n), pre_ui * os_)
    return p, int(np.argmax(np.abs(p)))


def make_eye_and_jitter(
    frequency_hz: np.ndarray,
    s21: np.ndarray,
    bit_rate_gbps: float,
    swing_vpp: float,
    total_bits: int = 262_144,
    oversample: int = 64,
    block_bits: int | None = None,
    v_bins: int = 400,
    seed: int = 17,
    signal: str = DEFAULT_SIGNAL,
    progress: Callable[[float], None] | None = None,
    ffe_pre: float = 0.0,
    ffe_post: float = 0.0,
    ctle_boost_db: float = 0.0,
    ctle_peak_ghz: float = 0.0,
    dfe_taps: int = 0,
    tx_rise_ps: float | None = None,
) -> dict[str, Any]:
    """Filter random symbol blocks with measured S21 and accumulate a 2-UI eye density.

    Optional equalisation: TX FFE (pre/post fractions), RX CTLE (boost dB at peak frequency) and an
    ideal-decision DFE (post-cursor taps taken from the equalised pulse response).
    Each block is an independent random sequence (periodic within the block), so memory stays constant
    while the number of observed symbols scales to millions. Below the first S21 frequency the first value
    is held; above the last point S21 is zero with a raised-cosine taper over the top 10 %.
    No noise, TX rise-time, RX model, clock recovery, or BER extrapolation is applied here.
    """
    if bit_rate_gbps <= 0 or swing_vpp <= 0:
        raise ValueError("Data rate and signal swing must both be greater than zero.")
    if total_bits < 1024:
        raise ValueError("Use at least 1,024 symbols for a meaningful eye view.")
    if signal not in SIGNALS:
        raise ValueError(f"Unknown signal type: {signal}")
    if abs(ffe_pre) + abs(ffe_post) >= 1.0:
        raise ValueError("FFE pre + post taps must stay below 100 %.")
    spec = SIGNALS[signal]

    f = np.asarray(frequency_hz, dtype=float)
    h = np.asarray(s21, dtype=complex)
    block_bits = int(min(block_bits or max(2048, 1_048_576 // oversample), total_bits))  # constant memory for any oversampling
    block_bits += block_bits % 2  # even, needed for Manchester
    n_blocks = max(1, int(math.ceil(total_bits / block_bits)))
    actual_bits = n_blocks * block_bits

    baud_gbd = bit_rate_gbps / spec["bps"]
    ui_s = 1.0 / (baud_gbd * 1e9)
    ui_ps = ui_s * 1e12
    fs = oversample / ui_s
    n = block_bits * oversample
    two_ui = 2 * oversample

    # Channel response on the rFFT grid.
    rfreq = np.fft.rfftfreq(n, d=1.0 / fs)
    measured_max = float(f[-1])
    h_i = np.interp(rfreq, f, h.real, left=h.real[0], right=0.0) + 1j * np.interp(
        rfreq, f, h.imag, left=h.imag[0], right=0.0
    )
    taper_start = 0.90 * measured_max
    taper = np.ones_like(rfreq)
    edge = (rfreq > taper_start) & (rfreq < measured_max)
    taper[edge] = 0.5 * (1.0 + np.cos(np.pi * (rfreq[edge] - taper_start) / (measured_max - taper_start)))
    taper[rfreq >= measured_max] = 0.0
    h_i = h_i * taper

    ctle_peak_hz = ctle_peak_ghz * 1e9 if ctle_peak_ghz > 0 else 0.5 / ui_s
    h_i = h_i * ctle_response(rfreq, ctle_boost_db, ctle_peak_hz)
    # Finite TX edge (no real driver has an infinite-bandwidth edge): Gaussian, 10-90 % rise time.
    # None = automatic 25 % of the UI; 0 = ideal (infinitely fast) edge.
    tx_rise_eff = 0.25 * ui_ps if tx_rise_ps is None else max(float(tx_rise_ps), 0.0)
    if tx_rise_eff > 0:
        h_i = h_i * np.exp(-2.0 * (np.pi * rfreq * tx_rise_eff * 1e-12 / 2.563) ** 2)

    amp = swing_vpp / 2.0
    ffe = (float(ffe_pre), float(ffe_post))
    pre_ui = 8
    p_c, d_pk_c = _pulse_response(h_i, n, oversample, amp, spec["kind"], ffe, pre_ui)
    d_pk = d_pk_c - pre_ui * oversample  # position of the main cursor inside one block

    def cursor(k: int) -> float:
        return float(p_c[(d_pk_c + k * oversample) % n])

    n_dfe = int(max(0, dfe_taps))
    dfe_v = np.array([cursor(k) for k in range(1, n_dfe + 1)])

    rng = np.random.default_rng(seed)
    hist = np.zeros(two_ui * v_bins, dtype=np.int64)
    hist_vis = np.zeros(two_ui * (v_bins + 1))  # display-only density: the trace drawn between samples
    t_off = (np.arange(two_ui, dtype=np.int64) * v_bins)[None, :]
    deviations: list[np.ndarray] = []
    center = 0.0
    shift = 0
    vmax = 1.0
    thresholds_v = np.array([0.0])
    cross_v = 0.0

    for b in range(n_blocks):
        sym = _generate_symbols(spec, rng, block_bits)
        tx = np.repeat(sym * amp, oversample)
        if spec["kind"] == "rz":
            tx.reshape(block_bits, oversample)[:, oversample // 2:] = 0.0
        tx = apply_ffe(tx, oversample, *ffe)
        rx = np.fft.irfft(np.fft.rfft(tx) * h_i, n=n)
        if n_dfe:
            corr_sym = np.zeros(block_bits)
            for k, c in enumerate(dfe_v, start=1):
                corr_sym += c * np.roll(sym, k)
            rx = rx - np.roll(np.repeat(corr_sym, oversample), d_pk - oversample // 2)

        if b == 0:
            # Decision thresholds: midpoints between the mean sampled voltage of each transmitted level.
            sampled = rx[(np.arange(block_bits) * oversample + d_pk) % n]
            lv = np.unique(np.round(sym, 6))
            means = np.sort([float(sampled[np.isclose(sym, v)].mean()) for v in lv])
            if len(means) >= 2:
                thresholds_v = 0.5 * (means[:-1] + means[1:])
            cross_v = float(thresholds_v[len(thresholds_v) // 2])
        phases = _zero_crossings(rx - cross_v, oversample)
        if b == 0:
            if len(phases):
                ang = phases * (2.0 * np.pi / oversample)
                mean_ang = math.atan2(float(np.mean(np.sin(ang))), float(np.mean(np.cos(ang))))
                center = (mean_ang % (2.0 * np.pi)) * oversample / (2.0 * np.pi)
            shift = int(round(center)) % oversample
            vmax = max(1.1 * float(np.max(np.abs(rx))), 0.05 * swing_vpp)
        if len(phases):
            dev = (phases - center + oversample / 2.0) % oversample - oversample / 2.0
            deviations.append(dev.astype(np.float32))

        # Roll so threshold crossings land at 0 / 1 / 2 UI, eye centre at 0.5 / 1.5 UI.
        r2 = np.roll(rx, -shift).reshape(block_bits, oversample)
        win = np.concatenate([r2, np.roll(r2, -1, axis=0), np.roll(r2, -2, axis=0)[:, :1]], axis=1)
        vi_all = np.clip(((win + vmax) / (2.0 * vmax) * v_bins).astype(np.int64), 0, v_bins - 1)
        vi = vi_all[:, :two_ui]
        hist += np.bincount((vi + t_off).ravel(), minlength=two_ui * v_bins)
        # Like a persistence scope, connect consecutive samples so steep edges are drawn as continuous
        # curves instead of isolated dots (each trace column keeps a total weight of one).
        lo_b = np.minimum(vi_all[:, :-1], vi_all[:, 1:])
        hi_b = np.maximum(vi_all[:, :-1], vi_all[:, 1:])
        wgt = (1.0 / (hi_b - lo_b + 1)).ravel()
        base = np.arange(two_ui, dtype=np.int64)[None, :] * (v_bins + 1)
        size = two_ui * (v_bins + 1)
        hist_vis += (np.bincount((base + lo_b).ravel(), weights=wgt, minlength=size)
                     - np.bincount((base + hi_b + 1).ravel(), weights=wgt, minlength=size))

        if progress is not None:
            progress((b + 1) / n_blocks)

    dens = hist.reshape(two_ui, v_bins).astype(float)
    dens_vis = np.maximum(np.cumsum(hist_vis.reshape(two_ui, v_bins + 1), axis=1)[:, :v_bins], 0.0)
    v_edges = np.linspace(-vmax, vmax, v_bins + 1)
    centers = 0.5 * (v_edges[:-1] + v_edges[1:])

    openings = _eye_openings(dens, centers, thresholds_v)
    height = openings.min(axis=0) if len(openings) else np.zeros(two_ui)
    best_col = int(np.argmax(height[:oversample]))

    if deviations:
        dev_all = np.concatenate(deviations)
        rms_ps = float(np.std(dev_all) / oversample * ui_ps)
        q01, q99 = np.percentile(dev_all, [0.1, 99.9])
        pp_ps = float((q99 - q01) / oversample * ui_ps)
        crossings = int(len(dev_all))
    else:
        rms_ps = pp_ps = float("nan")
        crossings = 0

    ks = np.arange(-8, 25)
    fine = np.arange(-8 * oversample, 25 * oversample)
    eq_text = ((f"TX rise {tx_rise_eff:.1f} ps{' (auto, 25 % UI)' if tx_rise_ps is None else ''}, " if tx_rise_eff > 0 else "TX edge ideal, ")
               + f"TX FFE pre {ffe_pre*100:.0f} % / post {ffe_post*100:.0f} %, "
               + (f"CTLE {ctle_boost_db:g} dB @ {ctle_peak_hz/1e9:.3g} GHz, " if ctle_boost_db > 0 else "CTLE off, ")
               + (f"DFE {n_dfe} tap(s)" if n_dfe else "DFE off"))
    return {
        "density": dens,
        "density_vis": dens_vis,
        "v_min": -vmax,
        "v_max": vmax,
        "ui_ps": ui_ps,
        "oversample": oversample,
        "bits": actual_bits,
        "sample_phase_ui": best_col / oversample,
        "eye_height_v": float(max(0.0, height[best_col])),
        "eye_height_raw_v": float(height[best_col]),
        "eye_width_ps": float(max(0.0, ui_ps - pp_ps)) if np.isfinite(pp_ps) else float("nan"),
        "jitter_rms_ps": rms_ps,
        "jitter_pp_ps": pp_ps,
        "crossing_count": crossings,
        "bit_rate_gbps": bit_rate_gbps,
        "baud_gbd": baud_gbd,
        "swing_vpp": swing_vpp,
        "signal": signal,
        "thresholds_v": thresholds_v,
        "n_eyes": len(thresholds_v),
        "eye_heights_v": openings[:, best_col] if len(openings) else np.zeros(1),
        "eq_text": eq_text,
        "dfe_n": n_dfe,
        "dfe_taps_v": dfe_v,
        "pulse": {"x_ui": fine / oversample, "y": p_c[(d_pk_c + fine) % n], "k": ks,
                  "cursors": np.array([cursor(int(k)) for k in ks])},
    }


# -- BER / bathtub analysis and eye masks (work on the stored eye density) -------------
def ber_analysis(r: dict[str, Any], sigma_v: float, rj_ps: float, target: float) -> dict[str, Any]:
    """Statistical BER estimate from the eye density plus Gaussian voltage noise and random jitter.

    Horizontal bathtub uses all decision thresholds; the vertical bathtub and the contour use the centre
    threshold. For multi-level signals the result is a symbol-error estimate, not a bit-error rate.
    """
    dens = r["density"]
    os_ = r["oversample"]
    nb = dens.shape[1]
    edges_v = np.linspace(r["v_min"], r["v_max"], nb + 1)
    centers = 0.5 * (edges_v[:-1] + edges_v[1:])
    p = 0.5 * (dens[:os_] + dens[os_:2 * os_])
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1.0)
    sigma = max(float(sigma_v), 1e-9)
    thr = np.sort(np.asarray(r["thresholds_v"], dtype=float))
    bounds = np.concatenate([[-np.inf], thr, [np.inf]])

    def q(x: np.ndarray) -> np.ndarray:
        return 0.5 * erfc(x / math.sqrt(2.0))

    sj = rj_ps / r["ui_ps"] * os_
    ber_t = np.zeros(os_)
    for k in range(1, len(bounds) - 1):
        lo = (centers > bounds[k - 1]) & (centers <= bounds[k])
        up = (centers > bounds[k]) & (centers <= bounds[k + 1])
        ber_t += p[:, up] @ q((centers[up] - thr[k - 1]) / sigma) + p[:, lo] @ q((thr[k - 1] - centers[lo]) / sigma)
    if sj > 0.05:
        ber_t = gaussian_filter1d(ber_t, sj, mode="wrap")
    ber_t = np.clip(ber_t, 1e-18, 0.5)
    x_ui = (np.arange(os_) + 0.5) / os_

    def opening(ok: np.ndarray, i0: int) -> tuple[int, int]:
        if not ok[i0]:
            return i0, i0 - 1
        lft = rgt = i0
        while lft > 0 and ok[lft - 1]:
            lft -= 1
        while rgt < len(ok) - 1 and ok[rgt + 1]:
            rgt += 1
        return lft, rgt

    l, rr = opening(ber_t < target, int(np.argmin(ber_t)))
    width_ui = max(0, rr - l + 1) / os_

    t_mid = thr[len(thr) // 2]
    y = np.linspace(r["v_min"], r["v_max"], 161)
    lo, up = centers <= t_mid, centers > t_mid
    q_lo = q((y[None, :] - centers[lo][:, None]) / sigma)
    q_up = q((centers[up][:, None] - y[None, :]) / sigma)
    contour = p[:, lo] @ q_lo + p[:, up] @ q_up
    if sj > 0.05:
        contour = gaussian_filter1d(contour, sj, axis=0, mode="wrap")
    contour = np.clip(contour, 1e-18, 0.5)
    col = min(int(round(r["sample_phase_ui"] * os_)), os_ - 1)
    ber_y = contour[col]
    yl, yr = opening(ber_y < target, int(np.argmin(ber_y)))
    height_v = (yr - yl + 1) * float(y[1] - y[0]) if yr >= yl else 0.0
    return {"x_ui": x_ui, "ber_t": ber_t, "y": y, "ber_y": ber_y, "contour": contour,
            "width_ui": width_ui, "width_ps": width_ui * r["ui_ps"], "height_v": height_v, "target": target}


def mask_polygon(c_ui: float, c_v: float, h_v: float, w_ui: float, top: float = 0.5) -> np.ndarray:
    a, b, h = w_ui / 2.0, w_ui / 2.0 * top, h_v / 2.0
    return np.array([[c_ui - a, c_v], [c_ui - b, c_v + h], [c_ui + b, c_v + h],
                     [c_ui + a, c_v], [c_ui + b, c_v - h], [c_ui - b, c_v - h]])


def mask_check(r: dict[str, Any], h_v: float, w_ui: float) -> tuple[list[np.ndarray], int, bool]:
    """Hexagonal eye mask on every sub-eye of both displayed UIs; PASS means zero hits inside."""
    dens = r["density"]
    os_ = r["oversample"]
    nb = dens.shape[1]
    edges_v = np.linspace(r["v_min"], r["v_max"], nb + 1)
    centers = 0.5 * (edges_v[:-1] + edges_v[1:])
    polys = [mask_polygon(c, float(t), h_v, w_ui) for c in (0.5, 1.5) for t in r["thresholds_v"]]
    xx, yy = np.meshgrid((np.arange(2 * os_) + 0.5) / os_, centers, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        inside |= MplPath(poly).contains_points(pts)
    hits = int(dens.ravel()[inside].sum())
    return polys, hits, hits == 0


# ----------------------------------------------------------------------------
# Advanced analysis helpers (pure numpy, no GUI)
# ----------------------------------------------------------------------------
SPEED_OF_LIGHT = 299_792_458.0
PAIRINGS = {"1-2 / 3-4": [0, 1, 2, 3], "1-3 / 2-4": [0, 2, 1, 3]}
WINDOWS = ["Hann", "Kaiser (β=6)", "Rectangular"]


def _db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-15))


def _pdb(x: np.ndarray) -> np.ndarray:
    """Power ratio -> dB, NaN where the ratio is not a positive finite number."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x > 0)
    out[ok] = 10.0 * np.log10(x[ok])
    return out


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched linear solve that returns NaN for singular frequency points instead of raising."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        out = np.full(np.broadcast(a, b).shape, np.nan, dtype=complex)
        for i in range(a.shape[0]):
            try:
                out[i] = np.linalg.solve(a[i], b[i])
            except np.linalg.LinAlgError:
                pass
        return out


# -- group delay ---------------------------------------------------------------
def group_delay(f: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Group delay in seconds: -d(phase)/d(omega)."""
    phase = np.unwrap(np.angle(s))
    return -np.gradient(phase, 2.0 * np.pi * np.asarray(f, dtype=float))


# -- stability factors and power gains ----------------------------------------
def stability_metrics(s: np.ndarray) -> dict[str, np.ndarray]:
    s11, s12, s21, s22 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
    delta = s11 * s22 - s12 * s21
    p = np.abs(s12 * s21)
    with np.errstate(all="ignore"):
        k = (1.0 - np.abs(s11) ** 2 - np.abs(s22) ** 2 + np.abs(delta) ** 2) / (2.0 * p)
        mu = (1.0 - np.abs(s11) ** 2) / (np.abs(s22 - delta * np.conj(s11)) + p)
        mu_p = (1.0 - np.abs(s22) ** 2) / (np.abs(s11 - delta * np.conj(s22)) + p)
        msg = np.abs(s21) / np.abs(s12)
        mag = np.where(k > 1.0, msg * (k - np.sqrt(np.maximum(k * k - 1.0, 0.0))), np.nan)
        ratio = s21 / s12
        u = np.abs(ratio - 1.0) ** 2 / (2.0 * k * np.abs(ratio) - 2.0 * np.real(ratio))
    return {"K": k, "delta": delta, "mu": mu, "mu_p": mu_p, "MSG": msg, "MAG": mag, "U": u}


def power_gains(s: np.ndarray, gs: complex, gl: complex) -> dict[str, np.ndarray]:
    """Transducer (GT), available (GA) and operating (GP) power gain for given terminations."""
    s11, s12, s21, s22 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
    with np.errstate(all="ignore"):
        gin = s11 + s12 * s21 * gl / (1.0 - s22 * gl)
        gout = s22 + s12 * s21 * gs / (1.0 - s11 * gs)
        gt = (1 - abs(gs) ** 2) * np.abs(s21) ** 2 * (1 - abs(gl) ** 2) / (
            np.abs(1 - gs * gin) ** 2 * np.abs(1 - s22 * gl) ** 2)
        ga = np.abs(s21) ** 2 * (1 - abs(gs) ** 2) / (np.abs(1 - s11 * gs) ** 2 * (1 - np.abs(gout) ** 2))
        gp = np.abs(s21) ** 2 * (1 - abs(gl) ** 2) / ((1 - np.abs(gin) ** 2) * np.abs(1 - s22 * gl) ** 2)
    return {"GT": gt, "GA": ga, "GP": gp}


def passivity(s: np.ndarray) -> np.ndarray:
    """Largest singular value of S at each frequency (<= 1 for a passive network)."""
    return np.linalg.svd(s, compute_uv=False)[:, 0]


# -- parameter conversions ------------------------------------------------------
def convert_parameters(s: np.ndarray, z0: float) -> dict[str, np.ndarray]:
    """S -> Z, Y, ABCD, T, H, G for a 2-port with equal real reference impedance z0."""
    n = s.shape[0]
    eye = np.broadcast_to(np.eye(2), (n, 2, 2))
    z = z0 * _solve(eye - s, eye + s)
    y = _solve(eye + s, eye - s) / z0
    s11, s12, s21, s22 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
    a = np.empty_like(s)
    t = np.empty_like(s)
    h = np.empty_like(s)
    with np.errstate(all="ignore"):
        den = 2.0 * s21
        a[:, 0, 0] = ((1 + s11) * (1 - s22) + s12 * s21) / den
        a[:, 0, 1] = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / den
        a[:, 1, 0] = ((1 - s11) * (1 - s22) - s12 * s21) / den / z0
        a[:, 1, 1] = ((1 - s11) * (1 + s22) + s12 * s21) / den
        det_s = s11 * s22 - s12 * s21
        t[:, 0, 0] = -det_s / s21
        t[:, 0, 1] = s11 / s21
        t[:, 1, 0] = -s22 / s21
        t[:, 1, 1] = 1.0 / s21
        y11, y12, y21, y22 = y[:, 0, 0], y[:, 0, 1], y[:, 1, 0], y[:, 1, 1]
        h[:, 0, 0] = 1.0 / y11
        h[:, 0, 1] = -y12 / y11
        h[:, 1, 0] = y21 / y11
        h[:, 1, 1] = (y11 * y22 - y12 * y21) / y11
    g = _solve(h, eye)
    return {"S": s, "Z": z, "Y": y, "ABCD": a, "T": t, "H": h, "G": g}


# -- time domain (TDR / TDT) ----------------------------------------------------
def time_response(f: np.ndarray, h: np.ndarray, rise_s: float, window: str = "Hann",
                  n_freq: int = 8192, ext: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step and impulse response of a measured transfer/reflection function.

    DC is extrapolated by holding the first point; data above the last point is zero
    (the grid is extended `ext`x to interpolate in time). A Gaussian edge with the given
    10-90 % rise time shapes the step. Returns (t, step, impulse) over the unaliased half.
    """
    f = np.asarray(f, dtype=float)
    h = np.asarray(h, dtype=complex)
    fmax = float(f[-1])
    f_ext = ext * fmax
    fg = np.linspace(0.0, f_ext, n_freq)
    hg = np.interp(fg, f, h.real, left=h.real[0], right=0.0) + 1j * np.interp(fg, f, h.imag, left=0.0, right=0.0)
    k = int(np.searchsorted(fg, fmax, side="right"))
    w = np.zeros(n_freq)
    if window.startswith("Hann"):
        w[:k] = 0.5 * (1.0 + np.cos(np.pi * fg[:k] / fmax))
    elif window.startswith("Kaiser"):
        w[:k] = np.kaiser(2 * k, 6.0)[k:]
    else:
        w[:k] = 1.0
    if rise_s > 0:
        sigma = rise_s / 2.563  # Gaussian edge: 10-90 % rise = 2.563 sigma
        w = w * np.exp(-2.0 * (np.pi * fg * sigma) ** 2)
    x = np.fft.irfft(hg * w, n=2 * (n_freq - 1))
    dt = 1.0 / (2.0 * f_ext)
    half = len(x) // 2
    t = np.arange(len(x)) * dt
    return t[:half], np.cumsum(x)[:half], (x / dt)[:half]


def impedance_from_step(rho: np.ndarray, z0: float) -> np.ndarray:
    rho = np.clip(np.real(rho), -0.98, 0.98)
    return z0 * (1.0 + rho) / (1.0 - rho)


# -- equivalent circuit / RLGC ----------------------------------------------------
def extract_rlgc(f: np.ndarray, s: np.ndarray, z0: float, length_m: float) -> dict[str, np.ndarray]:
    """Per-unit-length R, L, G, C, Zc, gamma from the ABCD matrix of a uniform line of known length."""
    abcd = convert_parameters(s, z0)["ABCD"]
    a, b, c, d = abcd[:, 0, 0], abcd[:, 0, 1], abcd[:, 1, 0], abcd[:, 1, 1]
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    with np.errstate(all="ignore"):
        x = 0.5 * (a + d)
        root = np.sqrt(x * x - 1.0 + 0j)
        l1, l2 = x - root, x + root
        lam = np.where(np.abs(l1) <= np.abs(l2), l1, l2)  # e^{-gamma*l}
        lam = np.where(np.abs(lam) < 1e-12, 1e-12 + 0j, lam)
        alpha_l = -np.log(np.abs(lam))
        beta_l = -np.unwrap(np.angle(lam))
        beta_l = beta_l + 2.0 * np.pi * np.round(-beta_l[0] / (2.0 * np.pi))
        gamma = (alpha_l + 1j * beta_l) / length_m
        zc = np.sqrt(b / c)
        zc = np.where(np.real(zc) < 0, -zc, zc)
        zp, yp = gamma * zc, gamma / zc
        eps_eff = (np.imag(gamma) * SPEED_OF_LIGHT / w) ** 2
    return {"r": zp.real, "l": zp.imag / w, "g": yp.real, "c": yp.imag / w, "zc": zc, "gamma": gamma, "eps_eff": eps_eff}


def fit_pi_model(f: np.ndarray, s: np.ndarray, z0: float, frac: float = 0.25) -> dict[str, Any]:
    """Lumped pi model (series R+L, shunt G||C at both ends) fitted to the low part of the band."""
    y = convert_parameters(s, z0)["Y"]
    ys = -y[:, 0, 1]
    y1 = y[:, 0, 0] + y[:, 0, 1]
    y2 = y[:, 1, 1] + y[:, 0, 1]
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    k = min(len(f), max(5, int(round(frac * len(f)))))
    ww = w[:k]

    def slope(v: np.ndarray) -> float:
        ok = np.isfinite(v)
        return float(np.sum(ww[ok] * v[ok]) / np.sum(ww[ok] ** 2)) if ok.any() else float("nan")

    with np.errstate(all="ignore"):
        zs = 1.0 / ys
    r = float(np.nanmedian(zs.real[:k]))
    l_s = slope(zs.imag[:k])
    g1, g2 = float(np.nanmedian(y1.real[:k])), float(np.nanmedian(y2.real[:k]))
    c1, c2 = slope(y1.imag[:k]), slope(y2.imag[:k])
    with np.errstate(all="ignore"):
        ys_m = 1.0 / (r + 1j * w * l_s)
        y1m, y2m = g1 + 1j * w * c1, g2 + 1j * w * c2
    ym = np.empty_like(s)
    ym[:, 0, 0], ym[:, 0, 1], ym[:, 1, 0], ym[:, 1, 1] = y1m + ys_m, -ys_m, -ys_m, y2m + ys_m
    eye = np.broadcast_to(np.eye(2), s.shape)
    sm = _solve(eye + z0 * ym, eye - z0 * ym)
    return {"r": r, "l": l_s, "g1": g1, "c1": c1, "g2": g2, "c2": c2, "s_model": sm, "k": k}


# -- mixed-mode -------------------------------------------------------------------
def mixed_mode(s4: np.ndarray, pairing: str = "1-2 / 3-4") -> dict[str, np.ndarray]:
    """Single-ended 4-port -> mixed-mode 2x2 blocks (Sdd, Sdc, Scd, Scc)."""
    order = PAIRINGS[pairing]
    sp = s4[:, order][:, :, order]
    r = 1.0 / math.sqrt(2.0)
    m = r * np.array([[1, -1, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    smm = m @ sp @ m.T
    return {"sdd": smm[:, :2, :2], "sdc": smm[:, :2, 2:], "scd": smm[:, 2:, :2], "scc": smm[:, 2:, 2:]}


# -- noise parameters -------------------------------------------------------------
def parse_touchstone_noise(path: str | Path) -> dict[str, Any] | None:
    """Read the optional noise block (freq, NFmin dB, |Gopt|, angle, Rn/R0) of a 2-port Touchstone file."""
    units = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
    unit, ref = 1e9, 50.0
    rows: list[list[float]] = []
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            line = raw.split("!")[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                tok = line[1:].upper().split()
                for i, t in enumerate(tok):
                    if t in units:
                        unit = units[t]
                    elif t == "R" and i + 1 < len(tok):
                        try:
                            ref = float(tok[i + 1])
                        except ValueError:
                            pass
                continue
            try:
                vals = [float(t) for t in line.split()]
            except ValueError:
                continue
            if len(vals) == 5:
                rows.append(vals)
    if not rows:
        return None
    a = np.array(sorted(rows))
    return {
        "freq_hz": a[:, 0] * unit,
        "nfmin_db": a[:, 1],
        "gopt": a[:, 2] * np.exp(1j * np.deg2rad(a[:, 3])),
        "rn": a[:, 4] * ref,
        "ref": ref,
    }


def noise_figure_db(noise: dict[str, Any], gs: complex) -> np.ndarray:
    fmin = 10.0 ** (noise["nfmin_db"] / 10.0)
    gopt = noise["gopt"]
    with np.errstate(all="ignore"):
        f = fmin + 4.0 * (noise["rn"] / noise["ref"]) * np.abs(gs - gopt) ** 2 / (
            (1.0 - abs(gs) ** 2) * np.abs(1.0 + gopt) ** 2)
    return 10.0 * np.log10(f)


# -- RF metrics -------------------------------------------------------------------
def rf_metrics_lines(f: np.ndarray, s: np.ndarray) -> list[str]:
    s11, s12, s21, s22 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
    g = f / 1e9
    il = -_db(s21)
    lines = [
        f"Insertion loss: min {il.min():.2f} dB @ {g[il.argmin()]:.4g} GHz, max {il.max():.2f} dB @ {g[il.argmax()]:.4g} GHz",
    ]
    idx = np.flatnonzero(il >= il[0] + 3.0)
    if len(idx):
        i = int(idx[0])
        f3 = f[0] if i == 0 else float(np.interp(il[0] + 3.0, il[i - 1:i + 1], f[i - 1:i + 1]))
        lines.append(f"-3 dB bandwidth (relative to IL at {g[0]:.3g} GHz): {f3/1e9:.4g} GHz")
    else:
        lines.append("-3 dB bandwidth: not reached inside the measured band")
    for name, sp in (("S11", s11), ("S22", s22)):
        mag = np.minimum(np.abs(sp), 0.9999)
        rl = -_db(sp)
        lines.append(
            f"{name}: worst RL {rl.min():.2f} dB @ {g[rl.argmin()]:.4g} GHz, mean RL {rl.mean():.2f} dB, "
            f"max VSWR {np.max((1 + mag) / (1 - mag)):.2f}, max mismatch loss {abs(np.max(-10*np.log10(1 - mag**2))):.3f} dB")
    lines.append(f"Passivity: max singular value of S = {passivity(s).max():.4f} (passive if <= 1)")
    lines.append(f"Reciprocity: max |S21 - S12| = {_db(s21 - s12).max():.1f} dB")
    return lines


# ----------------------------------------------------------------------------
# Advanced tabs (mixin for the main window)
# ----------------------------------------------------------------------------
PARAM_CHOICES = ["S", "Z", "Y", "ABCD", "T", "H", "G"]
FORMAT_CHOICES = ["Magnitude (dB)", "Magnitude", "Phase (°)", "Real", "Imag"]


def _format_values(v: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "Magnitude (dB)":
        return _db(v)
    if fmt == "Magnitude":
        return np.abs(v)
    if fmt == "Phase (°)":
        return np.degrees(np.unwrap(np.angle(np.where(np.isfinite(v), v, 0.0))))
    if fmt == "Real":
        return np.real(v)
    return np.imag(v)


class AdvancedTabs:
    """Adds an 'Advanced' tab: TDR/Z-profile, group delay, gains, stability, parameter conversions,
    RF metrics, mixed-mode, equivalent circuit, noise and TRL de-embedding."""

    # -- helpers ---------------------------------------------------------------
    def _fv(self, var: tk.StringVar, default: float) -> float:
        try:
            return float(var.get())
        except (ValueError, tk.TclError):
            return default

    def _z0s(self) -> tuple[float, float]:
        z = self.channel.z0
        return float(np.median(np.real(z[:, 0]))), float(np.median(np.real(z[:, 1])))

    def _terminations(self) -> tuple[complex, complex]:
        z1, z2 = self._z0s()
        zs = complex(self._fv(self.rs_var, z1), self._fv(self.xs_var, 0.0))
        zl = complex(self._fv(self.rl_var, z2), self._fv(self.xl_var, 0.0))
        return (zs - z1) / (zs + z1), (zl - z2) / (zl + z2)

    @staticmethod
    def _adv_field(parent: tk.Widget, label: str, factory: Callable[[tk.Widget], tk.Widget]) -> tk.Widget:
        box = ttk.Frame(parent, style="Card.TFrame")
        box.pack(side="left", padx=(0, 14))
        ttk.Label(box, text=label, style="CardMuted.TLabel").pack(anchor="w")
        widget = factory(box)
        widget.pack(anchor="w", pady=(2, 0))
        return widget

    @staticmethod
    def _adv_button(parent: tk.Widget, text: str, command: Callable[[], None], accent: bool = True) -> ttk.Button:
        box = ttk.Frame(parent, style="Card.TFrame")
        box.pack(side="left", padx=(0, 10))
        ttk.Label(box, text=" ", style="CardMuted.TLabel").pack(anchor="w")
        btn = ttk.Button(box, text=text, style="Accent.TButton" if accent else "TButton", command=command)
        btn.pack(anchor="w", pady=(2, 0))
        return btn

    def _placeholder(self, name: str, message: str) -> None:
        fig = getattr(self, f"fig_{name}")
        fig.clf()
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, fontsize=12, transform=ax.transAxes)
        getattr(self, f"canvas_{name}").draw_idle()

    def _finish(self, name: str) -> None:
        getattr(self, f"canvas_{name}").draw_idle()

    @staticmethod
    def _fit_y(ax: Any, t: np.ndarray, curves: list[np.ndarray], span: float) -> None:
        m = t <= span
        vals = np.concatenate([c[m] for c in curves])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = float(vals.min()), float(vals.max())
            pad = 0.08 * ((hi - lo) or abs(hi) or 1.0)
            ax.set_ylim(lo - pad, hi + pad)

    def _set_section(self, key: str, title: str, lines: list[str]) -> None:
        self.adv_summary[key] = f"## {title}\n" + "\n".join(lines)

    def _advanced_summary(self) -> str:
        data = getattr(self, "adv_summary", None) or {}
        return "\n\n".join(data[k] for k in ("rf", "gd", "stab", "tdr", "eq", "noise", "eqmask", "lim", "qual", "xt", "sm") if k in data)

    def _redraw(self, name: str) -> None:
        getattr(self, f"_draw_{name}")()
        self._update_summary()

    # -- construction ----------------------------------------------------------
    def _adv_tab(self, title: str, name: str, controls: Callable[[tk.Widget], None] | None = None, nb: Any = None) -> None:
        parent_nb = nb if nb is not None else self.adv_nb
        tab = ttk.Frame(parent_nb)
        parent_nb.add(tab, text=title)
        if controls is not None:
            bar = ttk.Frame(tab, style="Card.TFrame", padding=(12, 8))
            bar.pack(fill="x", pady=(0, 6))
            controls(bar)
        fig = Figure(figsize=(9, 6), dpi=100, constrained_layout=True)
        setattr(self, f"fig_{name}", fig)
        setattr(self, f"canvas_{name}", self._make_plot(tab, fig))

    def _build_advanced_tabs(self) -> None:
        self.adv_summary: dict[str, str] = {}
        self.tr_var = tk.StringVar(value="35")
        self.er_var = tk.StringVar(value="4.0")
        self.span_var = tk.StringVar(value="0")
        self.win_var = tk.StringVar(value=WINDOWS[0])
        self.rs_var, self.xs_var = tk.StringVar(value="50"), tk.StringVar(value="0")
        self.rl_var, self.xl_var = tk.StringVar(value="50"), tk.StringVar(value="0")
        self.par_var = tk.StringVar(value="S")
        self.fmt_var = tk.StringVar(value=FORMAT_CHOICES[0])
        self.pair_var = tk.StringVar(value="1-2 / 3-4")
        self.len_var = tk.StringVar(value="50")
        self.fitpct_var = tk.StringVar(value="5")
        self.reflect_var = tk.StringVar(value="Short (−1)")
        self.trl_status = tk.StringVar(value="Load Thru, Reflect, Line and the raw DUT, then press Run TRL.")
        self.trl_nets: dict[str, Any] = {}
        self.trl_labels: dict[str, ttk.Label] = {}
        self.trl_result: Any = None

        style = ttk.Style(self)
        style.configure("Inner.TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure("Inner.TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(11, 6), borderwidth=0)
        style.map("Inner.TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", ACCENT2)])

        self.adv_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.adv_tab, text="Advanced")
        self.adv_nb = ttk.Notebook(self.adv_tab, style="Inner.TNotebook")
        self.adv_nb.pack(fill="both", expand=True, pady=(6, 0))

        def tdr_controls(bar: tk.Widget) -> None:
            self._adv_field(bar, "Rise time 10–90 % (ps)", lambda p: ttk.Entry(p, textvariable=self.tr_var, width=8))
            self._adv_field(bar, "εr,eff (distance axis)", lambda p: ttk.Entry(p, textvariable=self.er_var, width=7))
            self._adv_field(bar, "Window", lambda p: ttk.Combobox(p, textvariable=self.win_var, values=WINDOWS, width=13, state="readonly"))
            self._adv_field(bar, "View span (ns, 0 = auto)", lambda p: ttk.Entry(p, textvariable=self.span_var, width=8))
            self._adv_button(bar, "Update", lambda: self._redraw("tdr"))

        def gain_controls(bar: tk.Widget) -> None:
            self._adv_field(bar, "Source R (Ω)", lambda p: ttk.Entry(p, textvariable=self.rs_var, width=8))
            self._adv_field(bar, "Source X (Ω)", lambda p: ttk.Entry(p, textvariable=self.xs_var, width=8))
            self._adv_field(bar, "Load R (Ω)", lambda p: ttk.Entry(p, textvariable=self.rl_var, width=8))
            self._adv_field(bar, "Load X (Ω)", lambda p: ttk.Entry(p, textvariable=self.xl_var, width=8))
            self._adv_button(bar, "Update", self._update_terminations)

        def param_controls(bar: tk.Widget) -> None:
            for var, label, values, width in ((self.par_var, "Parameter set", PARAM_CHOICES, 8),
                                              (self.fmt_var, "Display", FORMAT_CHOICES, 15)):
                box = self._adv_field(bar, label, lambda p, v=var, vals=values, w=width: ttk.Combobox(
                    p, textvariable=v, values=vals, width=w, state="readonly"))
                box.bind("<<ComboboxSelected>>", lambda _e: self._redraw("params"))

        def mm_controls(bar: tk.Widget) -> None:
            self._adv_field(bar, "Differential pairs (4-port files)", lambda p: ttk.Combobox(
                p, textvariable=self.pair_var, values=list(PAIRINGS), width=12, state="readonly"))
            self._adv_button(bar, "Apply pairing", self._apply_pairing)

        def eq_controls(bar: tk.Widget) -> None:
            self._adv_field(bar, "Line length (mm, for RLGC)", lambda p: ttk.Entry(p, textvariable=self.len_var, width=8))
            self._adv_field(bar, "π-model fit: lowest % of band", lambda p: ttk.Entry(p, textvariable=self.fitpct_var, width=8))
            self._adv_button(bar, "Extract", lambda: self._redraw("eq"))

        def trl_controls(bar: tk.Widget) -> None:
            for key, title in (("thru", "Thru"), ("reflect", "Reflect"), ("line", "Line"), ("dut", "DUT (raw)")):
                box = ttk.Frame(bar, style="Card.TFrame")
                box.pack(side="left", padx=(0, 12))
                ttk.Button(box, text=f"Load {title}", command=lambda k=key: self._trl_load(k)).pack(anchor="w")
                lbl = ttk.Label(box, text="—", style="CardMuted.TLabel")
                lbl.pack(anchor="w", pady=(2, 0))
                self.trl_labels[key] = lbl
            self._adv_field(bar, "Reflect standard", lambda p: ttk.Combobox(
                p, textvariable=self.reflect_var, values=["Short (−1)", "Open (+1)"], width=10, state="readonly"))
            self._adv_button(bar, "Run TRL", self._trl_run)
            self._adv_button(bar, "Use corrected as channel", self._trl_apply, accent=False)
            ttk.Label(bar, textvariable=self.trl_status, style="CardMuted.TLabel", wraplength=260).pack(side="left", padx=(6, 0))

        self._adv_tab("TDR / Z-profile", "tdr", tdr_controls)
        self._adv_tab("Group delay", "gd")
        self._adv_tab("Gains & stability", "gains", gain_controls)
        self._adv_tab("Parameters", "params", param_controls)
        self._adv_tab("RF metrics", "rf")
        self._adv_tab("Mixed-mode", "mm", mm_controls)
        self._adv_tab("Equivalent circuit", "eq", eq_controls)
        self._adv_tab("Noise", "noise")
        self._adv_tab("TRL de-embed", "trl", trl_controls)
        self._placeholder("trl", "TRL calibration: load Thru, Reflect, Line and the raw DUT measurements.")
        for name in ("tdr", "gd", "gains", "params", "rf", "mm", "eq", "noise"):
            self._placeholder(name, "Load a Touchstone file first.")

    def _refresh_advanced(self) -> None:
        self.adv_summary = {}
        for name in ("tdr", "gd", "gains", "params", "rf", "mm", "eq", "noise"):
            try:
                getattr(self, f"_draw_{name}")()
            except Exception as exc:  # keep the other tabs alive if one analysis fails
                self._placeholder(name, f"{name} analysis failed:\n{exc}")

    def _update_terminations(self) -> None:
        self._draw_gains()
        self._draw_noise()
        self._update_summary()

    def _apply_pairing(self) -> None:
        d = self.channel
        if d is None or d.s_full is None or d.s_full.shape[1] != 4:
            messagebox.showinfo("4-port file needed", "Mixed-mode pairing applies to 4-port (.s4p) files only.")
            return
        d.pairing = self.pair_var.get()
        d.s = mixed_mode(d.s_full, d.pairing)["sdd"]
        self._refresh_analysis()

    # -- TDR / impedance profile -------------------------------------------------
    def _draw_tdr(self) -> None:
        if self.channel is None:
            return self._placeholder("tdr", "Load a Touchstone file first.")
        d, f = self.channel, self.channel.frequency_hz
        tr = max(self._fv(self.tr_var, 35.0), 0.0) * 1e-12
        er = max(self._fv(self.er_var, 4.0), 1.0)
        v = SPEED_OF_LIGHT / math.sqrt(er)
        z1, z2 = self._z0s()
        win = self.win_var.get()
        resp = {name: time_response(f, d.s[:, i, j], tr, win)
                for name, (i, j) in {"S11": (0, 0), "S22": (1, 1), "S21": (1, 0), "S12": (0, 1)}.items()}
        t = resp["S11"][0]
        tau = float(np.nanmedian(np.abs(group_delay(f, d.s[:, 1, 0]))))
        span = self._fv(self.span_var, 0.0) * 1e-9
        if span <= 0:
            span = max(6.0 * (tau if np.isfinite(tau) else 0.0) + 8.0 * tr, 20.0 / f[-1])
        tn = t * 1e9

        fig = self.fig_tdr
        fig.clf()
        ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()
        ax1.plot(tn, resp["S11"][1], color=ACCENT, label="Port 1 (S11)")
        ax1.plot(tn, resp["S22"][1], color=PURPLE, linestyle="--", label="Port 2 (S22)")
        ax1.set_title("TDR — reflection step response")
        ax1.set_ylabel("Reflection ρ")
        z_a, z_b = impedance_from_step(resp["S11"][1], z1), impedance_from_step(resp["S22"][1], z2)
        ax2.plot(tn, z_a, color=ACCENT, label="Port 1 profile")
        ax2.plot(tn, z_b, color=PURPLE, linestyle="--", label="Port 2 profile")
        ax2.axhline(z1, color=MUTED, linewidth=0.8, linestyle=":")
        ax2.set_title("Impedance profile Z(t)")
        ax2.set_ylabel("Impedance (Ω)")
        to_mm = lambda x: x * 1e-9 * v / 2.0 * 1e3  # noqa: E731  (round-trip distance)
        top = ax2.secondary_xaxis("top", functions=(to_mm, lambda x: x / (1e-9 * v / 2.0 * 1e3)))
        top.set_xlabel(f"Distance (mm), εr,eff = {er:g}", color=MUTED)
        top.tick_params(colors=MUTED)
        ax3.plot(tn, resp["S21"][1], color=ACCENT2, label="S21 step")
        ax3.plot(tn, resp["S12"][1], color=GREEN, linestyle="--", label="S12 step")
        ax3.set_title("TDT — transmission step response")
        ax3.set_ylabel("Amplitude")
        ax4.plot(tn, resp["S21"][2] * 1e-9, color=ACCENT2, label="S21 impulse")
        ax4.set_title("Impulse response of S21")
        ax4.set_ylabel("h(t) (1/ns)")
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xlim(0, span * 1e9)
            ax.set_xlabel("Time (ns)")
            ax.grid(True)
            ax.legend(loc="best")
        self._fit_y(ax1, t, [resp["S11"][1], resp["S22"][1]], span)
        self._fit_y(ax2, t, [z_a, z_b], span)
        self._fit_y(ax3, t, [resp["S21"][1], resp["S12"][1]], span)
        self._fit_y(ax4, t, [resp["S21"][2] * 1e-9], span)
        self._finish("tdr")

        m = t <= span
        self._set_section("tdr", "Impedance profile (TDR)", [
            f"Rise time {tr*1e12:g} ps, window {win}, view {span*1e9:.3g} ns",
            f"Port 1: Z min {np.nanmin(z_a[m]):.1f} Ω, max {np.nanmax(z_a[m]):.1f} Ω (reference {z1:g} Ω)",
            f"Port 2: Z min {np.nanmin(z_b[m]):.1f} Ω, max {np.nanmax(z_b[m]):.1f} Ω (reference {z2:g} Ω)",
            self._disc_line("Port 1 discontinuities", t, z_a, z1, span, v),
            self._disc_line("Port 2 discontinuities", t, z_b, z2, span, v),
        ])

    # -- group delay ---------------------------------------------------------------
    def _draw_gd(self) -> None:
        if self.channel is None:
            return self._placeholder("gd", "Load a Touchstone file first.")
        d, f = self.channel, self.channel.frequency_hz
        g = f / 1e9
        gd21, gd12 = group_delay(f, d.s[:, 1, 0]) * 1e12, group_delay(f, d.s[:, 0, 1]) * 1e12
        gd11 = group_delay(f, d.s[:, 0, 0]) * 1e12
        phase = np.degrees(np.unwrap(np.angle(d.s[:, 1, 0])))
        dev = phase - np.polyval(np.polyfit(f, phase, 1), f)
        fig = self.fig_gd
        fig.clf()
        ax1, ax2, ax3 = fig.subplots(1, 3)
        ax1.plot(g, gd21, color=ACCENT2, label="S21")
        ax1.plot(g, gd12, color=GREEN, linestyle="--", label="S12")
        ax1.set_title("Transmission group delay")
        ax1.set_ylabel("τg (ps)")
        ax2.plot(g, gd11, color=ACCENT, label="S11")
        ax2.set_title("Reflection group delay")
        ax2.set_ylabel("τg (ps)")
        ax3.plot(g, dev, color=PURPLE, label="S21 phase − linear fit")
        ax3.set_title("Phase linearity")
        ax3.set_ylabel("Deviation (°)")
        for ax in (ax1, ax2, ax3):
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True)
            ax.legend(loc="best")
        self._finish("gd")
        self._set_section("gd", "Group delay (S21)", [
            f"Mean {np.nanmean(gd21):.2f} ps, min {np.nanmin(gd21):.2f} ps, max {np.nanmax(gd21):.2f} ps, "
            f"variation (p-p) {np.nanmax(gd21) - np.nanmin(gd21):.2f} ps",
            f"Max deviation from linear phase: {np.max(np.abs(dev)):.2f}°",
        ])

    # -- power gains and stability ---------------------------------------------------
    def _draw_gains(self) -> None:
        if self.channel is None:
            return self._placeholder("gains", "Load a Touchstone file first.")
        d, f = self.channel, self.channel.frequency_hz
        g = f / 1e9
        gs, gl = self._terminations()
        gains = power_gains(d.s, gs, gl)
        st = stability_metrics(d.s)
        fig = self.fig_gains
        fig.clf()
        ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()
        for key, col, ls in (("GT", ACCENT, "-"), ("GA", GREEN, "-"), ("GP", PURPLE, "-")):
            ax1.plot(g, _pdb(gains[key]), ls, color=col, label=f"{key} (Γs={abs(gs):.2f}, ΓL={abs(gl):.2f})" if key == "GT" else key)
        ax1.plot(g, _pdb(st["MSG"]), "--", color=MUTED, label="MSG")
        ax1.plot(g, _pdb(st["MAG"]), "--", color=ACCENT2, label="MAG")
        ax1.plot(g, _pdb(st["U"]), ":", color=RED, label="Mason U")
        ax1.set_title("Power gains")
        ax1.set_ylabel("Gain (dB)")
        ax2.semilogy(g, np.where(np.isfinite(st["K"]) & (st["K"] > 0), st["K"], np.nan), color=ACCENT, label="Rollett K")
        ax2.axhline(1.0, color=RED, linestyle="--", linewidth=1, label="K = 1")
        ax2.set_title("K-factor")
        ax3.plot(g, np.abs(st["delta"]), color=ACCENT2, label="|Δ|")
        ax3.axhline(1.0, color=RED, linestyle="--", linewidth=1, label="|Δ| = 1")
        ax3.set_title("Determinant |Δ|")
        ax4.plot(g, st["mu"], color=GREEN, label="μ (load)")
        ax4.plot(g, st["mu_p"], color=PURPLE, linestyle="--", label="μ′ (source)")
        ax4.axhline(1.0, color=RED, linestyle="--", linewidth=1, label="μ = 1")
        ax4.set_title("Edwards–Sinsky μ / μ′")
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)
        self._finish("gains")

        k, delta = st["K"], st["delta"]
        ok_k = bool(np.all((k > 1) & (np.abs(delta) < 1)))
        ok_mu = bool(np.all(st["mu"] > 1))
        self._set_section("stab", "Stability & power gain", [
            f"Terminations: Γs = {abs(gs):.3f}∠{np.degrees(np.angle(gs)):.0f}°, ΓL = {abs(gl):.3f}∠{np.degrees(np.angle(gl)):.0f}°",
            f"Unconditionally stable (K > 1 and |Δ| < 1) over the band: {'yes' if ok_k else 'no'}; μ > 1 everywhere: {'yes' if ok_mu else 'no'}",
            f"K min {np.nanmin(k):.3g}, |Δ| max {np.nanmax(np.abs(delta)):.3g}, μ min {np.nanmin(st['mu']):.3g}, μ′ min {np.nanmin(st['mu_p']):.3g}",
            f"At {g[-1]:.4g} GHz: GT {_pdb(gains['GT'])[-1]:.2f} dB, GA {_pdb(gains['GA'])[-1]:.2f} dB, GP {_pdb(gains['GP'])[-1]:.2f} dB",
        ])

    # -- parameter conversions ---------------------------------------------------------
    def _draw_params(self) -> None:
        if self.channel is None:
            return self._placeholder("params", "Load a Touchstone file first.")
        d, f = self.channel, self.channel.frequency_hz
        z1, z2 = self._z0s()
        name, fmt = self.par_var.get(), self.fmt_var.get()
        mats = convert_parameters(d.s, 0.5 * (z1 + z2))
        m = mats[name]
        fig = self.fig_params
        fig.clf()
        axes = fig.subplots(2, 2)
        for i in range(2):
            for j in range(2):
                ax = axes[i, j]
                ax.plot(f / 1e9, _format_values(m[:, i, j], fmt), color=(ACCENT, ACCENT2, GREEN, PURPLE)[2 * i + j])
                ax.set_title(f"{name}{i + 1}{j + 1}")
                ax.set_xlabel("Frequency (GHz)")
                ax.set_ylabel(fmt)
                ax.grid(True)
        fig.suptitle(f"{name}-parameters (equal reference Z0 = {0.5 * (z1 + z2):g} Ω; T: [b1,a1] = T [a2,b2])",
                     color=MUTED, fontsize=9)
        self._finish("params")

    # -- RF metrics ------------------------------------------------------------------------
    def _draw_rf(self) -> None:
        if self.channel is None:
            return self._placeholder("rf", "Load a Touchstone file first.")
        d, f = self.channel, self.channel.frequency_hz
        g = f / 1e9
        s11, s12, s21, s22 = d.s[:, 0, 0], d.s[:, 0, 1], d.s[:, 1, 0], d.s[:, 1, 1]
        vswr = lambda x: (1 + np.minimum(np.abs(x), 0.9999)) / (1 - np.minimum(np.abs(x), 0.9999))  # noqa: E731
        fig = self.fig_rf
        fig.clf()
        ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()
        ax1.plot(g, vswr(s11), color=ACCENT, label="VSWR in")
        ax1.plot(g, vswr(s22), color=PURPLE, linestyle="--", label="VSWR out")
        ax1.set_ylim(1.0, min(max(float(np.max(vswr(s11))), float(np.max(vswr(s22)))) * 1.05, 50.0))
        ax1.set_title("VSWR")
        ax2.plot(g, -_db(s11), color=ACCENT, label="RL S11")
        ax2.plot(g, -_db(s22), color=PURPLE, linestyle="--", label="RL S22")
        ax2.set_title("Return loss")
        ax2.set_ylabel("dB")
        ax3.plot(g, passivity(d.s), color=GREEN, label="σmax(S)")
        ax3.axhline(1.0, color=RED, linestyle="--", linewidth=1, label="passive limit")
        ax3.set_title("Passivity")
        ax4.plot(g, _db(s21 - s12), color=ACCENT2, label="|S21 − S12|")
        ax4.set_title("Reciprocity error")
        ax4.set_ylabel("dB")
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True)
            ax.legend(loc="best")
        self._finish("rf")
        self._set_section("rf", "RF metrics", rf_metrics_lines(f, d.s))

    # -- mixed-mode --------------------------------------------------------------------------
    def _draw_mm(self) -> None:
        d = self.channel
        if d is None:
            return self._placeholder("mm", "Load a Touchstone file first.")
        if d.s_full is None or d.s_full.shape[1] != 4:
            return self._placeholder("mm", "Mixed-mode S-parameters need a 4-port (.s4p) file.\n"
                                           "The current file is a 2-port, so no differential/common-mode split exists.")
        g = d.frequency_hz / 1e9
        mm = mixed_mode(d.s_full, d.pairing)
        fig = self.fig_mm
        fig.clf()
        ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()
        plots = (
            (ax1, "Return loss (diff / common)", (("Sdd11", mm["sdd"][:, 0, 0], ACCENT, "-"), ("Sdd22", mm["sdd"][:, 1, 1], PURPLE, "--"),
                                                   ("Scc11", mm["scc"][:, 0, 0], ACCENT2, "-"), ("Scc22", mm["scc"][:, 1, 1], GREEN, "--"))),
            (ax2, "Transmission (diff / common)", (("Sdd21", mm["sdd"][:, 1, 0], ACCENT, "-"), ("Sdd12", mm["sdd"][:, 0, 1], PURPLE, "--"),
                                                    ("Scc21", mm["scc"][:, 1, 0], ACCENT2, "-"), ("Scc12", mm["scc"][:, 0, 1], GREEN, "--"))),
            (ax3, "Mode conversion  diff → common (Scd)", (("Scd11", mm["scd"][:, 0, 0], ACCENT, "-"), ("Scd21", mm["scd"][:, 1, 0], ACCENT2, "-"),
                                                            ("Scd12", mm["scd"][:, 0, 1], GREEN, "--"), ("Scd22", mm["scd"][:, 1, 1], PURPLE, "--"))),
            (ax4, "Mode conversion  common → diff (Sdc)", (("Sdc11", mm["sdc"][:, 0, 0], ACCENT, "-"), ("Sdc21", mm["sdc"][:, 1, 0], ACCENT2, "-"),
                                                            ("Sdc12", mm["sdc"][:, 0, 1], GREEN, "--"), ("Sdc22", mm["sdc"][:, 1, 1], PURPLE, "--"))),
        )
        for ax, title, curves in plots:
            for label, y, col, ls in curves:
                ax.plot(g, _db(y), ls, color=col, label=label)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True)
            ax.legend(loc="best", fontsize=8, ncol=2)
        fig.suptitle(f"Mixed-mode S-parameters — pairs {d.pairing} (the Sdd block drives the other tabs)", color=MUTED, fontsize=9)
        self._finish("mm")

    # -- equivalent circuit ------------------------------------------------------------------
    def _draw_eq(self) -> None:
        if self.channel is None:
            return self._placeholder("eq", "Load a Touchstone file first.")
        d, f = self.channel, self.channel.frequency_hz
        g = f / 1e9
        z1, z2 = self._z0s()
        z0 = 0.5 * (z1 + z2)
        frac = min(max(self._fv(self.fitpct_var, 25.0), 1.0), 100.0) / 100.0
        length = self._fv(self.len_var, 0.0) * 1e-3
        pi = fit_pi_model(f, d.s, z0, frac)
        rlgc = extract_rlgc(f, d.s, z0, length) if length > 0 else None

        fig = self.fig_eq
        fig.clf()
        axes = fig.subplots(2, 4).ravel()
        rlgc_plots = (("R′ (Ω/m)", "r", 1.0), ("L′ (nH/m)", "l", 1e9), ("G′ (S/m)", "g", 1.0), ("C′ (pF/m)", "c", 1e12))
        for ax, (title, key, scale) in zip(axes[:4], rlgc_plots):
            if rlgc is None:
                ax.text(0.5, 0.5, "Enter a line length\n(mm) > 0", ha="center", va="center", color=MUTED, transform=ax.transAxes)
                ax.axis("off")
                continue
            ax.plot(g, rlgc[key] * scale, color=ACCENT)
            ax.set_title(title)
        if rlgc is not None:
            axes[4].plot(g, np.real(rlgc["zc"]), color=ACCENT2, label="Re Zc")
            axes[4].plot(g, np.imag(rlgc["zc"]), color=PURPLE, linestyle="--", label="Im Zc")
            axes[4].set_title("Characteristic impedance (Ω)")
            axes[4].legend(loc="best")
            axes[5].plot(g, rlgc["eps_eff"], color=GREEN)
            axes[5].set_title("εr,eff from β")
        else:
            axes[4].axis("off")
            axes[5].axis("off")
        axes[6].plot(g, _db(d.s[:, 1, 0]), color=ACCENT2, label="Measured")
        axes[6].plot(g, _db(pi["s_model"][:, 1, 0]), "--", color=GREEN, label="π model")
        axes[6].set_title("|S21| (dB)")
        axes[7].plot(g, _db(d.s[:, 0, 0]), color=ACCENT, label="Measured")
        axes[7].plot(g, _db(pi["s_model"][:, 0, 0]), "--", color=GREEN, label="π model")
        axes[7].set_title("|S11| (dB)")
        for ax in axes:
            if ax.axison:
                ax.set_xlabel("Frequency (GHz)")
                ax.grid(True)
        for ax in axes[6:]:
            ax.legend(loc="best", fontsize=8)
        self._finish("eq")

        lines = [
            f"π model fitted on the lowest {pi['k']} points ({g[pi['k'] - 1]:.3g} GHz): "
            f"R = {pi['r']*1e3:.2f} mΩ, L = {pi['l']*1e9:.3f} nH",
            f"Shunt ends: C1 = {pi['c1']*1e12:.3f} pF, G1 = {pi['g1']*1e3:.4g} mS; C2 = {pi['c2']*1e12:.3f} pF, G2 = {pi['g2']*1e3:.4g} mS",
        ]
        if rlgc is not None:
            k = pi["k"]
            lines.append(
                f"Per length (median, same range): R′ {np.nanmedian(rlgc['r'][:k]):.4g} Ω/m, L′ {np.nanmedian(rlgc['l'][:k])*1e9:.4g} nH/m, "
                f"G′ {np.nanmedian(rlgc['g'][:k]):.4g} S/m, C′ {np.nanmedian(rlgc['c'][:k])*1e12:.4g} pF/m, "
                f"|Zc| {np.nanmedian(np.abs(rlgc['zc'][:k])):.2f} Ω")
        self._set_section("eq", "Equivalent circuit", lines)

    # -- noise -------------------------------------------------------------------------------
    def _draw_noise(self) -> None:
        d = self.channel
        if d is None:
            return self._placeholder("noise", "Load a Touchstone file first.")
        gs, _ = self._terminations()
        fig = self.fig_noise
        fig.clf()
        nz = d.noise
        if nz is None:
            ax = fig.add_subplot(111)
            nf = -_pdb(power_gains(d.s, gs, 0j)["GA"])
            g = d.frequency_hz / 1e9
            ax.plot(g, nf, color=ACCENT2, label="−10 log10(GA)  (passive, 290 K)")
            ax.set_title("No noise block in this file — passive-channel noise-figure estimate")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Noise figure (dB)")
            ax.grid(True)
            ax.legend(loc="best")
            self._finish("noise")
            sig = float(passivity(d.s).max())
            self._set_section("noise", "Noise", [
                "No noise-parameter block found in the Touchstone file.",
                f"Passive estimate NF = −10 log10(GA) at 290 K: {np.nanmin(nf):.2f} to {np.nanmax(nf):.2f} dB "
                f"(valid only for a passive network; max σ(S) = {sig:.3f}).",
            ])
            return
        gnf = nz["freq_hz"] / 1e9
        nf_gs = noise_figure_db(nz, gs)
        ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()
        ax1.plot(gnf, nz["nfmin_db"], "o-", color=ACCENT, label="NFmin")
        ax1.plot(gnf, nf_gs, "s--", color=ACCENT2, label=f"NF @ source Γs = {abs(gs):.2f}")
        ax1.set_title("Noise figure")
        ax1.set_ylabel("dB")
        ax2.plot(gnf, nz["rn"], "o-", color=GREEN, label="Rn")
        ax2.set_title("Equivalent noise resistance")
        ax2.set_ylabel("Ω")
        ax3.plot(gnf, np.abs(nz["gopt"]), "o-", color=PURPLE, label="|Γopt|")
        ax3.set_title("Optimum source reflection magnitude")
        ax4.plot(gnf, np.degrees(np.angle(nz["gopt"])), "o-", color=RED, label="∠Γopt")
        ax4.set_title("Optimum source reflection angle")
        ax4.set_ylabel("deg")
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True)
            ax.legend(loc="best")
        self._finish("noise")
        self._set_section("noise", "Noise parameters", [
            f"{len(gnf)} noise points, {gnf[0]:.4g}–{gnf[-1]:.4g} GHz",
            f"NFmin {nz['nfmin_db'].min():.2f}–{nz['nfmin_db'].max():.2f} dB; NF at the chosen source {np.nanmin(nf_gs):.2f}–{np.nanmax(nf_gs):.2f} dB",
            f"Rn {nz['rn'].min():.2f}–{nz['rn'].max():.2f} Ω",
        ])

    # -- TRL calibration --------------------------------------------------------------------------
    def _trl_load(self, key: str) -> None:
        path = filedialog.askopenfilename(
            title=f"Select the {key} measurement",
            filetypes=[("Touchstone", "*.s1p *.s2p *.S1P *.S2P"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.trl_nets[key] = rf.Network(path)
        except Exception as exc:
            messagebox.showerror("Could not load file", str(exc))
            return
        self.trl_labels[key].configure(text=Path(path).name[:22])

    def _trl_run(self) -> None:
        missing = [k for k in ("thru", "reflect", "line", "dut") if k not in self.trl_nets]
        if missing:
            messagebox.showinfo("Missing standards", "Still to load: " + ", ".join(missing))
            return
        try:
            dut = self.trl_nets["dut"]
            if dut.nports != 2:
                raise ValueError("The raw DUT must be a 2-port file.")

            def prep(net: Any) -> Any:
                if net.nports == 1:
                    from skrf.network import two_port_reflect  # location differs between scikit-rf versions
                    net = two_port_reflect(net, net)
                if net.nports != 2:
                    raise ValueError("Standards must be 1-port (reflect) or 2-port files.")
                if len(net.f) != len(dut.f) or not np.allclose(net.f, dut.f):
                    net = net.interpolate(dut.frequency)
                return net

            thru, refl, line = prep(self.trl_nets["thru"]), prep(self.trl_nets["reflect"]), prep(self.trl_nets["line"])
            est = -1 if self.reflect_var.get().startswith("Short") else 1
            cal, errors = None, []
            for kwargs in ({"n_reflects": 1, "reflect_est": est}, {"ideals": None, "n_reflects": 1, "reflect_est": est}):
                try:
                    from skrf.calibration import TRL  # location differs between scikit-rf versions
                    cal = TRL(measured=[thru, refl, line], **kwargs)
                    break
                except Exception as exc:
                    errors.append(str(exc))
            if cal is None:
                raise RuntimeError("; ".join(errors))
            cal.run()
            corrected = cal.apply_cal(dut)
        except Exception as exc:
            self.trl_status.set(f"TRL failed: {exc}")
            messagebox.showerror("TRL calibration failed", str(exc))
            return
        self.trl_result = corrected
        self.trl_status.set(f"TRL done: {len(corrected.f)} points. Plots show raw vs corrected.")
        fig = self.fig_trl
        fig.clf()
        ax1, ax2 = fig.subplots(1, 2)
        g = dut.f / 1e9
        for ax, (i, j), title in ((ax1, (1, 0), "|S21|"), (ax2, (0, 0), "|S11|")):
            ax.plot(g, _db(dut.s[:, i, j]), color=MUTED, label="Raw (with fixtures)")
            ax.plot(g, _db(corrected.s[:, i, j]), color=ACCENT2, label="TRL corrected")
            ax.set_title(title + " (dB)")
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True)
            ax.legend(loc="best")
        self._finish("trl")

    def _trl_apply(self) -> None:
        net = self.trl_result
        if net is None:
            messagebox.showinfo("Nothing to apply", "Run the TRL calibration first.")
            return
        z0 = np.asarray(net.z0, dtype=complex)
        if z0.ndim == 1:
            z0 = np.repeat(z0[:, None], 2, axis=1)
        self.channel = ChannelData(Path("TRL-corrected.s2p"), np.asarray(net.f, dtype=float),
                                   np.asarray(net.s, dtype=complex), z0)
        self.eye_result = None
        self._refresh_analysis()
        self._set_status("Using the TRL-corrected DUT as the active channel.")


# ----------------------------------------------------------------------------
# Extra helpers: causality, gating, passivity, crosstalk weighting, circles, matching
# ----------------------------------------------------------------------------
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def impulse_full(f: np.ndarray, h: np.ndarray, n_freq: int = 8192, ext: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided impulse response (t from -T/2 to +T/2) of a measured response, Hann-windowed."""
    f = np.asarray(f, dtype=float)
    h = np.asarray(h, dtype=complex)
    fmax = float(f[-1])
    fg = np.linspace(0.0, ext * fmax, n_freq)
    hg = np.interp(fg, f, h.real, left=h.real[0], right=0.0) + 1j * np.interp(fg, f, h.imag, left=0.0, right=0.0)
    k = int(np.searchsorted(fg, fmax, side="right"))
    w = np.zeros(n_freq)
    w[:k] = 0.5 * (1.0 + np.cos(np.pi * fg[:k] / fmax))
    x = np.fft.irfft(hg * w, n=2 * (n_freq - 1))
    dt = 1.0 / (2.0 * ext * fmax)
    t = (np.arange(len(x)) - len(x) // 2) * dt
    return t, np.fft.fftshift(x) / dt


def gate_response(f: np.ndarray, h: np.ndarray, t1: float, t2: float, edge: float, keep: bool = True,
                  n_freq: int = 8192, ext: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-gate a response: keep (or remove) the window [t1, t2] with raised-cosine edges.

    Returns (gated response at f, t, impulse, gate) with t sorted from negative to positive time.
    """
    f = np.asarray(f, dtype=float)
    h = np.asarray(h, dtype=complex)
    fg = np.linspace(0.0, ext * float(f[-1]), n_freq)
    hg = np.interp(fg, f, h.real, left=h.real[0], right=0.0) + 1j * np.interp(fg, f, h.imag, left=0.0, right=0.0)
    x = np.fft.irfft(hg, n=2 * (n_freq - 1))
    n = len(x)
    dt = 1.0 / (2.0 * ext * float(f[-1]))
    idx = np.arange(n)
    ts = np.where(idx <= n // 2, idx, idx - n) * dt
    e = max(edge, dt)

    def rc(u: np.ndarray) -> np.ndarray:
        return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))

    g = rc((ts - (t1 - e)) / e) * rc(((t2 + e) - ts) / e)
    xg = x * g if keep else x * (1.0 - g)
    hh = np.fft.rfft(xg)
    out = np.interp(f, fg, hh.real) + 1j * np.interp(f, fg, hh.imag)
    order = np.argsort(ts)
    return out, ts[order], (x / dt)[order], g[order]


def enforce_passivity(s: np.ndarray, limit: float = 0.999) -> np.ndarray:
    """Clip the singular values of S at every frequency to `limit` (simple passivity repair)."""
    u, sv, vh = np.linalg.svd(s)
    return (u * np.minimum(sv, limit)[:, None, :]) @ vh


def icn_rms(f: np.ndarray, xs: list[np.ndarray], fb: float, amp: float) -> float:
    """Integrated crosstalk noise (RMS volts) with an IEEE 802.3-style weighting (ft = fr = 0.75 fb)."""
    if not xs:
        return 0.0
    ft = fr = 0.75 * fb
    w = amp ** 2 / fb * np.sinc(f / fb) ** 2 / (1.0 + (f / ft) ** 4) / (1.0 + (f / fr) ** 8)
    return math.sqrt(sum(float(_TRAPZ(w * np.abs(x) ** 2, f)) for x in xs))


def parse_limit_points(text: str) -> tuple[np.ndarray, np.ndarray]:
    """'f_GHz:dB, f_GHz:dB, ...' -> sorted arrays (Hz, dB)."""
    pts = []
    for chunk in text.replace(";", ",").split(","):
        if ":" in chunk:
            a, b = chunk.split(":", 1)
            try:
                pts.append((float(a) * 1e9, float(b)))
            except ValueError:
                pass
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def _x_text(x: float, w: float) -> str:
    return f"{x / w * 1e9:.3g} nH" if x > 0 else f"{-1.0 / (w * x) * 1e12:.3g} pF"


def _b_text(b: float, w: float) -> str:
    return f"{b / w * 1e12:.3g} pF" if b > 0 else f"{-1.0 / (w * b) * 1e9:.3g} nH"


def l_match(z_load: complex, r0: float, w: float) -> list[str]:
    """Lumped L-section solutions matching z_load to r0 at angular frequency w."""
    sols: list[str] = []
    g, b = (1.0 / z_load).real, (1.0 / z_load).imag
    if 0 < g <= 1.0 / r0:
        for sgn in (1, -1):
            bt = sgn * math.sqrt(max(g / r0 - g * g, 0.0))
            sols.append(f"shunt {_b_text(bt - b, w)} across the load, then series {_x_text(bt * r0 / g, w)} toward the source")
    r, x = z_load.real, z_load.imag
    if 0 < r <= r0:
        for sgn in (1, -1):
            xt = sgn * math.sqrt(max(r * r0 - r * r, 0.0))
            sols.append(f"series {_x_text(xt - x, w)} at the load, then shunt {_b_text(xt / (r * r0), w)} toward the source")
    return sols


def stability_circles(s: np.ndarray) -> tuple[tuple[complex, float], tuple[complex, float]]:
    """Source- and load-plane stability circles (centre, radius) for one 2x2 S matrix."""
    s11, s12, s21, s22 = s[0, 0], s[0, 1], s[1, 0], s[1, 1]
    d = s11 * s22 - s12 * s21
    with np.errstate(all="ignore"):
        cs = np.conj(s11 - d * np.conj(s22)) / (abs(s11) ** 2 - abs(d) ** 2)
        rs = abs(s12 * s21) / abs(abs(s11) ** 2 - abs(d) ** 2)
        cl = np.conj(s22 - d * np.conj(s11)) / (abs(s22) ** 2 - abs(d) ** 2)
        rl = abs(s12 * s21) / abs(abs(s22) ** 2 - abs(d) ** 2)
    return (complex(cs), float(rs)), (complex(cl), float(rl))


def available_gain_circle(s: np.ndarray, ga_lin: float) -> tuple[complex, float] | None:
    s11, s12, s21, s22 = s[0, 0], s[0, 1], s[1, 0], s[1, 1]
    d = s11 * s22 - s12 * s21
    k = float(stability_metrics(s[None])["K"][0])
    g = ga_lin / abs(s21) ** 2
    den = 1.0 + g * (abs(s11) ** 2 - abs(d) ** 2)
    rad2 = 1.0 - 2.0 * k * abs(s12 * s21) * g + abs(s12 * s21) ** 2 * g * g
    if not np.isfinite(rad2) or rad2 <= 0 or abs(den) < 1e-12:
        return None
    return complex(g * np.conj(s11 - d * np.conj(s22)) / den), float(math.sqrt(rad2) / abs(den))


def noise_circle(fmin_lin: float, gopt: complex, rn_norm: float, f_lin: float) -> tuple[complex, float] | None:
    n = (f_lin - fmin_lin) / (4.0 * rn_norm) * abs(1.0 + gopt) ** 2
    if not np.isfinite(n) or n < 0:
        return None
    return complex(gopt / (1.0 + n)), float(math.sqrt(max(n * n + n * (1.0 - abs(gopt) ** 2), 0.0)) / (1.0 + n))


# ----------------------------------------------------------------------------
# Extra tabs (mixin): 'Quality & EQ' and 'Tools'
# ----------------------------------------------------------------------------
CASCADE_OPS = ["Cascade: channel → file", "Cascade: file → channel", "De-embed file at port 1 (left)",
               "De-embed file at port 2 (right)", "Flip ports"]
CMP_METRICS = ["Insertion loss", "Return loss S11", "Return loss S22", "Group delay S21", "Z-profile (port 1)"]
XT_ORDERS = ["Halves: 1..N/2 | N/2+1..N", "Odd = near end, even = far end"]
MASK_PRESETS = ["Off", "Custom", "100 mV × 0.40 UI", "50 mV × 0.30 UI", "25 mV × 0.20 UI"]


class ExtraTabs:
    """Equalisation/BER/mask support, limit lines, ILD/COM/causality, crosstalk, file comparison,
    cascade/de-embedding, vector fitting + SPICE export, time gating, Smith extras and exports."""

    # -- helpers ------------------------------------------------------------------
    def _eq_settings(self) -> dict[str, float]:
        fv = self._fv
        pre = min(max(fv(self.ffe_pre_var, 0.0), 0.0), 45.0) / 100.0
        post = min(max(fv(self.ffe_post_var, 0.0), 0.0), 45.0) / 100.0
        return {"ffe_pre": pre, "ffe_post": post, "ctle_boost_db": max(fv(self.ctle_db_var, 0.0), 0.0),
                "ctle_peak_ghz": max(fv(self.ctle_ghz_var, 0.0), 0.0),
                "dfe_taps": int(min(max(fv(self.dfe_var, 0.0), 0.0), 24)),
                "tx_rise_ps": self._tx_rise_setting(),
                "oversample": 256 if "256" in self.quality_var.get() else 128 if "128" in self.quality_var.get() else 64}

    def _tx_rise_setting(self) -> float | None:
        text = self.tx_rise_var.get().strip().lower()
        return None if text in ("", "auto") else max(self._fv(self.tx_rise_var, 0.0), 0.0)

    def _baud_hz(self) -> float:
        bps = SIGNALS.get(self.signal_var.get(), SIGNALS[DEFAULT_SIGNAL])["bps"]
        return self._fv(self.rate_var, 5.0) / bps * 1e9

    def _on_mask_preset(self) -> None:
        text = self.mask_var.get()
        if "mV" in text:
            h, _, rest = text.partition(" mV × ")
            self.mask_h_var.set(h.strip())
            self.mask_w_var.set(rest.replace("UI", "").strip())
        self._redraw_eye()

    def _draw_mask_overlay(self, ax: Any, r: dict[str, Any]) -> None:
        r.pop("mask_result", None)
        if self.mask_var.get() == "Off":
            return
        h = max(self._fv(self.mask_h_var, 50.0), 0.0) / 1e3
        w = min(max(self._fv(self.mask_w_var, 0.3), 0.01), 1.0)
        polys, hits, ok = mask_check(r, h, w)
        r["mask_result"] = (hits, ok)
        col = GREEN if ok else RED
        for poly in polys:
            ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=col, linewidth=1.6))
        ax.text(0.985, 0.03, f"Mask {'PASS' if ok else 'FAIL'}  ({hits:,} hits)", transform=ax.transAxes, ha="right",
                color=col, fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD, edgecolor=BORDER, alpha=0.9))

    def _push_history(self) -> None:
        import copy
        if self.channel is not None:
            self._history.append(copy.deepcopy(self.channel))
            del self._history[:-10]

    def _undo(self) -> None:
        if not self._history:
            messagebox.showinfo("Nothing to undo", "No earlier channel is stored.")
            return
        self.channel = self._history.pop()
        self.eye_result = None
        self._refresh_analysis()
        self._set_status("Restored the previous channel.")

    def _channel_network(self) -> Any:
        d = self.channel
        return rf.Network(frequency=rf.Frequency.from_f(d.frequency_hz, unit="hz"), s=d.s, z0=d.z0)

    # -- construction ----------------------------------------------------------------
    def _build_extra_tabs(self) -> None:
        self._history: list[ChannelData] = []
        self.cmp_items: list[tuple[str, ChannelData]] = []
        self.xt_net: Any = None
        self._xt_icn_v = 0.0
        self.vf: Any = None
        self.casc_net: Any = None
        S = tk.StringVar
        self.ber_var = S(value="12")
        self.il_lim_var = S(value="0:1, 5:10, 10:20")
        self.rl_lim_var = S(value="0:12, 5:10, 10:6")
        self.xt_order_var = S(value=XT_ORDERS[0])
        self.xt_victim_var = S(value="1")
        self.xt_label_var = S(value="no file loaded")
        self.cmp_metric_var = S(value=CMP_METRICS[0])
        self.casc_op_var = S(value=CASCADE_OPS[0])
        self.casc_label_var = S(value="no file loaded")
        self.vf_real_var, self.vf_cmplx_var = S(value="6"), S(value="16")
        self.vf_status_var = S(value="Press Fit to build a rational model of the channel.")
        self.gate_param_var = S(value="S11")
        self.gate_t1_var, self.gate_t2_var, self.gate_edge_var = S(value="0"), S(value="1"), S(value="0.1")
        self.gate_mode_var = S(value="Keep gate")
        self.sm_freq_var = S(value="5")
        self.sm_ga_var, self.sm_nf_var = S(value="-1, -2, -3"), S(value="0.5, 1, 2")
        self.export_status_var = S(value="Pick an export.")

        self.q_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.q_tab, text="Quality & EQ")
        self.q_nb = ttk.Notebook(self.q_tab, style="Inner.TNotebook")
        self.q_nb.pack(fill="both", expand=True, pady=(6, 0))
        self.t_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.t_tab, text="Tools")
        self.t_nb = ttk.Notebook(self.t_tab, style="Inner.TNotebook")
        self.t_nb.pack(fill="both", expand=True, pady=(6, 0))
        F, B = self._adv_field, self._adv_button

        def bath_c(bar: tk.Widget) -> None:
            F(bar, "BER target (10^-N)", lambda p: ttk.Entry(p, textvariable=self.ber_var, width=6))
            ttk.Label(bar, text="Noise σ, random jitter and EQ settings come from the cards under the eye controls.",
                      style="CardMuted.TLabel").pack(side="left", padx=(0, 12))
            B(bar, "Update", self._update_bath)

        def lim_c(bar: tk.Widget) -> None:
            F(bar, "Max insertion loss  f(GHz):dB, …", lambda p: ttk.Entry(p, textvariable=self.il_lim_var, width=26))
            F(bar, "Min return loss  f(GHz):dB, …", lambda p: ttk.Entry(p, textvariable=self.rl_lim_var, width=26))
            ttk.Label(bar, text="Defaults are placeholders — enter your specification.", style="CardMuted.TLabel").pack(side="left")
            B(bar, "Update", lambda: self._redraw("lim"))

        def qual_c(bar: tk.Widget) -> None:
            B(bar, "Update", lambda: self._redraw("qual"))
            B(bar, "Enforce passivity (σ ≤ 0.999)", self._enforce_passivity, accent=False)
            B(bar, "Undo", self._undo, accent=False)

        def xt_c(bar: tk.Widget) -> None:
            B(bar, "Load N-port file", self._xt_load, accent=False)
            ttk.Label(bar, textvariable=self.xt_label_var, style="CardMuted.TLabel", wraplength=170).pack(side="left", padx=(0, 12))
            F(bar, "Port order", lambda p: ttk.Combobox(p, textvariable=self.xt_order_var, values=XT_ORDERS, width=28, state="readonly"))
            F(bar, "Victim lane", lambda p: ttk.Entry(p, textvariable=self.xt_victim_var, width=5))
            B(bar, "Update", lambda: self._redraw("xt"))

        def cmp_c(bar: tk.Widget) -> None:
            B(bar, "Add file", self._cmp_add, accent=False)
            B(bar, "Clear", self._cmp_clear, accent=False)
            m = F(bar, "Metric", lambda p: ttk.Combobox(p, textvariable=self.cmp_metric_var, values=CMP_METRICS, width=20, state="readonly"))
            m.bind("<<ComboboxSelected>>", lambda _e: self._redraw("cmp"))

        def casc_c(bar: tk.Widget) -> None:
            B(bar, "Load 2-port file", self._casc_load, accent=False)
            ttk.Label(bar, textvariable=self.casc_label_var, style="CardMuted.TLabel", wraplength=170).pack(side="left", padx=(0, 12))
            F(bar, "Operation", lambda p: ttk.Combobox(p, textvariable=self.casc_op_var, values=CASCADE_OPS, width=30, state="readonly"))
            B(bar, "Apply to channel", self._casc_apply)
            B(bar, "Undo", self._undo, accent=False)

        def vf_c(bar: tk.Widget) -> None:
            F(bar, "Real poles", lambda p: ttk.Entry(p, textvariable=self.vf_real_var, width=5))
            F(bar, "Complex pairs", lambda p: ttk.Entry(p, textvariable=self.vf_cmplx_var, width=5))
            B(bar, "Fit", self._vf_fit)
            B(bar, "Enforce passivity", self._vf_passive, accent=False)
            B(bar, "Export SPICE", self._vf_export, accent=False)
            ttk.Label(bar, textvariable=self.vf_status_var, style="CardMuted.TLabel", wraplength=300).pack(side="left")

        def gate_c(bar: tk.Widget) -> None:
            F(bar, "Parameter", lambda p: ttk.Combobox(p, textvariable=self.gate_param_var, values=["S11", "S22", "S21", "S12"], width=6, state="readonly"))
            F(bar, "Start (ns)", lambda p: ttk.Entry(p, textvariable=self.gate_t1_var, width=7))
            F(bar, "Stop (ns)", lambda p: ttk.Entry(p, textvariable=self.gate_t2_var, width=7))
            F(bar, "Edge (ns)", lambda p: ttk.Entry(p, textvariable=self.gate_edge_var, width=7))
            F(bar, "Mode", lambda p: ttk.Combobox(p, textvariable=self.gate_mode_var, values=["Keep gate", "Remove gate"], width=11, state="readonly"))
            B(bar, "Update", lambda: self._redraw("gate"))
            B(bar, "Apply to channel", self._gate_apply, accent=False)
            B(bar, "Undo", self._undo, accent=False)

        def sm_c(bar: tk.Widget) -> None:
            F(bar, "Frequency (GHz)", lambda p: ttk.Entry(p, textvariable=self.sm_freq_var, width=8))
            F(bar, "Gain circles GA (dB)", lambda p: ttk.Entry(p, textvariable=self.sm_ga_var, width=14))
            F(bar, "Noise circles (dB above NFmin)", lambda p: ttk.Entry(p, textvariable=self.sm_nf_var, width=12))
            B(bar, "Update", lambda: self._redraw("sm"))

        self._adv_tab("Pulse / Bathtub", "bath", bath_c, self.q_nb)
        self._adv_tab("Limit lines", "lim", lim_c, self.q_nb)
        self._adv_tab("ILD / COM / causality", "qual", qual_c, self.q_nb)
        self._adv_tab("Crosstalk", "xt", xt_c, self.q_nb)
        self._adv_tab("Compare files", "cmp", cmp_c, self.q_nb)
        self._adv_tab("Cascade / De-embed", "casc", casc_c, self.t_nb)
        self._adv_tab("Vector fit / SPICE", "vf", vf_c, self.t_nb)
        self._adv_tab("Time gating", "gate", gate_c, self.t_nb)
        self._adv_tab("Smith+ / Matching", "sm", sm_c, self.t_nb)

        exp = ttk.Frame(self.t_nb)
        self.t_nb.add(exp, text="Export / Session")
        bar = ttk.Frame(exp, style="Card.TFrame", padding=(16, 14))
        bar.pack(fill="x", pady=(0, 6))
        for text, cmd in (("Figures → PNG folder", self._export_figures), ("S-parameters → CSV", self._export_csv),
                          ("HTML report", self._export_html), ("Save session", self._save_session),
                          ("Load session", self._load_session)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=(0, 10))
        ttk.Label(exp, textvariable=self.export_status_var, style="Muted.TLabel", padding=(16, 8)).pack(anchor="w")

        for name in ("bath", "lim", "qual", "xt", "cmp", "casc", "vf", "gate", "sm"):
            self._placeholder(name, "Load a Touchstone file first.")
        self._placeholder("bath", "Run the eye calculation first (needs a loaded channel).")
        self._placeholder("xt", "Load an N-port (4, 8, …) crosstalk file with the button above.")
        self._placeholder("vf", "Press Fit to build a rational (pole-residue) model of the channel.")

    def _refresh_extra(self) -> None:
        for name in ("lim", "qual", "cmp", "casc", "gate", "sm", "cond"):
            try:
                getattr(self, f"_draw_{name}")()
            except Exception as exc:
                self._placeholder(name, f"{name} analysis failed:\n{exc}")

    def _after_eye(self) -> None:
        for name in ("bath", "qual"):
            try:
                getattr(self, f"_draw_{name}")()
            except Exception as exc:
                self._placeholder(name, f"{name} analysis failed:\n{exc}")
        self._update_summary()

    def _update_bath(self) -> None:
        self._draw_bath()
        self._draw_qual()
        self._update_summary()

    # -- pulse response, bathtub, BER contour -----------------------------------------------
    def _draw_bath(self) -> None:
        r = self.eye_result
        if r is None:
            return self._placeholder("bath", "Run the eye calculation first (needs a loaded channel).")
        sigma = max(self._fv(self.noise_var, 1.0), 0.0) / 1e3
        rj = max(self._fv(self.rj_var, 0.0), 0.0)
        target = 10.0 ** (-abs(self._fv(self.ber_var, 12.0)))
        ba = ber_analysis(r, sigma, rj, target)
        pu = r["pulse"]
        fig = self.fig_bath
        fig.clf()
        ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()
        ax1.plot(pu["x_ui"], pu["y"] * 1e3, color=ACCENT)
        ks, cur = pu["k"], pu["cursors"] * 1e3
        ax1.vlines(ks, 0, cur, color=ACCENT2, linewidth=1.4)
        ax1.plot(ks, cur, "o", color=ACCENT2, markersize=4, label="UI-spaced cursors")
        if r["dfe_n"]:
            ax1.plot(np.arange(1, r["dfe_n"] + 1), r["dfe_taps_v"] * 1e3, "s", color=GREEN, markersize=5, label="DFE-cancelled")
        ax1.axhline(0, color=MUTED, linewidth=0.7)
        ax1.set_title("Pulse response (after FFE / CTLE)")
        ax1.set_xlabel("Time relative to main cursor (UI)")
        ax1.set_ylabel("mV")
        ax1.set_xlim(-8, 24)
        ax2.semilogy(ba["x_ui"], ba["ber_t"], color=ACCENT)
        ax2.axhline(target, color=RED, linestyle="--", linewidth=1, label=f"target {target:.0e}")
        ax2.set_title("Horizontal bathtub")
        ax2.set_xlabel("Sampling phase (UI)")
        ax2.set_ylabel("BER (symbol-error estimate)")
        ax2.set_ylim(1e-18, 1.0)
        ax3.semilogy(ba["y"] * 1e3, ba["ber_y"], color=ACCENT2)
        ax3.axhline(target, color=RED, linestyle="--", linewidth=1, label=f"target {target:.0e}")
        ax3.set_title("Vertical bathtub (centre eye, best phase)")
        ax3.set_xlabel("Decision threshold (mV)")
        ax3.set_ylabel("BER")
        ax3.set_ylim(1e-18, 1.0)
        z = np.log10(ba["contour"].T)
        ax4.imshow(z, origin="lower", aspect="auto", cmap="magma_r", vmin=-18, vmax=0,
                   extent=(0, 1, ba["y"][0] * 1e3, ba["y"][-1] * 1e3))
        try:
            cs = ax4.contour(ba["x_ui"], ba["y"] * 1e3, z, levels=[-3, -6, -9, -12], colors="white", linewidths=0.9)
            ax4.clabel(cs, fmt=lambda v: f"1e{int(v)}", fontsize=7)
        except Exception:
            pass
        ax4.set_title("BER contour (log10)")
        ax4.set_xlabel("Sampling phase (UI)")
        ax4.set_ylabel("Threshold (mV)")
        for ax in (ax1, ax2, ax3):
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)
        self._finish("bath")
        lines = [f"Settings: {r['eq_text']}",
                 f"Noise σ {sigma*1e3:.2f} mV, random jitter σ {rj:.2f} ps, BER target {target:.0e}",
                 f"Eye opening at target: width {ba['width_ui']:.3f} UI ({ba['width_ps']:.2f} ps), height {ba['height_v']*1e3:.2f} mV"]
        if "mask_result" in r:
            lines.append(f"Eye mask: {'PASS' if r['mask_result'][1] else 'FAIL'} ({r['mask_result'][0]:,} hits)")
        self._set_section("eqmask", "EQ / bathtub / mask", lines)

    # -- limit lines ---------------------------------------------------------------------------
    def _draw_lim(self) -> None:
        if self.channel is None:
            return self._placeholder("lim", "Load a Touchstone file first.")
        d, f = self.channel, self.channel.frequency_hz
        g = f / 1e9
        il = -_db(d.s[:, 1, 0])
        fi, li = parse_limit_points(self.il_lim_var.get())
        fr, lr = parse_limit_points(self.rl_lim_var.get())
        fig = self.fig_lim
        fig.clf()
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(g, il, color=ACCENT2, label="Insertion loss")
        ax2.plot(g, -_db(d.s[:, 0, 0]), color=ACCENT, label="RL S11")
        ax2.plot(g, -_db(d.s[:, 1, 1]), color=PURPLE, linestyle="--", label="RL S22")
        lines = []
        if len(fi) >= 2:
            m = (f >= fi[0]) & (f <= fi[-1])
            lim = np.interp(f, fi, li)
            ax1.plot(fi / 1e9, li, color=RED, linewidth=2, label="Limit (max IL)")
            if m.any():
                margin = lim[m] - il[m]
                lines.append(f"Insertion loss vs limit: {'PASS' if margin.min() >= 0 else 'FAIL'}, worst margin {margin.min():.2f} dB @ {g[m][margin.argmin()]:.4g} GHz")
        if len(fr) >= 2:
            m = (f >= fr[0]) & (f <= fr[-1])
            lim = np.interp(f, fr, lr)
            ax2.plot(fr / 1e9, lr, color=RED, linewidth=2, label="Limit (min RL)")
            if m.any():
                margin = np.minimum(-_db(d.s[m, 0, 0]), -_db(d.s[m, 1, 1])) - lim[m]
                lines.append(f"Return loss vs limit: {'PASS' if margin.min() >= 0 else 'FAIL'}, worst margin {margin.min():.2f} dB @ {g[m][margin.argmin()]:.4g} GHz")
        ax1.set_title("Insertion loss")
        ax2.set_title("Return loss")
        for ax in (ax1, ax2):
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("dB")
            ax.grid(True)
            ax.legend(loc="best")
        self._finish("lim")
        if lines:
            self._set_section("lim", "Limit lines", lines)

    # -- ILD, COM-like budget, causality ------------------------------------------------------------
    def _draw_qual(self) -> None:
        if self.channel is None:
            return self._placeholder("qual", "Load a Touchstone file first.")
        d, f = self.channel, self.channel.frequency_hz
        g = f / 1e9
        s21 = d.s[:, 1, 0]
        il = -_db(s21)
        basis = np.column_stack([np.ones_like(g), np.sqrt(g), g, g * g])
        coef = np.linalg.lstsq(basis, il, rcond=None)[0]
        fit = basis @ coef
        ild = il - fit
        t, x = impulse_full(f, s21)
        tau = float(np.nanmedian(group_delay(f, s21)))
        tau = tau if np.isfinite(tau) and tau > 0 else 0.0
        energy = float(np.sum(x ** 2))
        pre = float(np.sum(x[t < 0.5 * tau] ** 2) / energy * 100.0) if energy > 0 else float("nan")
        sig = float(passivity(d.s).max())

        fig = self.fig_qual
        fig.clf()
        ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()
        ax1.plot(g, il, color=ACCENT2, label="IL")
        ax1.plot(g, fit, "--", color=GREEN, label="a + b√f + c f + d f²")
        ax1.set_title("Insertion loss and smooth fit")
        ax1.set_ylabel("dB")
        ax2.plot(g, ild, color=PURPLE, label="ILD")
        ax2.axhline(0, color=MUTED, linewidth=0.7)
        ax2.set_title("Insertion loss deviation (ILD)")
        ax2.set_ylabel("dB")
        win = (t > -0.5e-9 - 2 * tau) & (t < 4 * tau + 1e-9)
        ax3.plot(t[win] * 1e9, x[win] * 1e-9, color=ACCENT, label="h(t) of S21")
        ax3.axvline(0.5 * tau * 1e9, color=RED, linestyle="--", linewidth=1, label="0.5 τ (pre-arrival limit)")
        ax3.set_title("Impulse response (causality view)")
        ax3.set_xlabel("Time (ns)")
        ax3.set_ylabel("1/ns")
        for ax in (ax1, ax2):
            ax.set_xlabel("Frequency (GHz)")
        for ax in (ax1, ax2, ax3):
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)

        lines = [f"ILD: RMS {np.sqrt(np.mean(ild ** 2)):.3f} dB, peak {np.max(np.abs(ild)):.3f} dB",
                 f"Causality (heuristic): {pre:.3f} % of the S21 impulse energy arrives before 0.5 × the median group delay "
                 f"({'looks causal' if pre < 1.0 else 'possible non-causality / truncation ringing'})",
                 f"Passivity: max σ(S) = {sig:.4f} ({'passive' if sig <= 1.0005 else 'ACTIVE — use Enforce passivity'})"]
        r = self.eye_result
        if r is not None:
            pu = r["pulse"]
            k, cur = pu["k"], pu["cursors"]
            main = float(abs(cur[k == 0][0]))
            isi = float(np.abs(cur[k < 0]).sum() + np.abs(cur[k > r["dfe_n"]]).sum())
            q = 7.03  # Gaussian Q for 1e-12
            xt = self._xt_icn_v * q
            nz = max(self._fv(self.noise_var, 1.0), 0.0) / 1e3 * q
            total = isi + math.sqrt(xt ** 2 + nz ** 2)
            com = 20.0 * math.log10(main / total) if total > 0 and main > 0 else float("nan")
            labels = ["Signal (main cursor)", "Residual ISI", "Crosstalk (Q·ICN)", "Noise (Q·σ)"]
            vals = [main * 1e3, isi * 1e3, xt * 1e3, nz * 1e3]
            ax4.bar(range(4), vals, color=[GREEN, ACCENT2, PURPLE, RED])
            ax4.set_xticks(range(4), labels, rotation=15, fontsize=8)
            ax4.set_ylabel("mV")
            ax4.set_title(f"COM-like budget: {com:.2f} dB")
            lines.append(f"COM-like (simplified, peak-distortion): {com:.2f} dB with {r['eq_text']}; "
                         f"signal {main*1e3:.1f} mV, ISI {isi*1e3:.1f} mV, crosstalk {xt*1e3:.1f} mV, noise {nz*1e3:.1f} mV. "
                         "Not an IEEE 802.3 COM.")
            ax4.grid(True, axis="y")
        else:
            ax4.axis("off")
            ax4.text(0.5, 0.5, "Run the eye calculation\nfor the COM-like budget", ha="center", va="center", color=MUTED, transform=ax4.transAxes)
        self._finish("qual")
        self._set_section("qual", "Channel quality", lines)

    def _enforce_passivity(self) -> None:
        if self.channel is None:
            return
        self._push_history()
        self.channel.s = enforce_passivity(self.channel.s)
        self.eye_result = None
        self._refresh_analysis()
        self._set_status("Passivity enforced on the active channel (singular values clipped to 0.999).")

    # -- crosstalk -----------------------------------------------------------------------------------
    def _xt_load(self) -> None:
        path = filedialog.askopenfilename(title="Select a multi-port Touchstone file",
                                          filetypes=[("Touchstone", "*.s4p *.s6p *.s8p *.s12p *.S4P *.S8P"), ("All files", "*.*")])
        if not path:
            return
        try:
            net = rf.Network(path)
            if net.nports < 4 or net.nports % 2:
                raise ValueError("Crosstalk needs an even number of ports (4 or more).")
        except Exception as exc:
            messagebox.showerror("Could not load file", str(exc))
            return
        self.xt_net = net
        self.xt_label_var.set(f"{Path(path).name} ({net.nports} ports)")
        self._redraw("xt")

    def _draw_xt(self) -> None:
        net = self.xt_net
        if net is None:
            return self._placeholder("xt", "Load an N-port (4, 8, …) crosstalk file with the button above.")
        n = net.nports
        lanes = n // 2
        if self.xt_order_var.get().startswith("Halves"):
            near, far = list(range(lanes)), list(range(lanes, n))
        else:
            near, far = list(range(0, n, 2)), list(range(1, n, 2))
        v = min(max(int(self._fv(self.xt_victim_var, 1.0)) - 1, 0), lanes - 1)
        aggs = [i for i in range(lanes) if i != v]
        f = np.asarray(net.f, dtype=float)
        g = f / 1e9
        s = np.asarray(net.s, dtype=complex)
        il = s[:, far[v], near[v]]
        nexts = [s[:, near[v], near[i]] for i in aggs]
        fexts = [s[:, far[v], near[i]] for i in aggs]
        ps = lambda xs: 10.0 * np.log10(np.maximum(sum(np.abs(x) ** 2 for x in xs), 1e-30))  # noqa: E731
        fig = self.fig_xt
        fig.clf()
        ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()
        ax1.plot(g, _db(il), color=ACCENT2, label=f"Victim lane {v + 1} IL")
        ax1.set_title("Victim through response")
        for i, x in zip(aggs, nexts):
            ax2.plot(g, _db(x), linewidth=1, alpha=0.8, label=f"NEXT {i + 1}→{v + 1}")
        ax2.plot(g, ps(nexts), color="white", linewidth=2, label="PSNEXT")
        ax2.set_title("Near-end crosstalk")
        for i, x in zip(aggs, fexts):
            ax3.plot(g, _db(x), linewidth=1, alpha=0.8, label=f"FEXT {i + 1}→{v + 1}")
        ax3.plot(g, ps(fexts), color="white", linewidth=2, label="PSFEXT")
        ax3.set_title("Far-end crosstalk")
        ax4.plot(g, _db(il) - ps(nexts), color=ACCENT, label="PSACR-N")
        ax4.plot(g, _db(il) - ps(fexts), color=GREEN, label="PSACR-F")
        ax4.axhline(0, color=RED, linestyle="--", linewidth=1)
        ax4.set_title("Attenuation-to-crosstalk ratio")
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("dB")
            ax.grid(True)
            ax.legend(loc="best", fontsize=7, ncol=2)
        self._finish("xt")
        fb = max(self._baud_hz(), 1.0)
        amp = max(self._fv(self.swing_var, 1.0), 0.0) / 2.0
        icn_n, icn_f = icn_rms(f, nexts, fb, amp), icn_rms(f, fexts, fb, amp)
        self._xt_icn_v = math.hypot(icn_n, icn_f)
        self._set_section("xt", "Crosstalk", [
            f"{n}-port file, {len(aggs)} aggressor(s) on victim lane {v + 1}",
            f"ICN (802.3-style weighting, fb = {fb/1e9:.4g} GBd, A = {amp:g} V): NEXT {icn_n*1e3:.3f} mV, FEXT {icn_f*1e3:.3f} mV, "
            f"total {self._xt_icn_v*1e3:.3f} mV RMS (feeds the COM-like budget)"])
        if self.eye_result is not None:
            self._draw_qual()

    # -- compare files ---------------------------------------------------------------------------------
    def _cmp_add(self) -> None:
        path = filedialog.askopenfilename(title="Select a Touchstone file to overlay",
                                          filetypes=[("Touchstone", "*.s2p *.s4p *.S2P *.S4P"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.cmp_items.append((Path(path).name, read_touchstone(path, self.pair_var.get())))
        except Exception as exc:
            messagebox.showerror("Could not load file", str(exc))
            return
        self._redraw("cmp")

    def _cmp_clear(self) -> None:
        self.cmp_items.clear()
        self._redraw("cmp")

    def _draw_cmp(self) -> None:
        if self.channel is None:
            return self._placeholder("cmp", "Load a Touchstone file first.")
        metric = self.cmp_metric_var.get()
        items = [("Active channel", self.channel)] + self.cmp_items
        fig = self.fig_cmp
        fig.clf()
        ax = fig.add_subplot(111)
        cols = [ACCENT, ACCENT2, GREEN, PURPLE, RED, "#ffd866", "#78dce8", "#ff8ab4"]
        for i, (name, ch) in enumerate(items):
            f = ch.frequency_hz
            if metric == "Insertion loss":
                x, y, xl, yl = f / 1e9, -_db(ch.s[:, 1, 0]), "Frequency (GHz)", "Insertion loss (dB)"
            elif metric == "Return loss S11":
                x, y, xl, yl = f / 1e9, -_db(ch.s[:, 0, 0]), "Frequency (GHz)", "Return loss (dB)"
            elif metric == "Return loss S22":
                x, y, xl, yl = f / 1e9, -_db(ch.s[:, 1, 1]), "Frequency (GHz)", "Return loss (dB)"
            elif metric == "Group delay S21":
                x, y, xl, yl = f / 1e9, group_delay(f, ch.s[:, 1, 0]) * 1e12, "Frequency (GHz)", "Group delay (ps)"
            else:
                t, st, _ = time_response(f, ch.s[:, 0, 0], max(self._fv(self.tr_var, 35.0), 0.0) * 1e-12, self.win_var.get())
                z0 = float(np.median(np.real(ch.z0[:, 0])))
                x, y, xl, yl = t * 1e9, impedance_from_step(st, z0), "Time (ns)", "Impedance (Ω)"
            ax.plot(x, y, color=cols[i % len(cols)], label=name, linewidth=2.0 if i == 0 else 1.4)
            if metric.startswith("Z-profile"):
                ax.set_xlim(0, min(float(x[-1]), 6.0 * float(np.nanmax(np.abs(group_delay(f, ch.s[:, 1, 0]))) * 1e9) + 1.0))
        ax.set_title(f"Comparison — {metric}")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True)
        ax.legend(loc="best")
        self._finish("cmp")

    # -- cascade / de-embedding ---------------------------------------------------------------------------
    def _casc_load(self) -> None:
        path = filedialog.askopenfilename(title="Select a 2-port Touchstone file",
                                          filetypes=[("Touchstone", "*.s2p *.S2P"), ("All files", "*.*")])
        if not path:
            return
        try:
            net = rf.Network(path)
            if net.nports != 2:
                raise ValueError("The file must be a 2-port.")
        except Exception as exc:
            messagebox.showerror("Could not load file", str(exc))
            return
        self.casc_net = net
        self.casc_label_var.set(Path(path).name)

    def _casc_result(self) -> Any:
        main = self._channel_network()
        op = self.casc_op_var.get()
        if op == "Flip ports":
            return main.flipped()
        if self.casc_net is None:
            raise ValueError("Load a 2-port file first.")
        other = self.casc_net
        if len(other.f) != len(main.f) or not np.allclose(other.f, main.f):
            other = other.interpolate(main.frequency)
        if op == CASCADE_OPS[0]:
            return main ** other
        if op == CASCADE_OPS[1]:
            return other ** main
        if op == CASCADE_OPS[2]:
            return other.inv ** main
        return main ** other.inv

    def _draw_casc(self) -> None:
        if self.channel is None:
            return self._placeholder("casc", "Load a Touchstone file first.")
        d = self.channel
        g = d.frequency_hz / 1e9
        try:
            res = self._casc_result() if (self.casc_net is not None or self.casc_op_var.get() == "Flip ports") else None
        except Exception as exc:
            res = None
            self.casc_label_var.set(f"error: {exc}")
        fig = self.fig_casc
        fig.clf()
        ax1, ax2 = fig.subplots(1, 2)
        for ax, (i, j), title in ((ax1, (1, 0), "S21 (dB)"), (ax2, (0, 0), "S11 (dB)")):
            ax.plot(g, _db(d.s[:, i, j]), color=MUTED, label="Active channel")
            if res is not None:
                ax.plot(g, _db(res.s[:, i, j]), color=ACCENT2, label="Result of the selected operation (preview)")
            ax.set_title(title)
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)
        self._finish("casc")

    def _casc_apply(self) -> None:
        if self.channel is None:
            return
        try:
            res = self._casc_result()
        except Exception as exc:
            messagebox.showerror("Operation failed", str(exc))
            return
        self._push_history()
        z0 = np.asarray(res.z0, dtype=complex)
        if z0.ndim == 1:
            z0 = np.repeat(z0[:, None], 2, axis=1)
        self.channel = ChannelData(Path(f"{self.channel.path.stem}-{self.casc_op_var.get().split(':')[0].split(' ')[0].lower()}.s2p"),
                                   np.asarray(res.f, dtype=float), np.asarray(res.s, dtype=complex), z0)
        self.eye_result = None
        self._refresh_analysis()
        self._set_status("Applied: " + self.casc_op_var.get())

    # -- vector fitting and SPICE -------------------------------------------------------------------------------
    def _vf_fit(self) -> None:
        if self.channel is None:
            return messagebox.showinfo("Load a channel", "Load a Touchstone file first.")
        try:
            from skrf.vectorFitting import VectorFitting
            net = self._channel_network()
            vf = VectorFitting(net)
            vf.vector_fit(n_poles_real=int(self._fv(self.vf_real_var, 3)), n_poles_cmplx=int(self._fv(self.vf_cmplx_var, 4)))
            self.vf, self.vf_net = vf, net
            err = float(np.max(vf.get_rms_error()))
            viol = vf.passivity_test()
            self.vf_status_var.set(f"RMS error {err:.2e}; passivity violation bands: {len(viol)}")
        except Exception as exc:
            self.vf_status_var.set(f"Vector fit failed: {exc}")
            messagebox.showerror("Vector fitting failed", str(exc))
            return
        self._draw_vf()

    def _vf_passive(self) -> None:
        if self.vf is None:
            return messagebox.showinfo("Fit first", "Run Fit first.")
        try:
            self.vf.passivity_enforce()
            self.vf_status_var.set("Passivity enforced; RMS error " + f"{float(np.max(self.vf.get_rms_error())):.2e}")
        except Exception as exc:
            self.vf_status_var.set(f"Passivity enforcement failed: {exc}")
            return
        self._draw_vf()

    def _vf_export(self) -> None:
        if self.vf is None:
            return messagebox.showinfo("Fit first", "Run Fit first.")
        path = filedialog.asksaveasfilename(defaultextension=".sp", filetypes=[("SPICE netlist", "*.sp *.cir"), ("All files", "*.*")])
        if path:
            try:
                self.vf.write_spice_subcircuit_s(path, fitted_model_name="channel_fit")
                self.vf_status_var.set(f"Wrote SPICE subcircuit 'channel_fit' to {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    def _draw_vf(self) -> None:
        if self.vf is None:
            return self._placeholder("vf", "Press Fit to build a rational (pole-residue) model of the channel.")
        f, g = self.vf_net.f, self.vf_net.f / 1e9
        fig = self.fig_vf
        fig.clf()
        ax1, ax2, ax3 = fig.subplots(1, 3)
        for ax, (i, j), title in ((ax1, (1, 0), "S21 (dB)"), (ax2, (0, 0), "S11 (dB)")):
            fit = self.vf.get_model_response(i, j, f)
            ax.plot(g, _db(self.vf_net.s[:, i, j]), color=MUTED, label="Measured")
            ax.plot(g, _db(fit), "--", color=ACCENT2, label="Rational fit")
            ax.set_title(title)
            ax3.plot(g, _db(fit - self.vf_net.s[:, i, j]), label=f"S{i + 1}{j + 1} error")
        ax3.set_title("Fit error (dB)")
        for ax in (ax1, ax2, ax3):
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)
        self._finish("vf")

    # -- time gating -------------------------------------------------------------------------------------------------
    def _gate_inputs(self) -> tuple[int, int, float, float, float, bool]:
        name = self.gate_param_var.get()
        i, j = int(name[1]) - 1, int(name[2]) - 1
        t1, t2 = self._fv(self.gate_t1_var, 0.0) * 1e-9, self._fv(self.gate_t2_var, 1.0) * 1e-9
        return i, j, min(t1, t2), max(t1, t2), max(self._fv(self.gate_edge_var, 0.1), 0.0) * 1e-9, self.gate_mode_var.get() == "Keep gate"

    def _draw_gate(self) -> None:
        if self.channel is None:
            return self._placeholder("gate", "Load a Touchstone file first.")
        d, f = self.channel, self.channel.frequency_hz
        i, j, t1, t2, edge, keep = self._gate_inputs()
        h = d.s[:, i, j]
        hg, t, x, gate = gate_response(f, h, t1, t2, edge, keep)
        fig = self.fig_gate
        fig.clf()
        ax1, ax2 = fig.subplots(1, 2)
        span = max(t2 + 4 * edge, 1e-9) * 1.5
        m = (t >= -0.1 * span) & (t <= span)
        ax1.plot(t[m] * 1e9, np.abs(x[m]) * 1e-9, color=ACCENT, label="|h(t)|")
        ax1.fill_between(t[m] * 1e9, 0, gate[m] * float(np.max(np.abs(x[m])) * 1e-9), color=ACCENT2, alpha=0.25, label="Gate")
        ax1.set_title(f"{self.gate_param_var.get()} impulse response and gate")
        ax1.set_xlabel("Time (ns)")
        ax1.set_ylabel("1/ns")
        ax2.plot(f / 1e9, _db(h), color=MUTED, label="Original")
        ax2.plot(f / 1e9, _db(hg), color=ACCENT2, label="Gated")
        ax2.set_title(f"{self.gate_param_var.get()} magnitude")
        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel("dB")
        for ax in (ax1, ax2):
            ax.grid(True)
            ax.legend(loc="best")
        self._finish("gate")

    def _gate_apply(self) -> None:
        if self.channel is None:
            return
        i, j, t1, t2, edge, keep = self._gate_inputs()
        self._push_history()
        hg = gate_response(self.channel.frequency_hz, self.channel.s[:, i, j], t1, t2, edge, keep)[0]
        self.channel.s = self.channel.s.copy()
        self.channel.s[:, i, j] = hg
        self.eye_result = None
        self._refresh_analysis()
        self._set_status(f"Gated S{i + 1}{j + 1} applied to the active channel.")

    # -- Smith extras and matching -----------------------------------------------------------------------------------------
    @staticmethod
    def _smith_grid(ax: Any, title: str) -> None:
        th = np.linspace(0, 2 * np.pi, 720)
        for r in (0.2, 0.5, 1.0, 2.0, 5.0):
            ax.plot(r / (1 + r) + 1 / (1 + r) * np.cos(th), 1 / (1 + r) * np.sin(th), color=GRID, linewidth=0.8)
        for x in (0.2, 0.5, 1.0, 2.0, 5.0):
            for sg in (1, -1):
                px, py = 1 + np.cos(th) / x, sg / x + np.sin(th) / x
                ok = px ** 2 + py ** 2 <= 1.0005
                ax.plot(np.where(ok, px, np.nan), np.where(ok, py, np.nan), color=GRID, linewidth=0.8)
        ax.plot(np.cos(th), np.sin(th), color=MUTED, linewidth=1.2)
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_title(title, fontsize=10)

    @staticmethod
    def _circle(ax: Any, c: complex, r: float, **kw: Any) -> None:
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(c.real + r * np.cos(th), c.imag + r * np.sin(th), **kw)

    def _draw_sm(self) -> None:
        if self.channel is None:
            return self._placeholder("sm", "Load a Touchstone file first.")
        d, f = self.channel, self.channel.frequency_hz
        idx = int(np.argmin(np.abs(f - self._fv(self.sm_freq_var, 5.0) * 1e9)))
        f0, s = float(f[idx]), d.s[idx]
        z1 = self._z0s()[0]
        fig = self.fig_sm
        fig.clf()
        axs, axl, axt = fig.subplots(1, 3)
        self._smith_grid(axs, f"Source plane Γs @ {f0/1e9:.4g} GHz")
        self._smith_grid(axl, "Load plane ΓL")
        (cs, rs), (cl, rl) = stability_circles(s)
        for ax, c, r, lab in ((axs, cs, rs, "Source stability circle"), (axl, cl, rl, "Load stability circle")):
            if np.isfinite(r):
                self._circle(ax, c, r, color=RED, linewidth=1.6, label=lab)
        st = stability_metrics(s[None])
        lines = [f"At {f0/1e9:.4g} GHz: K = {float(st['K'][0]):.3g}, |Δ| = {abs(complex(st['delta'][0])):.3g}, μ = {float(st['mu'][0]):.3g}"]
        for k, txt in enumerate(self.sm_ga_var.get().replace(";", ",").split(",")):
            try:
                circ = available_gain_circle(s, 10.0 ** (float(txt) / 10.0))
            except ValueError:
                continue
            if circ is not None:
                self._circle(axs, circ[0], circ[1], color=GREEN, linewidth=1.1, label="Available-gain circles" if k == 0 else None)
        nz = d.noise
        if nz is not None and len(nz["freq_hz"]):
            fn = nz["freq_hz"]
            fmin = float(np.interp(f0, fn, nz["nfmin_db"]))
            gopt = complex(np.interp(f0, fn, nz["gopt"].real), np.interp(f0, fn, nz["gopt"].imag))
            rn = float(np.interp(f0, fn, nz["rn"])) / nz["ref"]
            axs.plot(gopt.real, gopt.imag, "*", color=ACCENT2, markersize=9, label="Γopt")
            for k, txt in enumerate(self.sm_nf_var.get().replace(";", ",").split(",")):
                try:
                    nf = fmin + float(txt)
                except ValueError:
                    continue
                circ = noise_circle(10.0 ** (fmin / 10.0), gopt, max(rn, 1e-9), 10.0 ** (nf / 10.0))
                if circ is not None:
                    self._circle(axs, circ[0], circ[1], color=ACCENT2, linewidth=1.1, linestyle="--", label="Noise circles" if k == 0 else None)
            lines.append(f"NFmin {fmin:.2f} dB at this frequency, Γopt = {abs(gopt):.2f}∠{np.degrees(np.angle(gopt)):.0f}°")
        g11, g22 = s[0, 0], s[1, 1]
        axs.plot(np.conj(g11).real, np.conj(g11).imag, "o", color=ACCENT, markersize=5, label="S11*")
        axl.plot(np.conj(g22).real, np.conj(g22).imag, "o", color=ACCENT, markersize=5, label="S22*")
        for ax in (axs, axl):
            ax.legend(loc="lower left", fontsize=7)
        axt.axis("off")
        w = 2 * np.pi * f0
        zin = z1 * (1 + g11) / (1 - g11) if abs(1 - g11) > 1e-9 else complex("nan")
        lines.append(f"Zin (port 1) = {zin.real:.2f} {zin.imag:+.2f}j Ω; L-match to {z1:g} Ω:")
        sols = l_match(zin, z1, w) if np.isfinite(zin) else []
        lines += [f"  • {s_}" for s_ in sols] or ["  (no L-section solution for this load; use a different topology)"]
        axt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", color=FG, fontsize=8, transform=axt.transAxes, wrap=True)
        self._finish("sm")
        self._set_section("sm", "Smith+ / matching", lines)

    # -- export / session -------------------------------------------------------------------------------------------------------
    def _all_figures(self) -> list[tuple[str, Any]]:
        base = [("sparams", self.fig_s), ("loss", self.fig_loss), ("impedance_smith", self.fig_z), ("eye", self.fig_eye)]
        extra = [(n[4:], getattr(self, n)) for n in sorted(dir(self)) if n.startswith("fig_") and n not in
                 ("fig_s", "fig_loss", "fig_z", "fig_eye")]
        return base + extra

    def _export_figures(self) -> None:
        folder = filedialog.askdirectory(title="Choose a folder for the PNG files")
        if not folder:
            return
        count = 0
        for name, fig in self._all_figures():
            try:
                fig.savefig(Path(folder) / f"si_{name}.png", dpi=150)
                count += 1
            except Exception:
                pass
        self.export_status_var.set(f"Saved {count} figure(s) to {folder}")

    def _export_csv(self) -> None:
        if self.channel is None:
            return messagebox.showinfo("Nothing to export", "Load a channel first.")
        import csv
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        d, f = self.channel, self.channel.frequency_hz
        cols: dict[str, np.ndarray] = {"freq_GHz": f / 1e9}
        for (i, j, nm) in ((0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")):
            cols[f"{nm}_dB"] = _db(d.s[:, i, j])
            cols[f"{nm}_deg"] = np.degrees(np.angle(d.s[:, i, j]))
        cols["IL_dB"] = -_db(d.s[:, 1, 0])
        cols["groupdelay_S21_ps"] = group_delay(f, d.s[:, 1, 0]) * 1e12
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols.keys())
            w.writerows(zip(*cols.values()))
        self.export_status_var.set(f"Wrote {len(f)} rows to {Path(path).name}")

    def _export_html(self) -> None:
        import base64
        import html
        import io
        path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML", "*.html")])
        if not path:
            return
        parts = ["<html><head><meta charset='utf-8'><title>SI report</title></head>"
                 "<body style='font-family:sans-serif;background:#0b0f19;color:#e6eaf2;max-width:1100px;margin:auto'>",
                 "<h1>Signal Integrity report</h1><pre style='background:#151c2c;padding:12px;white-space:pre-wrap'>"
                 + html.escape(self.summary_text.get("1.0", "end")) + "</pre>"]
        for name, fig in self._all_figures():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            parts.append(f"<h3>{html.escape(name)}</h3><img style='max-width:100%' src='data:image/png;base64,"
                         + base64.b64encode(buf.getvalue()).decode() + "'>")
        parts.append("</body></html>")
        Path(path).write_text("".join(parts), encoding="utf-8")
        self.export_status_var.set(f"Wrote report {Path(path).name}")

    _SESSION_VARS = ("rate_var", "swing_var", "bits_var", "cmap_var", "signal_var", "ffe_pre_var", "ffe_post_var",
                     "ctle_db_var", "ctle_ghz_var", "dfe_var", "mask_var", "mask_h_var", "mask_w_var", "noise_var",
                     "rj_var", "tr_var", "er_var", "span_var", "win_var", "rs_var", "xs_var", "rl_var", "xl_var",
                     "par_var", "fmt_var", "pair_var", "len_var", "fitpct_var", "ber_var", "il_lim_var", "rl_lim_var")

    def _save_session(self) -> None:
        import json
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Session", "*.json")])
        if not path:
            return
        data = {k: getattr(self, k).get() for k in self._SESSION_VARS if hasattr(self, k)}
        data["file"] = str(self.channel.path) if self.channel is not None else ""
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.export_status_var.set(f"Session saved to {Path(path).name}")

    def _load_session(self) -> None:
        import json
        path = filedialog.askopenfilename(filetypes=[("Session", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            return messagebox.showerror("Could not read session", str(exc))
        for k in self._SESSION_VARS:
            if k in data and hasattr(self, k):
                getattr(self, k).set(data[k])
        target = data.get("file", "")
        if target and Path(target).exists():
            self.load_s2p(target)
            self.export_status_var.set(f"Session restored from {Path(path).name}")
        else:
            self.export_status_var.set("Settings restored; the channel file was not found — load it manually.")


# ----------------------------------------------------------------------------
# Ultimate additions: terminations, discontinuities, conditioning, batch, preferences
# ----------------------------------------------------------------------------
PREFS_PATH = Path.home() / ".si_analyzer.json"


def load_prefs() -> dict[str, Any]:
    try:
        import json
        return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_prefs(prefs: dict[str, Any]) -> None:
    try:
        import json
        PREFS_PATH.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except Exception:
        pass


def terminated_response(s: np.ndarray, gs: complex, gl: complex) -> np.ndarray:
    """Voltage transfer to the load with arbitrary source/load reflection coefficients.

    Normalised so that gs = gl = 0 returns exactly S21 (matched source and load). Source and load
    reflections (multiple bounces) are included, which is what makes a mismatched eye ring.
    """
    s11, s12, s21, s22 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
    with np.errstate(all="ignore"):
        gin = s11 + s12 * s21 * gl / (1.0 - s22 * gl)
        return s21 * (1.0 - gs) * (1.0 + gl) / ((1.0 - s22 * gl) * (1.0 - gs * gin))


def find_discontinuities(t: np.ndarray, z: np.ndarray, z0: float, span: float, max_n: int = 6) -> list[tuple[float, float]]:
    """Strongest impedance excursions (time, Z) inside the viewed window, in time order."""
    from scipy.signal import find_peaks
    m = t <= span
    dev = np.abs(z[m] - z0)
    thr = max(2.0, 0.03 * z0)
    pk, _ = find_peaks(dev, height=thr, prominence=thr / 2.0)
    pk = sorted(pk, key=lambda i: -dev[i])[:max_n]
    return [(float(t[m][i]), float(z[m][i])) for i in sorted(pk)]


def extrapolate_dc(f: np.ndarray, s: np.ndarray, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Prepend a DC point: real part extrapolated in f^2 from the first k points, imaginary part = 0."""
    if f[0] <= 0:
        return f, s
    k = min(k, len(f))
    dc = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            dc[i, j] = np.polyfit(f[:k] ** 2, s[:k, i, j].real, 1)[1]
    dc = 0.5 * (dc + dc.T)
    u, sv, vh = np.linalg.svd(dc)
    dc = (u * np.minimum(sv, 0.9999)) @ vh
    return np.concatenate([[0.0], f]), np.concatenate([dc[None], s], axis=0)


def resample_uniform(f: np.ndarray, s: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.interpolate import interp1d
    fn = np.linspace(float(f[0]), float(f[-1]), int(n))
    kind = "cubic" if len(f) >= 4 else "linear"
    out = np.empty((len(fn), 2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            out[:, i, j] = (interp1d(f, s[:, i, j].real, kind=kind)(fn) + 1j * interp1d(f, s[:, i, j].imag, kind=kind)(fn))
    return fn, out


BATCH_COLUMNS = ("File", "Ports", "fmax GHz", "IL@Nyq dB", "IL@fmax dB", "RL S11 dB", "RL S22 dB",
                 "Re Z11 Ω", "Z min Ω", "Z max Ω", "σmax(S)", "Eye H mV", "Eye W ps")


def batch_row(path: Path, nyq_hz: float, pairing: str, rate: float, swing: float, signal: str,
              eq: dict[str, float], with_eye: bool, tr_ps: float, window: str) -> list[str]:
    ch = read_touchstone(path, pairing)
    f, s = ch.frequency_hz, ch.s
    il = -_db(s[:, 1, 0])
    z1 = float(np.median(np.real(ch.z0[:, 0])))
    zd = impedance_from_reflection(s[:, 0, 0], np.real(ch.z0[:, 0]))
    t, st, _ = time_response(f, s[:, 0, 0], tr_ps * 1e-12, window)
    tau = float(np.nanmedian(np.abs(group_delay(f, s[:, 1, 0]))))
    span = max(6.0 * (tau if np.isfinite(tau) else 0.0) + 8.0 * tr_ps * 1e-12, 20.0 / f[-1])
    z = impedance_from_step(st, z1)[t <= span]
    nyq = f"{np.interp(nyq_hz, f, il):.2f}" if f[0] <= nyq_hz <= f[-1] else "n/a"
    row = [path.name, str(ch.ports), f"{f[-1]/1e9:.4g}", nyq, f"{il[-1]:.2f}", f"{(-_db(s[:, 0, 0])).min():.2f}",
           f"{(-_db(s[:, 1, 1])).min():.2f}", f"{np.nanmedian(np.real(zd)):.1f}", f"{np.nanmin(z):.1f}",
           f"{np.nanmax(z):.1f}", f"{passivity(s).max():.3f}", "", ""]
    if with_eye:
        r = make_eye_and_jitter(f, s[:, 1, 0], rate, swing, total_bits=16_384, signal=signal, **{**eq, "oversample": 64})
        row[-2], row[-1] = f"{r['eye_height_raw_v']*1e3:.1f}", f"{r['eye_width_ps']:.1f}"
    return row


class UltimateTabs:
    """File menu with recent files, port-mapping dialog, conditioning tools and batch processing."""

    # -- menu, preferences ---------------------------------------------------------------
    def _build_menu(self) -> None:
        self.prefs = load_prefs()
        opts = dict(tearoff=0, bg=PANEL, fg=FG, activebackground=ACCENT, activeforeground="#04111c", borderwidth=0)
        bar = tk.Menu(self, **opts)
        fm = tk.Menu(bar, **opts)
        fm.add_command(label="Open…    Ctrl+O", command=self._choose_s2p)
        fm.add_command(label="Open with port mapping…", command=lambda: self._choose_s2p(force_map=True))
        self._recent_menu = tk.Menu(fm, **opts)
        fm.add_cascade(label="Recent files", menu=self._recent_menu)
        fm.add_separator()
        fm.add_command(label="Save session…", command=self._save_session)
        fm.add_command(label="Load session…", command=self._load_session)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.destroy)
        bar.add_cascade(label="File", menu=fm)
        self.config(menu=bar)
        self.bind("<Control-o>", lambda _e: self._choose_s2p())
        self._rebuild_recent()

    def _rebuild_recent(self) -> None:
        self._recent_menu.delete(0, "end")
        recent = [p for p in self.prefs.get("recent", []) if Path(p).exists()]
        for p in recent:
            self._recent_menu.add_command(label=Path(p).name, command=lambda q=p: self.load_s2p(q))
        if not recent:
            self._recent_menu.add_command(label="(none)", state="disabled")

    def _remember(self, path: str | Path) -> None:
        p = str(path)
        recent = [q for q in self.prefs.get("recent", []) if q != p]
        self.prefs["recent"] = [p] + recent[:7]
        self.prefs["last_dir"] = str(Path(p).parent)
        save_prefs(self.prefs)
        self._rebuild_recent()

    # -- port mapping dialog -----------------------------------------------------------------
    def _ask_port_map(self, nports: int, path: str | Path) -> tuple | None:
        win = tk.Toplevel(self)
        win.title("Port mapping")
        win.configure(bg=BG)
        win.transient(self)
        ttk.Style(self).configure("TRadiobutton", background=BG, foreground=FG)
        ports = [str(i + 1) for i in range(nports)]
        mode = tk.StringVar(value="se")
        se_vars = [tk.StringVar(value=ports[0]), tk.StringVar(value=ports[min(1, nports - 1)])]
        diff_vars = [tk.StringVar(value=ports[min(k, nports - 1)]) for k in range(4)]
        result: dict[str, Any] = {"map": None}
        ttk.Label(win, text=f"{Path(path).name} has {nports} ports.\nChoose which ports form the channel to analyse.",
                  padding=(14, 12)).pack(anchor="w")
        frm = ttk.Frame(win, padding=(14, 0))
        frm.pack(fill="x")
        ttk.Radiobutton(frm, text="Single-ended   input port", variable=mode, value="se").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(frm, values=ports, textvariable=se_vars[0], width=4, state="readonly").grid(row=0, column=1, padx=6)
        ttk.Label(frm, text="output port").grid(row=0, column=2)
        ttk.Combobox(frm, values=ports, textvariable=se_vars[1], width=4, state="readonly").grid(row=0, column=3, padx=6)
        if nports >= 4:
            ttk.Radiobutton(frm, text="Differential   input pair (+, −)", variable=mode, value="diff").grid(row=1, column=0, sticky="w", pady=4)
            for k, col in ((0, 1), (1, 2)):
                ttk.Combobox(frm, values=ports, textvariable=diff_vars[k], width=4, state="readonly").grid(row=1, column=col, padx=6)
            ttk.Label(frm, text="output pair (+, −)").grid(row=2, column=0, sticky="e")
            for k, col in ((2, 1), (3, 2)):
                ttk.Combobox(frm, values=ports, textvariable=diff_vars[k], width=4, state="readonly").grid(row=2, column=col, padx=6)

        def ok() -> None:
            if mode.get() == "se":
                a, b = int(se_vars[0].get()) - 1, int(se_vars[1].get()) - 1
                if a == b:
                    return messagebox.showerror("Invalid mapping", "Input and output ports must differ.", parent=win)
                result["map"] = ("se", a, b)
            else:
                sel = [int(v.get()) - 1 for v in diff_vars]
                if len(set(sel)) != 4:
                    return messagebox.showerror("Invalid mapping", "The four differential ports must all differ.", parent=win)
                result["map"] = ("diff", *sel)
            win.destroy()

        row = ttk.Frame(win, padding=14)
        row.pack(fill="x")
        ttk.Button(row, text="OK", style="Accent.TButton", command=ok).pack(side="right")
        ttk.Button(row, text="Cancel", command=win.destroy).pack(side="right", padx=8)
        win.grab_set()
        self.wait_window(win)
        return result["map"]

    # -- construction ---------------------------------------------------------------------------
    def _build_ultimate_tabs(self) -> None:
        S = tk.StringVar
        self.z0_new_var, self.npts_var = S(value="50"), S(value="801")
        self.batch_eye_var = S(value="Eye: off")
        self.batch_status_var = S(value="Pick files or a folder, then Run.")
        self._batch_rows: list[list[str]] = []
        F, B = self._adv_field, self._adv_button

        def cond_c(bar: tk.Widget) -> None:
            F(bar, "New reference Z0 (Ω)", lambda p: ttk.Entry(p, textvariable=self.z0_new_var, width=7))
            B(bar, "Renormalize", self._cond_renorm, accent=False)
            F(bar, "Uniform points", lambda p: ttk.Entry(p, textvariable=self.npts_var, width=7))
            B(bar, "Resample", self._cond_resample, accent=False)
            B(bar, "Add DC point", self._cond_dc, accent=False)
            B(bar, "Enforce reciprocity", self._cond_recip, accent=False)
            B(bar, "Undo", self._undo, accent=False)

        self._adv_tab("Conditioning", "cond", cond_c, self.t_nb)
        self._placeholder("cond", "Load a Touchstone file first.")

        tab = ttk.Frame(self.t_nb)
        self.t_nb.add(tab, text="Batch")
        bar = ttk.Frame(tab, style="Card.TFrame", padding=(12, 8))
        bar.pack(fill="x", pady=(0, 6))
        B(bar, "Add files…", self._batch_files, accent=False)
        B(bar, "Add folder…", self._batch_folder, accent=False)
        B(bar, "Clear", self._batch_clear, accent=False)
        F(bar, "Eye metrics", lambda p: ttk.Combobox(p, textvariable=self.batch_eye_var, values=["Eye: off", "Eye: on (16k symbols)"], width=18, state="readonly"))
        B(bar, "Run", self._batch_run)
        B(bar, "Export CSV", self._batch_csv, accent=False)
        ttk.Label(bar, textvariable=self.batch_status_var, style="CardMuted.TLabel", wraplength=300).pack(side="left")
        style = ttk.Style(self)
        style.configure("Treeview", background=CARD, fieldbackground=CARD, foreground=FG, borderwidth=0, rowheight=24)
        style.configure("Treeview.Heading", background=CARD_HI, foreground=FG, relief="flat")
        style.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", "#04111c")])
        holder = ttk.Frame(tab)
        holder.pack(fill="both", expand=True)
        self.batch_tree = ttk.Treeview(holder, columns=BATCH_COLUMNS, show="headings")
        for c in BATCH_COLUMNS:
            self.batch_tree.heading(c, text=c)
            self.batch_tree.column(c, width=170 if c == "File" else 84, anchor="w" if c == "File" else "e")
        sb = ttk.Scrollbar(holder, command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=sb.set)
        self.batch_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._batch_files_list: list[Path] = []

    # -- conditioning -------------------------------------------------------------------------------
    def _apply_arrays(self, f: np.ndarray, s: np.ndarray, z0: np.ndarray, label: str) -> None:
        d = self.channel
        self._push_history()
        if len(z0) != len(f):
            z0 = np.repeat(z0[:1], len(f), axis=0)
        self.channel = ChannelData(Path(f"{d.path.stem}-{label}.s2p"), np.asarray(f, dtype=float),
                                   np.asarray(s, dtype=complex), z0)
        self.eye_result = None
        self._refresh_analysis()
        self._set_status(f"Conditioning applied: {label}")

    def _cond_renorm(self) -> None:
        if self.channel is None:
            return
        try:
            net = self._channel_network()
            net.renormalize(max(self._fv(self.z0_new_var, 50.0), 1e-3))
            z0 = np.asarray(net.z0, dtype=complex)
            if z0.ndim == 1:
                z0 = np.repeat(z0[:, None], 2, axis=1)
            self._apply_arrays(net.f, net.s, z0, f"z{self._fv(self.z0_new_var, 50.0):g}")
        except Exception as exc:
            messagebox.showerror("Renormalization failed", str(exc))

    def _cond_resample(self) -> None:
        if self.channel is None:
            return
        n = int(max(self._fv(self.npts_var, 801.0), 8))
        fn, sn = resample_uniform(self.channel.frequency_hz, self.channel.s, n)
        self._apply_arrays(fn, sn, self.channel.z0, f"uniform{n}")

    def _cond_dc(self) -> None:
        if self.channel is None:
            return
        if self.channel.frequency_hz[0] <= 0:
            return messagebox.showinfo("Already has DC", "The first frequency point is already 0 Hz.")
        fn, sn = extrapolate_dc(self.channel.frequency_hz, self.channel.s)
        self._apply_arrays(fn, sn, self.channel.z0, "dc")

    def _cond_recip(self) -> None:
        if self.channel is None:
            return
        s = 0.5 * (self.channel.s + self.channel.s.transpose(0, 2, 1))
        self._apply_arrays(self.channel.frequency_hz, s, self.channel.z0, "recip")

    def _draw_cond(self) -> None:
        if self.channel is None:
            return self._placeholder("cond", "Load a Touchstone file first.")
        d, f = self.channel, self.channel.frequency_hz
        g = f / 1e9
        fig = self.fig_cond
        fig.clf()
        ax1, ax2, ax3 = fig.subplots(1, 3)
        ax1.plot(g, _db(d.s[:, 1, 0]), ".-", color=ACCENT2, markersize=3)
        ax1.set_title("|S21| (dB)")
        ax2.plot(g, _db(d.s[:, 0, 0]), ".-", color=ACCENT, markersize=3)
        ax2.set_title("|S11| (dB)")
        ax3.plot(g[1:], np.diff(f) / 1e6, ".-", color=GREEN, markersize=3)
        ax3.set_title("Frequency step (MHz)")
        for ax in (ax1, ax2, ax3):
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True)
        step = np.diff(f)
        uniform = bool(np.allclose(step, step[0], rtol=1e-3))
        fig.suptitle(f"{len(f)} points, {g[0]:.4g}–{g[-1]:.4g} GHz, {'uniform' if uniform else 'non-uniform'} grid, "
                     f"{'includes DC' if f[0] <= 0 else 'no DC point'}, Z0 = {np.median(np.real(d.z0[:, 0])):g} Ω",
                     color=MUTED, fontsize=9)
        self._finish("cond")

    # -- impedance discontinuity text -----------------------------------------------------------------------
    @staticmethod
    def _disc_line(label: str, t: np.ndarray, z: np.ndarray, z0: float, span: float, v: float) -> str:
        found = find_discontinuities(t, z, z0, span)
        if not found:
            return f"{label}: no excursion larger than {max(2.0, 0.03 * z0):.1f} Ω from {z0:g} Ω"
        items = "; ".join(f"{tt*1e9:.3g} ns ({tt*v/2*1e3:.3g} mm): {zz:.1f} Ω" for tt, zz in found)
        return f"{label}: {items}"

    # -- batch ----------------------------------------------------------------------------------------------------
    def _batch_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Select Touchstone files",
                                            filetypes=[("Touchstone", "*.s2p *.s4p *.S2P *.S4P"), ("All files", "*.*")])
        self._batch_files_list += [Path(p) for p in paths]
        self.batch_status_var.set(f"{len(self._batch_files_list)} file(s) queued")

    def _batch_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose a folder with Touchstone files")
        if folder:
            found = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".s2p", ".s4p"))
            self._batch_files_list += found
            self.batch_status_var.set(f"{len(self._batch_files_list)} file(s) queued")

    def _batch_clear(self) -> None:
        self._batch_files_list.clear()
        self._batch_rows.clear()
        self.batch_tree.delete(*self.batch_tree.get_children())
        self.batch_status_var.set("Cleared.")

    def _batch_run(self) -> None:
        if not self._batch_files_list:
            return messagebox.showinfo("Nothing queued", "Add files or a folder first.")
        self.batch_tree.delete(*self.batch_tree.get_children())
        self._batch_rows.clear()
        nyq = self._nyquist_hz() or 0.0
        eq = self._eq_settings()
        with_eye = self.batch_eye_var.get().startswith("Eye: on")
        tr = max(self._fv(self.tr_var, 35.0), 0.0)
        for k, path in enumerate(self._batch_files_list, start=1):
            self.batch_status_var.set(f"Processing {k}/{len(self._batch_files_list)}: {path.name}")
            self.update_idletasks()
            try:
                row = batch_row(path, nyq, self.pair_var.get(), self._fv(self.rate_var, 5.0), self._fv(self.swing_var, 1.0),
                                self.signal_var.get(), eq, with_eye, tr, self.win_var.get())
            except Exception as exc:
                note = "needs port mapping (File ▸ Open with port mapping)" if isinstance(exc, NeedPortMap) else "error: " + str(exc)[:60]
                row = [path.name, note] + [""] * (len(BATCH_COLUMNS) - 2)
            self._batch_rows.append(row)
            self.batch_tree.insert("", "end", values=row)
        self.batch_status_var.set(f"Done: {len(self._batch_rows)} file(s).")

    def _batch_csv(self) -> None:
        if not self._batch_rows:
            return messagebox.showinfo("No results", "Run the batch first.")
        import csv
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(BATCH_COLUMNS)
                w.writerows(self._batch_rows)
            self.batch_status_var.set(f"Wrote {Path(path).name}")


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
BIT_CHOICES = ["16,384", "65,536", "262,144", "1,048,576", "4,194,304"]
CMAP_CHOICES = ["inferno", "magma", "viridis", "turbo", "plasma", "cividis"]


class SIAnalyzerApp(UltimateTabs, ExtraTabs, AdvancedTabs, tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x880")
        self.minsize(1000, 700)
        self.configure(bg=BG)
        self.channel: ChannelData | None = None
        self.eye_result: dict[str, Any] | None = None
        self._eye_state: dict[str, Any] | None = None
        self._smith_cbar: Any = None
        self._build_ui()
        self._set_status("File ▸ Open (Ctrl+O) to load a Touchstone file. Double-click a plot to drop a marker; right double-click clears.")

    # -- styling -----------------------------------------------------------
    def _setup_style(self) -> None:
        families = set(tkfont.families())
        self.ui_font = next((f for f in ("Inter", "Segoe UI", "SF Pro Text", "Ubuntu", "Cantarell", "DejaVu Sans") if f in families), "TkDefaultFont")
        self.mono_font = next((f for f in ("JetBrains Mono", "Cascadia Mono", "Consolas", "Ubuntu Mono", "DejaVu Sans Mono") if f in families), "TkFixedFont")
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            try:
                tkfont.nametofont(name).configure(family=self.ui_font, size=10)
            except tk.TclError:
                pass

        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", background=BG, foreground=FG, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                    troughcolor=PANEL, focuscolor=ACCENT, font=(self.ui_font, 10))
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=CARD, relief="flat")
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Card.TLabel", background=CARD, foreground=FG)
        s.configure("CardMuted.TLabel", background=CARD, foreground=MUTED, font=(self.ui_font, 9))
        s.configure("Muted.TLabel", background=BG, foreground=MUTED)
        s.configure("Title.TLabel", background=BG, foreground=FG, font=(self.ui_font, 19, "bold"))
        s.configure("Sub.TLabel", background=BG, foreground=MUTED, font=(self.ui_font, 10))
        s.configure("KpiTitle.TLabel", background=CARD, foreground=MUTED, font=(self.ui_font, 9))
        s.configure("KpiValue.TLabel", background=CARD, foreground=ACCENT, font=(self.ui_font, 17, "bold"))

        s.configure("TButton", background=CARD_HI, foreground=FG, borderwidth=0, padding=(14, 7), relief="flat")
        s.map("TButton", background=[("active", "#27345a"), ("disabled", PANEL)], foreground=[("disabled", MUTED)])
        s.configure("Accent.TButton", background=ACCENT, foreground="#04111c", font=(self.ui_font, 10, "bold"))
        s.map("Accent.TButton", background=[("active", "#7bd3ff"), ("disabled", "#26516a")], foreground=[("disabled", "#0a1a26")])

        s.configure("TEntry", fieldbackground=PANEL, foreground=FG, insertcolor=FG, bordercolor=BORDER, padding=4)
        s.map("TEntry", bordercolor=[("focus", ACCENT)])
        s.configure("TCombobox", fieldbackground=PANEL, background=CARD_HI, foreground=FG, arrowcolor=FG,
                    bordercolor=BORDER, padding=4)
        s.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)],
              bordercolor=[("focus", ACCENT)], selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", FG)])
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", "#04111c")

        s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(18, 9), borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", ACCENT)])

        s.configure("TScrollbar", background=CARD_HI, troughcolor=PANEL, bordercolor=PANEL, arrowcolor=MUTED)
        s.configure("Eye.Horizontal.TProgressbar", troughcolor=PANEL, background=ACCENT, bordercolor=PANEL,
                    lightcolor=ACCENT, darkcolor=ACCENT, thickness=6)
        s.configure("Status.TLabel", background=PANEL, foreground=MUTED, padding=(14, 6))

    def _build_ui(self) -> None:
        self._setup_style()
        self._build_menu()

        header = ttk.Frame(self, padding=(20, 16, 20, 8))
        header.pack(fill="x")
        titles = ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles, text="Signal Integrity Analyzer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(titles, text="Touchstone S-parameters · loss · impedance · statistical eye · TDR · mixed-mode · noise", style="Sub.TLabel").pack(anchor="w")
        ttk.Button(header, text="Load .s2p", style="Accent.TButton", command=self._choose_s2p).pack(side="right", padx=(10, 0))

        # Eye controls card
        controls = ttk.Frame(self, style="Card.TFrame", padding=(16, 12))
        controls.pack(fill="x", padx=20, pady=(4, 10))
        self.rate_var = tk.StringVar(value="5.0")
        self.swing_var = tk.StringVar(value="1.0")
        self.bits_var = tk.StringVar(value="262,144")
        self.cmap_var = tk.StringVar(value="inferno")
        self.signal_var = tk.StringVar(value=DEFAULT_SIGNAL)

        def field(label: str, factory: Callable[[tk.Widget], tk.Widget]) -> tk.Widget:
            # The widget is created INSIDE the box so the label and the widget
            # stack vertically (label above, input below) instead of sitting side by side.
            box = ttk.Frame(controls, style="Card.TFrame")
            box.pack(side="left", padx=(0, 16))
            ttk.Label(box, text=label, style="CardMuted.TLabel").pack(anchor="w")
            widget = factory(box)
            widget.pack(anchor="w", pady=(2, 0))
            return widget

        signal_box = field("Signal type", lambda p: ttk.Combobox(
            p, textvariable=self.signal_var, values=SIGNAL_CHOICES, width=15, state="readonly"))
        signal_box.bind("<<ComboboxSelected>>", lambda _e: self._on_signal_change())
        field("Data rate (Gb/s)", lambda p: ttk.Entry(p, textvariable=self.rate_var, width=9))
        field("Swing (Vpp, diff-equivalent)", lambda p: ttk.Entry(p, textvariable=self.swing_var, width=9))
        field("Symbols observed", lambda p: ttk.Combobox(p, textvariable=self.bits_var, values=BIT_CHOICES, width=12))
        cmap_box = field("Colormap", lambda p: ttk.Combobox(
            p, textvariable=self.cmap_var, values=CMAP_CHOICES, width=10, state="readonly"))
        cmap_box.bind("<<ComboboxSelected>>", lambda _e: self._redraw_eye())

        action = ttk.Frame(controls, style="Card.TFrame")
        action.pack(side="left", fill="y")
        ttk.Label(action, text=" ", style="CardMuted.TLabel").pack(anchor="w")  # aligns the button with the inputs
        self.eye_btn = ttk.Button(action, text="Recalculate eye", style="Accent.TButton", command=self._recalculate_eye)
        self.eye_btn.pack(anchor="w", pady=(2, 0))

        right = ttk.Frame(controls, style="Card.TFrame")
        right.pack(side="right", fill="both", expand=True, padx=(16, 0))
        ttk.Label(right, text="Random pattern through measured S21 · TX edge, EQ, terminations and noise as set below", style="CardMuted.TLabel").pack(anchor="e")
        self.progress = ttk.Progressbar(right, style="Eye.Horizontal.TProgressbar", mode="determinate", maximum=1.0)
        self.progress.pack(fill="x", pady=(14, 0))

        # Equalisation, mask and noise card
        eqc = ttk.Frame(self, style="Card.TFrame", padding=(16, 8))
        eqc.pack(fill="x", padx=20, pady=(0, 10))
        self.ffe_pre_var, self.ffe_post_var = tk.StringVar(value="0"), tk.StringVar(value="0")
        self.ctle_db_var, self.ctle_ghz_var = tk.StringVar(value="0"), tk.StringVar(value="0")
        self.dfe_var = tk.StringVar(value="0")
        self.mask_var = tk.StringVar(value="Off")
        self.mask_h_var, self.mask_w_var = tk.StringVar(value="50"), tk.StringVar(value="0.3")
        self.noise_var, self.rj_var = tk.StringVar(value="1.0"), tk.StringVar(value="0.0")

        def efield(label: str, factory: Callable[[tk.Widget], tk.Widget]) -> tk.Widget:
            box = ttk.Frame(eqc, style="Card.TFrame")
            box.pack(side="left", padx=(0, 12))
            ttk.Label(box, text=label, style="CardMuted.TLabel").pack(anchor="w")
            w = factory(box)
            w.pack(anchor="w", pady=(2, 0))
            return w

        efield("TX FFE pre (%)", lambda p: ttk.Entry(p, textvariable=self.ffe_pre_var, width=6))
        efield("TX FFE post (%)", lambda p: ttk.Entry(p, textvariable=self.ffe_post_var, width=6))
        efield("CTLE boost (dB)", lambda p: ttk.Entry(p, textvariable=self.ctle_db_var, width=6))
        efield("CTLE peak (GHz, 0 = Nyq.)", lambda p: ttk.Entry(p, textvariable=self.ctle_ghz_var, width=8))
        efield("DFE taps", lambda p: ttk.Entry(p, textvariable=self.dfe_var, width=5))
        mask_box = efield("Eye mask", lambda p: ttk.Combobox(p, textvariable=self.mask_var, values=MASK_PRESETS, width=15, state="readonly"))
        mask_box.bind("<<ComboboxSelected>>", lambda _e: self._on_mask_preset())
        efield("Mask height (mV)", lambda p: ttk.Entry(p, textvariable=self.mask_h_var, width=6))
        efield("Mask width (UI)", lambda p: ttk.Entry(p, textvariable=self.mask_w_var, width=6))
        eqc2 = ttk.Frame(self, style="Card.TFrame", padding=(16, 8))
        eqc2.pack(fill="x", padx=20, pady=(0, 10))
        self.tx_rise_var = tk.StringVar(value="auto")
        self.quality_var = tk.StringVar(value="High (128 samples/UI)")
        self.eye_view_var = tk.StringVar(value="Deterministic")
        self.term_var = tk.StringVar(value="Ideal (matched)")

        def efield2(label: str, factory: Callable[[tk.Widget], tk.Widget]) -> tk.Widget:
            box = ttk.Frame(eqc2, style="Card.TFrame")
            box.pack(side="left", padx=(0, 12))
            ttk.Label(box, text=label, style="CardMuted.TLabel").pack(anchor="w")
            w = factory(box)
            w.pack(anchor="w", pady=(2, 0))
            return w

        efield2("TX rise 10–90 % (ps)", lambda p: ttk.Entry(p, textvariable=self.tx_rise_var, width=7))
        efield2("Terminations", lambda p: ttk.Combobox(p, textvariable=self.term_var, values=["Ideal (matched)", "Zs/ZL from Gains tab"], width=19, state="readonly"))
        efield2("Noise σ (mV)", lambda p: ttk.Entry(p, textvariable=self.noise_var, width=6))
        efield2("Random jitter σ (ps)", lambda p: ttk.Entry(p, textvariable=self.rj_var, width=6))
        efield2("Eye quality", lambda p: ttk.Combobox(p, textvariable=self.quality_var, values=["Normal (64 samples/UI)", "High (128 samples/UI)", "Ultra (256 samples/UI)"], width=20, state="readonly"))
        view_box = efield2("Eye view", lambda p: ttk.Combobox(p, textvariable=self.eye_view_var, values=["Deterministic", "With noise + RJ"], width=15, state="readonly"))
        view_box.bind("<<ComboboxSelected>>", lambda _e: self._redraw_eye())
        ttk.Label(eqc2, text="TX rise: 'auto' = 25 % UI, 0 = ideal edge.", style="CardMuted.TLabel").pack(side="left")

        # KPI strip
        kpi_row = ttk.Frame(self)
        kpi_row.pack(fill="x", padx=20, pady=(0, 10))
        self.kpis: dict[str, ttk.Label] = {}
        for i, (key, title) in enumerate((
            ("nyq", "IL @ Nyquist"), ("il_max", "IL @ f max"), ("z11", "Median Re Z11"),
            ("height", "Eye height (0.1 %)"), ("width", "Eye width"), ("jitter", "RMS crossing spread"),
        )):
            card = ttk.Frame(kpi_row, style="Card.TFrame", padding=(14, 10))
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            kpi_row.columnconfigure(i, weight=1, uniform="kpi")
            ttk.Label(card, text=title, style="KpiTitle.TLabel").pack(anchor="w")
            lbl = ttk.Label(card, text="—", style="KpiValue.TLabel")
            lbl.pack(anchor="w")
            self.kpis[key] = lbl

        # Status bar first so it stays at the bottom
        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var, style="Status.TLabel", anchor="w").pack(fill="x", side="bottom")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        self.summary_tab = ttk.Frame(self.notebook)
        self.sparam_tab = ttk.Frame(self.notebook)
        self.loss_tab = ttk.Frame(self.notebook)
        self.impedance_tab = ttk.Frame(self.notebook)
        self.eye_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_tab, text="Project & summary")
        self.notebook.add(self.sparam_tab, text="S-parameters")
        self.notebook.add(self.loss_tab, text="Loss")
        self.notebook.add(self.impedance_tab, text="Impedance & Smith")
        self.notebook.add(self.eye_tab, text="Eye & jitter")

        self.summary_text = tk.Text(
            self.summary_tab, wrap="word", font=(self.mono_font, 10), padx=18, pady=14, bg=CARD, fg=FG,
            insertbackground=FG, relief="flat", borderwidth=0, highlightthickness=0, spacing1=2, spacing3=2,
        )
        self.summary_text.tag_configure("h", foreground=ACCENT, font=(self.mono_font, 11, "bold"), spacing1=10)
        summary_scroll = ttk.Scrollbar(self.summary_tab, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        self.summary_text.pack(side="left", fill="both", expand=True)
        summary_scroll.pack(side="right", fill="y")
        self._set_summary("## Getting started\nSelect a Touchstone .s2p file to begin.")

        self.fig_s = Figure(figsize=(9, 6), dpi=100, constrained_layout=True)
        self.ax_s_mag, self.ax_s_phase = self.fig_s.subplots(2, 1, sharex=True)
        self.canvas_s = self._make_plot(self.sparam_tab, self.fig_s)

        self.fig_loss = Figure(figsize=(9, 6), dpi=100, constrained_layout=True)
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.canvas_loss = self._make_plot(self.loss_tab, self.fig_loss)

        self.fig_z = Figure(figsize=(9, 6), dpi=100, constrained_layout=True)
        self.ax_z, self.ax_smith = self.fig_z.subplots(1, 2)
        self.canvas_z = self._make_plot(self.impedance_tab, self.fig_z)

        self.fig_eye = Figure(figsize=(9, 6), dpi=100, constrained_layout=True)
        self.ax_eye = self.fig_eye.add_subplot(111)
        self.canvas_eye = self._make_plot(self.eye_tab, self.fig_eye)
        self._build_advanced_tabs()
        self._build_extra_tabs()
        self._build_ultimate_tabs()

    @staticmethod
    def _make_plot(parent: ttk.Frame, figure: Figure) -> FigureCanvasTkAgg:
        holder = tk.Frame(parent, bg=PLOT_BG)
        holder.pack(fill="both", expand=True)
        canvas = FigureCanvasTkAgg(figure, master=holder)
        canvas.get_tk_widget().configure(bg=PLOT_BG, highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, holder, pack_toolbar=False)
        # Re-theme toolbar widgets and recolor their icons for the dark background.
        toolbar.configure(background=PANEL)
        for child in toolbar.winfo_children():
            try:
                child.configure(background=PANEL, foreground=FG, activebackground=CARD_HI, highlightbackground=PANEL,
                                selectcolor=CARD_HI)
            except tk.TclError:
                pass
            try:
                toolbar._set_image_for_button(child)  # matplotlib private API; harmless if missing
            except Exception:
                pass
        try:
            toolbar._message_label.configure(background=PANEL, foreground=MUTED)
        except Exception:
            pass
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        def on_click(event: Any) -> None:
            ax = event.inaxes
            if ax is None or not getattr(event, "dblclick", False) or getattr(toolbar, "mode", ""):
                return
            if event.button == 3:  # right double-click clears the markers of this axes
                for art in getattr(ax, "_si_markers", []):
                    try:
                        art.remove()
                    except Exception:
                        pass
                ax._si_markers = []
                canvas.draw_idle()
                return
            if event.button != 1:
                return
            best = None
            for line in ax.get_lines():
                if str(line.get_label()).startswith("_si"):
                    continue
                try:
                    xs, ys = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
                    if len(xs) < 2:
                        continue
                    xy = ax.transData.transform(np.column_stack([xs, ys]))
                    dist = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
                    i = int(np.nanargmin(dist))
                except Exception:
                    continue
                if best is None or dist[i] < best[0]:
                    best = (float(dist[i]), xs[i], ys[i])
            if best is None:
                return
            _, x0, y0 = best
            mk = ax.plot([x0], [y0], "o", color="white", markersize=6, zorder=10, label="_si_marker", scalex=False, scaley=False)[0]
            ann = ax.annotate(f"{x0:.4g}, {y0:.4g}", (x0, y0), xytext=(8, 8), textcoords="offset points", color=FG, fontsize=8,
                              zorder=11, bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD, edgecolor=BORDER, alpha=0.95))
            ax._si_markers = list(getattr(ax, "_si_markers", [])) + [mk, ann]
            canvas.draw_idle()

        canvas.mpl_connect("button_press_event", on_click)
        return canvas

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _on_signal_change(self) -> None:
        # The Nyquist frequency depends on the symbol rate, which depends on the signal type.
        self._draw_loss()
        self._update_kpis()

    # -- loading -----------------------------------------------------------
    def _choose_s2p(self, force_map: bool = False) -> None:
        path = filedialog.askopenfilename(
            title="Select a Touchstone file", initialdir=self.prefs.get("last_dir") or None,
            filetypes=[("Touchstone (.s2p, .s4p, .s8p, …)", "*.s?p *.S?P *.s??p *.S??p"), ("All files", "*.*")])
        if path:
            self.load_s2p(path, force_map=force_map)

    def load_s2p(self, path: str | Path, port_map: tuple | None = None, force_map: bool = False) -> None:
        try:
            if force_map and port_map is None:
                m = re.search(r"\.s(\d+)p$", str(path).lower())
                port_map = self._ask_port_map(int(m.group(1)) if m else 4, path)
                if port_map is None:
                    return self._set_status("Load cancelled.")
            try:
                channel = read_touchstone(path, self.pair_var.get(), port_map)
            except NeedPortMap as need:
                port_map = self._ask_port_map(need.nports, path)
                if port_map is None:
                    return self._set_status("Load cancelled.")
                channel = read_touchstone(path, self.pair_var.get(), port_map)
            self.channel = channel
            self._history.clear()
            self._remember(path)
            self.eye_result = None
            self._refresh_analysis()
            self._set_status(f"Loaded {self.channel.path.name}: {len(self.channel.frequency_hz)} points, "
                             f"{self.channel.frequency_hz[0]/1e9:.4g}–{self.channel.frequency_hz[-1]/1e9:.4g} GHz.")
        except Exception as exc:
            messagebox.showerror("Could not load Touchstone file", str(exc))
            self._set_status("Load failed.")

    # -- summary -----------------------------------------------------------
    def _channel_summary(self) -> str:
        if self.channel is None:
            return "## Channel\nNo Touchstone file loaded."
        s11 = self.channel.s[:, 0, 0]
        s21 = self.channel.s[:, 1, 0]
        il = -20.0 * np.log10(np.maximum(np.abs(s21), 1e-15))
        z11 = impedance_from_reflection(s11, np.real(self.channel.z0[:, 0]))
        return (
            "## Channel\n"
            f"Touchstone: {self.channel.path}\n"
            f"Ports: {self.channel.ports}" + (f" ({self.channel.note})" if self.channel.note else (f" (differential Sdd, pairs {self.channel.pairing}, is the active 2-port)" if self.channel.ports == 4 else "")) + "\n"
            f"Frequency points: {len(self.channel.frequency_hz)}\n"
            f"Frequency range: {self.channel.frequency_hz[0]/1e9:.6g} to {self.channel.frequency_hz[-1]/1e9:.6g} GHz\n"
            f"S21 insertion loss at highest frequency: {il[-1]:.3f} dB\n"
            f"Median real(Z11): {np.nanmedian(np.real(z11)):.3f} ohm\n"
            "The selected .s2p file supplies all channel metrics. Geometry and stackup files provide design context only; no EM solve is run here."
        )

    def _eye_summary(self) -> str:
        r = self.eye_result
        if r is None:
            return ""
        pe = (f"Per-eye heights (mV): {', '.join(f'{x*1e3:.1f}' for x in r['eye_heights_v'])}\n"
              if len(r["eye_heights_v"]) > 1 else "")
        return (
            "## Eye / jitter estimate (assumptions apply)\n"
            f"Signal: {r['signal']} ({r['n_eyes']} decision threshold(s))\n"
            f"Equalization: {r['eq_text']}\n"
            f"Data rate: {r['bit_rate_gbps']:g} Gb/s; symbol rate: {r['baud_gbd']:.4g} GBd; UI: {r['ui_ps']:.3f} ps; swing: {r['swing_vpp']:g} Vpp\n"
            f"Symbols observed: {r['bits']:,} (independent random blocks, {r['oversample']} samples/UI)\n"
            f"Best sample phase: {r['sample_phase_ui']:.3f} UI\n"
            f"Statistical eye height (0.1 % tails, smallest sub-eye): {r['eye_height_raw_v']*1e3:.2f} mV\n"
            f"{pe}"
            f"Eye width (UI − crossing spread, 0.1–99.9 %): {r['eye_width_ps']:.3f} ps\n"
            f"Crossing spread: RMS {r['jitter_rms_ps']:.3f} ps; 0.1–99.9 % {r['jitter_pp_ps']:.3f} ps; {r['crossing_count']:,} crossings\n"
            "This is deterministic pattern-dependent spread. It excludes noise, TX/RX behavior, clock recovery, equalization, and BER extrapolation; it is not a standards-compliant jitter result."
        )

    def _update_summary(self) -> None:
        parts = [self._channel_summary()]
        eye = self._eye_summary()
        if eye:
            parts.append(eye)
        adv = self._advanced_summary()
        if adv:
            parts.append(adv)
        self._set_summary("\n\n".join(parts))

    def _set_summary(self, text: str) -> None:
        t = self.summary_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        for line in text.split("\n"):
            if line.startswith("## "):
                t.insert("end", line[3:] + "\n", "h")
            else:
                t.insert("end", line + "\n")
        t.configure(state="disabled")

    def _update_kpis(self) -> None:
        if self.channel is None:
            return
        d = self.channel
        il = -20.0 * np.log10(np.maximum(np.abs(d.s[:, 1, 0]), 1e-15))
        z11 = impedance_from_reflection(d.s[:, 0, 0], np.real(d.z0[:, 0]))
        self.kpis["il_max"].configure(text=f"{il[-1]:.2f} dB")
        self.kpis["z11"].configure(text=f"{np.nanmedian(np.real(z11)):.1f} Ω")
        nyq = self._nyquist_hz()
        if nyq is not None and d.frequency_hz[0] <= nyq <= d.frequency_hz[-1]:
            self.kpis["nyq"].configure(text=f"{np.interp(nyq, d.frequency_hz, il):.2f} dB")
        else:
            self.kpis["nyq"].configure(text="out of range")
        r = self.eye_result
        if r is not None:
            self.kpis["height"].configure(text=f"{r['eye_height_raw_v']*1e3:.1f} mV")
            self.kpis["width"].configure(text=f"{r['eye_width_ps']:.1f} ps")
            self.kpis["jitter"].configure(text=f"{r['jitter_rms_ps']:.2f} ps")
        else:
            for k in ("height", "width", "jitter"):
                self.kpis[k].configure(text="—")

    def _nyquist_hz(self) -> float | None:
        """Nyquist = symbol rate / 2 (symbol rate = data rate / bits-per-symbol of the chosen signal)."""
        try:
            rate = float(self.rate_var.get())
            bps = SIGNALS.get(self.signal_var.get(), SIGNALS[DEFAULT_SIGNAL])["bps"]
            return (rate / bps) * 1e9 / 2.0 if rate > 0 else None
        except ValueError:
            return None

    # -- plots -------------------------------------------------------------
    def _refresh_analysis(self) -> None:
        if self.channel is None:
            return
        d = self.channel
        f_ghz = d.frequency_hz / 1e9
        colors = {"S11": ACCENT, "S21": ACCENT2, "S12": GREEN, "S22": PURPLE}
        self.ax_s_mag.clear()
        self.ax_s_phase.clear()
        for row, col, name in ((0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")):
            values = d.s[:, row, col]
            ls = "--" if name in ("S12", "S22") else "-"
            self.ax_s_mag.plot(f_ghz, 20 * np.log10(np.maximum(np.abs(values), 1e-15)), ls, color=colors[name], label=name)
            self.ax_s_phase.plot(f_ghz, np.unwrap(np.angle(values)) * 180 / np.pi, ls, color=colors[name], label=name)
        self.ax_s_mag.set_ylabel("Magnitude (dB)")
        self.ax_s_phase.set_ylabel("Unwrapped phase (°)")
        self.ax_s_phase.set_xlabel("Frequency (GHz)")
        self.ax_s_mag.set_title("Two-port S-parameters")
        for ax in (self.ax_s_mag, self.ax_s_phase):
            ax.grid(True)
            ax.legend(ncol=4, loc="best")
        self.canvas_s.draw_idle()

        self._draw_loss()

        self.ax_z.clear()
        z11 = impedance_from_reflection(d.s[:, 0, 0], np.real(d.z0[:, 0]))
        z22 = impedance_from_reflection(d.s[:, 1, 1], np.real(d.z0[:, 1]))
        self.ax_z.plot(f_ghz, np.real(z11), color=ACCENT, label="Re Z11")
        self.ax_z.plot(f_ghz, np.real(z22), color=PURPLE, label="Re Z22")
        self.ax_z.plot(f_ghz, np.abs(z11), "--", color=ACCENT2, alpha=0.85, label="|Z11|")
        self.ax_z.set_title("Input impedance from S11 / S22")
        self.ax_z.set_xlabel("Frequency (GHz)")
        self.ax_z.set_ylabel("Impedance (Ω)")
        self.ax_z.grid(True)
        self.ax_z.legend()
        self._draw_smith(self.ax_smith, d.s[:, 0, 0], f_ghz)
        self.canvas_z.draw_idle()

        self._refresh_advanced()
        self._refresh_extra()
        self._update_kpis()
        self._update_summary()
        self._recalculate_eye(silent=True)

    def _draw_loss(self) -> None:
        if self.channel is None:
            return
        d = self.channel
        f_ghz = d.frequency_hz / 1e9
        il = -20 * np.log10(np.maximum(np.abs(d.s[:, 1, 0]), 1e-15))
        self.ax_loss.clear()
        self.ax_loss.fill_between(f_ghz, 0, il, color=ACCENT, alpha=0.12)
        self.ax_loss.plot(f_ghz, il, color=ACCENT, label="Insertion loss = −20 log10 |S21|")
        nyq = self._nyquist_hz()
        if nyq is not None and d.frequency_hz[0] <= nyq <= d.frequency_hz[-1]:
            self.ax_loss.axvline(nyq / 1e9, color=ACCENT2, linestyle="--", linewidth=1.2, label=f"Nyquist ({nyq/1e9:.3g} GHz)")
        self.ax_loss.axhline(0, color=MUTED, linewidth=0.7)
        self.ax_loss.set_title("Forward insertion loss")
        self.ax_loss.set_xlabel("Frequency (GHz)")
        self.ax_loss.set_ylabel("Insertion loss (dB)")
        self.ax_loss.grid(True)
        self.ax_loss.legend()
        self.canvas_loss.draw_idle()

    def _draw_smith(self, ax: Any, gamma: np.ndarray, f_ghz: np.ndarray) -> None:
        # Drop the previous colorbar first, otherwise a new one is stacked on every reload.
        if self._smith_cbar is not None:
            try:
                self._smith_cbar.remove()
            except Exception:
                pass
            self._smith_cbar = None
        ax.clear()

        theta = np.linspace(0, 2 * np.pi, 720)
        for r in (0.2, 0.5, 1.0, 2.0, 5.0):
            c, rad = r / (1 + r), 1 / (1 + r)
            ax.plot(c + rad * np.cos(theta), rad * np.sin(theta), color=GRID, linewidth=0.9)
        for x in (0.2, 0.5, 1.0, 2.0, 5.0):
            for sgn in (1, -1):
                px = 1 + (1 / x) * np.cos(theta)
                py = sgn / x + (1 / x) * np.sin(theta)
                inside = px ** 2 + py ** 2 <= 1.0005
                ax.plot(np.where(inside, px, np.nan), np.where(inside, py, np.nan), color=GRID, linewidth=0.9)
        ax.plot(np.cos(theta), np.sin(theta), color=MUTED, linewidth=1.3)
        ax.axhline(0, color=GRID, linewidth=0.9)

        # S11 trace as a gradient line: blue (lowest f) -> red (highest f).
        pts = np.column_stack([np.real(gamma), np.imag(gamma)]).reshape(-1, 1, 2)
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segments, cmap=FREQ_CMAP, norm=Normalize(float(f_ghz[0]), float(f_ghz[-1])), linewidth=2.4)
        lc.set_array(0.5 * (f_ghz[:-1] + f_ghz[1:]))
        ax.add_collection(lc)

        self._smith_cbar = ax.figure.colorbar(lc, ax=ax, fraction=0.046, pad=0.03)
        self._smith_cbar.set_label("Frequency (GHz)")
        self._smith_cbar.outline.set_edgecolor(BORDER)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("S11 on the Smith chart")
        ax.set_xlabel("Real(Γ)")
        ax.set_ylabel("Imag(Γ)")

    # -- eye ---------------------------------------------------------------
    def _recalculate_eye(self, silent: bool = False) -> None:
        if self.channel is None:
            if not silent:
                messagebox.showinfo("Load channel first", "Load a .s2p file before calculating an eye diagram.")
            return
        if self._eye_state is not None and not self._eye_state["done"]:
            return  # a run is already in progress
        try:
            rate = float(self.rate_var.get())
            swing = float(self.swing_var.get())
            bits = int(self.bits_var.get().replace(",", "").replace("_", "").strip())
        except ValueError:
            if not silent:
                messagebox.showerror("Invalid input", "Data rate, swing, and symbol count must be numbers.")
            return
        signal = self.signal_var.get()
        eq = self._eq_settings()

        freq = self.channel.frequency_hz
        if self.term_var.get().startswith("Zs/ZL"):
            gs, gl = self._terminations()
            s21 = terminated_response(self.channel.s, gs, gl)
        else:
            s21 = self.channel.s[:, 1, 0]
        state: dict[str, Any] = {"done": False, "progress": 0.0, "result": None, "error": None, "silent": silent}
        self._eye_state = state

        def work() -> None:
            try:
                state["result"] = make_eye_and_jitter(
                    freq, s21, rate, swing, total_bits=bits, signal=signal,
                    progress=lambda p: state.__setitem__("progress", p), **eq,
                )
            except Exception as exc:  # reported on the Tk thread
                state["error"] = exc
            state["done"] = True

        self.eye_btn.configure(state="disabled")
        self.progress.configure(value=0.0)
        self._set_status(f"Calculating {signal} eye over {bits:,} symbols…")
        threading.Thread(target=work, daemon=True).start()
        self.after(80, self._poll_eye)

    def _poll_eye(self) -> None:
        state = self._eye_state
        if state is None:
            return
        self.progress.configure(value=state["progress"])
        if not state["done"]:
            self.after(80, self._poll_eye)
            return
        self.eye_btn.configure(state="normal")
        self.progress.configure(value=1.0 if state["error"] is None else 0.0)
        if state["error"] is not None:
            if not state["silent"]:
                messagebox.showerror("Eye analysis failed", str(state["error"]))
            self._set_status(f"Eye analysis failed: {state['error']}")
            return
        self.eye_result = state["result"]
        self._redraw_eye()
        self._draw_loss()
        self._update_kpis()
        self._update_summary()
        self._after_eye()
        self._set_status(f"Eye done ({self.eye_result['signal']}): {self.eye_result['bits']:,} symbols. "
                         "Interpret jitter only under the displayed assumptions.")

    def _redraw_eye(self) -> None:
        r = self.eye_result
        if r is None:
            return
        fig = self.fig_eye
        fig.clf()
        ax = fig.add_subplot(111)
        self.ax_eye = ax

        dens = np.array(r.get("density_vis", r["density"]), dtype=float)
        os_ = r["oversample"]
        noisy = self.eye_view_var.get().startswith("With")
        if noisy:  # statistical view: convolve with the voltage-noise and random-jitter Gaussians
            dv = (r["v_max"] - r["v_min"]) / dens.shape[1]
            sv = max(self._fv(self.noise_var, 0.0), 0.0) / 1e3 / dv
            sj = max(self._fv(self.rj_var, 0.0), 0.0) / r["ui_ps"] * os_
            if sv > 0.05:
                dens = gaussian_filter1d(dens, sv, axis=1, mode="nearest")
            if sj > 0.05:
                dens = gaussian_filter1d(dens, sj, axis=0, mode="wrap")
        dens = gaussian_filter(dens, sigma=(0.9 * os_ / 64.0, 1.2), mode="nearest")
        norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=float(dens.max()) or 1.0)
        im = ax.imshow(
            dens.T, origin="lower", aspect="auto", cmap=self.cmap_var.get(), norm=norm, interpolation="bilinear",
            extent=(0.0, 2.0, r["v_min"], r["v_max"]),
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
        cbar.set_label("Hit density (relative)")
        cbar.outline.set_edgecolor(BORDER)
        cbar.set_ticks([])

        for i, t in enumerate(r["thresholds_v"]):
            ax.axhline(float(t), color=RED, linewidth=0.9, linestyle="--", alpha=0.8,
                       label="Decision threshold(s)" if i == 0 else None)
        for k in (0, 1):
            ax.axvline(r["sample_phase_ui"] + k, color="#ffffff", linewidth=0.8, linestyle=":", alpha=0.7,
                       label="Best sample phase" if k == 0 else None)
        ax.set_xlim(0, 2)
        ax.set_ylim(r["v_min"], r["v_max"])
        ax.set_title(f"{r['signal']} eye density from measured S21 — {r['bits']:,} symbols" + (" (noise + RJ applied)" if noisy else ""))
        ax.set_xlabel("Time (UI)")
        ax.set_ylabel("Received voltage (V)")
        ui_ps = r["ui_ps"]
        top = ax.secondary_xaxis("top", functions=(lambda x: x * ui_ps, lambda x: x / ui_ps))
        top.set_xlabel("Time (ps)", color=MUTED)
        top.tick_params(colors=MUTED)
        self._draw_mask_overlay(ax, r)
        ax.legend(loc="upper right")
        ax.text(
            0.015, 0.03,
            f"Height {r['eye_height_raw_v']*1e3:.1f} mV   Width {r['eye_width_ps']:.1f} ps   "
            f"RMS spread {r['jitter_rms_ps']:.2f} ps",
            transform=ax.transAxes, color=FG, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD, edgecolor=BORDER, alpha=0.9),
        )
        self.canvas_eye.draw_idle()


def main() -> None:
    import sys
    app = SIAnalyzerApp()
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        app.after(300, lambda: app.load_s2p(sys.argv[1]))  # python si_analyzer.py channel.s2p
    app.mainloop()


if __name__ == "__main__":
    main()