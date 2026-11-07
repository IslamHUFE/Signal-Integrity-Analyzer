#!/usr/bin/env python3
"""SI Analyzer for Touchstone .s2p channels (dark UI).

The S-parameter data drives the plotted channel analysis. A synthetic NRZ eye is
calculated from S21 using the user-selected data rate and swing; it is not a
replacement for a standards-compliant transmitter/receiver or BER simulator.

The eye is accumulated as a 2-D hit-density histogram (time x voltage) over
hundreds of thousands to millions of bits and drawn like a spectrogram.
"""
from __future__ import annotations

import math
import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import matplotlib as mpl
import numpy as np
import skrf as rf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

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
    path: Path
    frequency_hz: np.ndarray
    s: np.ndarray
    z0: np.ndarray


def read_touchstone(path: str | Path) -> ChannelData:
    """Load a Touchstone file through scikit-rf and validate it is a 2-port network."""
    file_path = Path(path)
    network = rf.Network(str(file_path))
    if network.nports != 2:
        raise ValueError(f"Expected a 2-port Touchstone file; found {network.nports} port(s).")
    f = np.asarray(network.f, dtype=float)
    s = np.asarray(network.s, dtype=complex)
    z0 = np.asarray(network.z0, dtype=complex)
    if len(f) < 3 or not np.all(np.isfinite(f)) or np.any(np.diff(f) <= 0):
        raise ValueError("The Touchstone file must contain at least three increasing frequency points.")
    if not np.all(np.isfinite(s)):
        raise ValueError("The S-parameter file contains non-finite values.")
    if z0.ndim == 1:
        z0 = np.repeat(z0[:, None], 2, axis=1)
    return ChannelData(file_path, f, s, z0)


def impedance_from_reflection(gamma: np.ndarray, reference_z: np.ndarray) -> np.ndarray:
    """Convert a one-port reflection coefficient to impedance."""
    denominator = 1.0 - gamma
    with np.errstate(divide="ignore", invalid="ignore"):
        z = reference_z * (1.0 + gamma) / denominator
    z[np.abs(denominator) < 1e-12] = np.nan + 1j * np.nan
    return z


# ----------------------------------------------------------------------------
# Eye / jitter engine (block-wise, constant memory, any number of bits)
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


def make_eye_and_jitter(
    frequency_hz: np.ndarray,
    s21: np.ndarray,
    bit_rate_gbps: float,
    swing_vpp: float,
    total_bits: int = 262_144,
    oversample: int = 64,
    block_bits: int = 16_384,
    v_bins: int = 400,
    seed: int = 17,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Filter random NRZ blocks with measured S21 and accumulate a 2-UI eye density.

    Each block is an independent random sequence (periodic within the block), so
    memory stays constant while the number of observed bits scales to millions.
    Below the first S21 frequency the first value is held; above the last point
    S21 is zero with a raised-cosine taper over the top 10 %.
    No noise, equalizer, TX rise-time, RX model, or BER extrapolation is applied.
    """
    if bit_rate_gbps <= 0 or swing_vpp <= 0:
        raise ValueError("Data rate and signal swing must both be greater than zero.")
    if total_bits < 1024:
        raise ValueError("Use at least 1,024 bits for a meaningful eye view.")

    f = np.asarray(frequency_hz, dtype=float)
    h = np.asarray(s21, dtype=complex)
    block_bits = int(min(block_bits, total_bits))
    n_blocks = max(1, int(math.ceil(total_bits / block_bits)))
    actual_bits = n_blocks * block_bits

    ui_s = 1.0 / (bit_rate_gbps * 1e9)
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

    rng = np.random.default_rng(seed)
    hist = np.zeros(two_ui * v_bins, dtype=np.int64)
    t_off = (np.arange(two_ui, dtype=np.int64) * v_bins)[None, :]
    deviations: list[np.ndarray] = []
    center = 0.0
    shift = 0
    vmax = 1.0

    for b in range(n_blocks):
        bits = rng.integers(0, 2, size=block_bits, dtype=np.int8)
        tx = np.repeat((2.0 * bits - 1.0) * (swing_vpp / 2.0), oversample)
        rx = np.fft.irfft(np.fft.rfft(tx) * h_i, n=n)

        phases = _zero_crossings(rx, oversample)
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
        win = np.concatenate([r2, np.roll(r2, -1, axis=0)], axis=1)
        vi = np.clip(((win + vmax) / (2.0 * vmax) * v_bins).astype(np.int64), 0, v_bins - 1)
        hist += np.bincount((vi + t_off).ravel(), minlength=two_ui * v_bins)

        if progress is not None:
            progress((b + 1) / n_blocks)

    dens = hist.reshape(two_ui, v_bins).astype(float)
    v_edges = np.linspace(-vmax, vmax, v_bins + 1)
    centers = 0.5 * (v_edges[:-1] + v_edges[1:])

    # Statistical eye height per time column (0.1 % tails of each rail).
    pos = centers > 0
    up = dens[:, pos]
    dn = dens[:, ~pos][:, ::-1]
    cum_up = np.cumsum(up, axis=1) / np.maximum(up.sum(axis=1, keepdims=True), 1.0)
    cum_dn = np.cumsum(dn, axis=1) / np.maximum(dn.sum(axis=1, keepdims=True), 1.0)
    up_edge = centers[pos][(cum_up >= 1e-3).argmax(axis=1)]
    dn_edge = centers[~pos][::-1][(cum_dn >= 1e-3).argmax(axis=1)]
    height = up_edge - dn_edge
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

    return {
        "density": dens,
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
        "swing_vpp": swing_vpp,
    }


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
BIT_CHOICES = ["16,384", "65,536", "262,144", "1,048,576", "4,194,304"]
CMAP_CHOICES = ["inferno", "magma", "viridis", "turbo", "plasma", "cividis"]


class SIAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x880")
        self.minsize(1000, 700)
        self.configure(bg=BG)
        self.channel: ChannelData | None = None
        self.eye_result: dict[str, Any] | None = None
        self._eye_state: dict[str, Any] | None = None
        self._build_ui()
        self._set_status("Load a .s2p file to begin.")

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

        header = ttk.Frame(self, padding=(20, 16, 20, 8))
        header.pack(fill="x")
        titles = ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles, text="Signal Integrity Analyzer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(titles, text="Touchstone S-parameters · loss · impedance · statistical eye", style="Sub.TLabel").pack(anchor="w")
        ttk.Button(header, text="Load .s2p", style="Accent.TButton", command=self._choose_s2p).pack(side="right", padx=(10, 0))

        # Eye controls card
        controls = ttk.Frame(self, style="Card.TFrame", padding=(16, 12))
        controls.pack(fill="x", padx=20, pady=(4, 10))
        self.rate_var = tk.StringVar(value="5.0")
        self.swing_var = tk.StringVar(value="1.0")
        self.bits_var = tk.StringVar(value="262,144")
        self.cmap_var = tk.StringVar(value="inferno")

        def field(label: str, widget: tk.Widget) -> None:
            box = ttk.Frame(controls, style="Card.TFrame")
            box.pack(side="left", padx=(0, 16))
            ttk.Label(box, text=label, style="CardMuted.TLabel").pack(anchor="w")
            widget.pack(anchor="w", pady=(2, 0))

        field("Data rate (Gb/s)", ttk.Entry(controls, textvariable=self.rate_var, width=9))
        field("Swing (Vpp, diff-equivalent)", ttk.Entry(controls, textvariable=self.swing_var, width=9))
        field("Bits observed", ttk.Combobox(controls, textvariable=self.bits_var, values=BIT_CHOICES, width=12))
        cmap_box = ttk.Combobox(controls, textvariable=self.cmap_var, values=CMAP_CHOICES, width=10, state="readonly")
        cmap_box.bind("<<ComboboxSelected>>", lambda _e: self._redraw_eye())
        field("Colormap", cmap_box)

        action = ttk.Frame(controls, style="Card.TFrame")
        action.pack(side="left", fill="y")
        self.eye_btn = ttk.Button(action, text="Recalculate eye", style="Accent.TButton", command=self._recalculate_eye)
        self.eye_btn.pack(anchor="w", pady=(14, 0))

        right = ttk.Frame(controls, style="Card.TFrame")
        right.pack(side="right", fill="both", expand=True, padx=(16, 0))
        ttk.Label(right, text="Random NRZ filtered by measured S21 · no noise / EQ / BER model", style="CardMuted.TLabel").pack(anchor="e")
        self.progress = ttk.Progressbar(right, style="Eye.Horizontal.TProgressbar", mode="determinate", maximum=1.0)
        self.progress.pack(fill="x", pady=(14, 0))

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
        return canvas

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    # -- loading -----------------------------------------------------------
    def _choose_s2p(self) -> None:
        path = filedialog.askopenfilename(title="Select Touchstone 2-port file", filetypes=[("Touchstone .s2p", "*.s2p *.S2P"), ("All files", "*.*")])
        if path:
            self.load_s2p(path)

    def load_s2p(self, path: str | Path) -> None:
        try:
            self.channel = read_touchstone(path)
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
        return (
            "## Eye / jitter estimate (assumptions apply)\n"
            f"Data rate: {r['bit_rate_gbps']:g} Gb/s; UI: {r['ui_ps']:.3f} ps; swing: {r['swing_vpp']:g} Vpp\n"
            f"Bits observed: {r['bits']:,} (independent random blocks, {r['oversample']} samples/UI)\n"
            f"Best sample phase: {r['sample_phase_ui']:.3f} UI\n"
            f"Statistical eye height (0.1 % tails): {r['eye_height_raw_v']*1e3:.2f} mV\n"
            f"Eye width (UI − crossing spread, 0.1–99.9 %): {r['eye_width_ps']:.3f} ps\n"
            f"Crossing spread: RMS {r['jitter_rms_ps']:.3f} ps; 0.1–99.9 % {r['jitter_pp_ps']:.3f} ps; {r['crossing_count']:,} crossings\n"
            "This is deterministic pattern-dependent spread. It excludes noise, TX/RX behavior, clock recovery, equalization, and BER extrapolation; it is not a standards-compliant jitter result."
        )

    def _update_summary(self) -> None:
        parts = [self._channel_summary()]
        eye = self._eye_summary()
        if eye:
            parts.append(eye)
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
        try:
            rate = float(self.rate_var.get())
            return rate * 1e9 / 2.0 if rate > 0 else None
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
        self._draw_smith(self.ax_smith, d.s[:, 0, 0])
        self.canvas_z.draw_idle()

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

    @staticmethod
    def _draw_smith(ax: Any, gamma: np.ndarray) -> None:
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
        ax.plot(np.real(gamma), np.imag(gamma), color=ACCENT2, linewidth=1.8)
        ax.plot(np.real(gamma[0]), np.imag(gamma[0]), "o", color=GREEN, markersize=6, label="low f")
        ax.plot(np.real(gamma[-1]), np.imag(gamma[-1]), "s", color=RED, markersize=6, label="high f")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("S11 on the Smith chart")
        ax.set_xlabel("Real(Γ)")
        ax.set_ylabel("Imag(Γ)")
        ax.legend(loc="lower left")

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
                messagebox.showerror("Invalid input", "Data rate, swing, and bit count must be numbers.")
            return

        freq, s21 = self.channel.frequency_hz, self.channel.s[:, 1, 0]
        state: dict[str, Any] = {"done": False, "progress": 0.0, "result": None, "error": None, "silent": silent}
        self._eye_state = state

        def work() -> None:
            try:
                state["result"] = make_eye_and_jitter(
                    freq, s21, rate, swing, total_bits=bits,
                    progress=lambda p: state.__setitem__("progress", p),
                )
            except Exception as exc:  # reported on the Tk thread
                state["error"] = exc
            state["done"] = True

        self.eye_btn.configure(state="disabled")
        self.progress.configure(value=0.0)
        self._set_status(f"Calculating eye over {bits:,} bits…")
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
        self._set_status(f"Eye done: {self.eye_result['bits']:,} bits. Interpret jitter only under the displayed assumptions.")

    def _redraw_eye(self) -> None:
        r = self.eye_result
        if r is None:
            return
        fig = self.fig_eye
        fig.clf()
        ax = fig.add_subplot(111)
        self.ax_eye = ax

        dens = gaussian_filter(r["density"], sigma=(0.9, 1.4), mode="nearest")
        norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=float(dens.max()) or 1.0)
        im = ax.imshow(
            dens.T, origin="lower", aspect="auto", cmap=self.cmap_var.get(), norm=norm, interpolation="bilinear",
            extent=(0.0, 2.0, r["v_min"], r["v_max"]),
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
        cbar.set_label("Hit density (relative)")
        cbar.outline.set_edgecolor(BORDER)
        cbar.set_ticks([])

        ax.axhline(0.0, color=RED, linewidth=0.9, linestyle="--", alpha=0.8, label="Decision threshold")
        for k in (0, 1):
            ax.axvline(r["sample_phase_ui"] + k, color="#ffffff", linewidth=0.8, linestyle=":", alpha=0.7,
                       label="Best sample phase" if k == 0 else None)
        ax.set_xlim(0, 2)
        ax.set_ylim(r["v_min"], r["v_max"])
        ax.set_title(f"NRZ eye density from measured S21 — {r['bits']:,} bits")
        ax.set_xlabel("Time (UI)")
        ax.set_ylabel("Received voltage (V)")
        ui_ps = r["ui_ps"]
        top = ax.secondary_xaxis("top", functions=(lambda x: x * ui_ps, lambda x: x / ui_ps))
        top.set_xlabel("Time (ps)", color=MUTED)
        top.tick_params(colors=MUTED)
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
    app = SIAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()