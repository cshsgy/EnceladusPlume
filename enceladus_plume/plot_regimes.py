#!/usr/bin/env python3
"""Generate figures for the behavioral regimes of liquid dynamics (with overflow clamp)."""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from enceladus_plume.config import load_config
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics

logging.basicConfig(level=logging.WARNING)

cfg = load_config()
P = cfg.physical.orbital_period
L = cfg.physical.equilibrium_depth
D = L / 10.0

cfg.liquid_dynamics.n_periods = 4
cfg.liquid_dynamics.npts_velocity = 200

omega = 2.0 * np.pi / P
t_in = np.arange(100, P + 1, 200.0)

representatives = [
    (0.10,  8.0, "Case 1", "Narrow, large ratio (overflow)"),
    (0.10,  3.0, "Case 2", "Narrow, moderate ratio"),
    (0.20,  4.0, "Case 3", "Medium, large ratio"),
    (0.20,  2.0, "Case 4", "Medium, small ratio"),
    (0.50,  3.0, "Case 5", "Wide, moderate ratio"),
    (1.00,  2.0, "Case 6", "Wide, small ratio"),
]
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]

# --- Figure 1: full time series, each row auto-scaled ---
fig = plt.figure(figsize=(14, 18))
gs = GridSpec(len(representatives), 2, width_ratios=[3, 1], hspace=0.42, wspace=0.28,
              left=0.08, right=0.96, top=0.96, bottom=0.03)

for idx, (wmin, wmaxmin, label, desc) in enumerate(representatives):
    print(f"Running {label}: wmin={wmin}, wmaxmin={wmaxmin} ...")
    eps = 0.5 * (wmaxmin - 1.0)
    w_in = wmin * (1.0 + eps * (1.0 - np.cos(omega * t_in)))
    w_rec, h_rec, t_rec, _v_rec = liquid_dynamics(w_in, t_in, L, cfg)

    t_days = t_rec / 86400.0

    # --- Left panel: full h(t) ---
    ax_h = fig.add_subplot(gs[idx, 0])
    ax_h.plot(t_days, h_rec, color=colors[idx], linewidth=0.8)
    ax_h.axhline(0, color="gray", linewidth=0.4, linestyle="--")

    h_max = np.nanmax(h_rec)
    h_min = np.nanmin(h_rec)
    margin = max(0.15 * (h_max - h_min), 5.0)
    ax_h.set_ylim(h_min - margin, h_max + margin)

    if h_max > D * 0.8:
        ax_h.axhline(D, color="red", linewidth=0.7, linestyle=":",
                     alpha=0.6, label=f"surface +{D:.0f} m")
    if h_min < -D * 0.5:
        ax_h.axhline(-L, color="blue", linewidth=0.7, linestyle=":",
                     alpha=0.6, label=f"ice floor −{L:.0f} m")
    if h_max > D * 0.8 or h_min < -D * 0.5:
        ax_h.legend(fontsize=7, loc="best")

    ax_h.set_ylabel("h  (m)", fontsize=10)
    if idx == len(representatives) - 1:
        ax_h.set_xlabel("Time  (days)", fontsize=10)

    title_str = (f"{label}: {desc}\n"
                 f"$w_{{min}}$ = {wmin} m,  $w_{{max}}/w_{{min}}$ = {wmaxmin}")
    ax_h.set_title(title_str, fontsize=10, loc="left", fontweight="bold")

    # --- Right panel: last period, auto-scaled ---
    ax_z = fig.add_subplot(gs[idx, 1])
    last_mask = t_rec > (cfg.liquid_dynamics.n_periods - 1) * P
    if np.sum(last_mask) > 10:
        t_phase = (t_rec[last_mask] - t_rec[last_mask][0]) / P
        h_lp = h_rec[last_mask]
    else:
        t_phase = t_rec / (P * cfg.liquid_dynamics.n_periods)
        h_lp = h_rec

    ax_z.plot(t_phase, h_lp, color=colors[idx], linewidth=1.0)
    ax_z.axhline(0, color="gray", linewidth=0.4, linestyle="--")

    lp_max = np.nanmax(h_lp)
    lp_min = np.nanmin(h_lp)
    lp_margin = max(0.15 * (lp_max - lp_min), 5.0)
    ax_z.set_ylim(lp_min - lp_margin, lp_max + lp_margin)

    if lp_max > D * 0.8:
        ax_z.axhline(D, color="red", linewidth=0.6, linestyle=":", alpha=0.5)

    h_amp = lp_max - lp_min
    ax_z.set_title(f"Last period (amp={h_amp:.0f} m)", fontsize=9)
    if idx == len(representatives) - 1:
        ax_z.set_xlabel("Phase", fontsize=9)
    ax_z.set_ylabel("h  (m)", fontsize=9)

    for ax in (ax_h, ax_z):
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

fig.savefig("liquid_regimes.png", dpi=180)
print("Saved liquid_regimes.png")

# --- Figure 2: zoomed last-period, all 6 cases ---
fig2, axes2 = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
axes2 = axes2.flatten()

for i, (wmin, wmaxmin, label, desc) in enumerate(representatives):
    ax = axes2[i]
    print(f"Zoom: {label} wmin={wmin}, wmaxmin={wmaxmin} ...")
    eps = 0.5 * (wmaxmin - 1.0)
    w_in = wmin * (1.0 + eps * (1.0 - np.cos(omega * t_in)))
    w_rec, h_rec, t_rec, _v_rec = liquid_dynamics(w_in, t_in, L, cfg)

    last_mask = t_rec > (cfg.liquid_dynamics.n_periods - 1) * P
    if np.sum(last_mask) > 10:
        t_phase = (t_rec[last_mask] - t_rec[last_mask][0]) / P
        h_lp = h_rec[last_mask]
    else:
        t_phase = t_rec / (P * cfg.liquid_dynamics.n_periods)
        h_lp = h_rec

    ax.plot(t_phase, h_lp, color=colors[i], linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")

    lp_max = np.nanmax(h_lp)
    lp_min = np.nanmin(h_lp)
    lp_margin = max(0.15 * (lp_max - lp_min), 5.0)
    ax.set_ylim(lp_min - lp_margin, lp_max + lp_margin)

    if lp_max > D * 0.8:
        ax.axhline(D, color="red", linewidth=0.7, linestyle=":", alpha=0.5,
                   label=f"surface +{D:.0f} m")

    h_mean = np.mean(h_lp)
    h_amp = lp_max - lp_min
    n_surface = np.sum(h_lp >= D - 1.0)
    pct_surface = 100.0 * n_surface / len(h_lp)

    ax.axhline(h_mean, color=colors[i], linewidth=0.8, linestyle=":",
               label=f"mean = {h_mean:+.1f} m")

    clamp_str = f", surface clamp {pct_surface:.0f}%" if pct_surface > 0.5 else ""
    ax.set_title(f"{label}: {desc}\n"
                 f"$w_{{min}}$={wmin} m, ratio={wmaxmin}, "
                 f"amp={h_amp:.0f} m{clamp_str}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Phase (last orbital period)", fontsize=9)
    ax.set_ylabel("h  (m)", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

fig2.savefig("liquid_regimes_zoom.png", dpi=180)
print("Saved liquid_regimes_zoom.png")
