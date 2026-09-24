#!/usr/bin/env python3
"""Self-sealing figure for the adopted model (single cosine, free lip, 1 m cap):
(a) orbits needed for a crack of initial width delta_0 to narrow to its closure width
at the wall-condensation supply, for several tidal width amplitudes at L = 1.55 km;
(b) the closure width delta*(Delta-delta) for several source depths (bisection grid).
    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/fig_selfsealing_d1.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import single_d1_common as C, run_fit as R
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics
from enceladus_plume.wall_budget import wall_mass_budget
st = C.setup(); cfg = st["cfg"]; P = cfg.physical.orbital_period; P_days = P / 86400
L0 = 1.55; dws = [3.5, 5.4, 8.0, 12.0, 20.0]
dep, wstar = [], []
for dw in dws:
    cfg.physical.equilibrium_depth = L0 * 1e3; we = 0.06
    t_in = np.arange(100, P + 1, 200.0); w_in = build_width_series(t_in, 1 + dw * 1e-3 / we, we, orbital_period=P)
    w, h, t, v = liquid_dynamics(w_in, t_in, L0 * 1e3, cfg)
    b = wall_mass_budget(t, h, w, v, np.full_like(t, np.nan), cfg, R.TB, n_z=300)
    dep.append(float(np.nanmax(b.net_thickness) * 1e3)); wstar.append(st["weff_of"](dw, L0) * 1e3)
    print(f"dw={dw:5.1f} mm: deposition {dep[-1]:.2f} mm/cycle/wall, closure width {wstar[-1]:.2f} mm", flush=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
w0 = np.geomspace(6.0, 1000.0, 200); cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(dws)))
for c, dw, d, ws in zip(cols, dws, dep, wstar):
    ax[0].plot(w0, np.maximum(w0 - ws, 0) / (2 * d), color=c, lw=2, label=rf"$\Delta\delta$={dw:g} mm ({d:.0f} mm/orbit/wall)")
ax[0].set_xscale("log"); ax[0].set_xlabel(r"initial crack width $\delta_0$ [mm]"); ax[0].set_ylabel("orbits to reach the closure width"); ax[0].set_title("(a)", loc="left")
ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8, title=rf"$L$={L0} km", title_fontsize=8)
axd = ax[0].secondary_yaxis("right", functions=(lambda n: n * P_days, lambda t: t / P_days)); axd.set_ylabel("time [days]")
d = np.load(C.GRID); dwg, Lg, W = d["dw_mm"], d["L_km"], d["w_eff"] * 1e3
for Lk, c in zip([1.0, 1.55, 3.0, 8.0], plt.cm.plasma(np.linspace(0.1, 0.8, 4))):
    ws = [st["weff_of"](x, Lk) for x in dwg]; ax[1].plot(dwg, np.array(ws) * 1e3, "-", color=c, lw=2, label=rf"$L$={Lk:g} km")
ax[1].plot([5.4], [st["weff_of"](5.4, 1.55) * 1e3], "r*", ms=14, label="adopted fit")
ax[1].set_xscale("log"); ax[1].set_xlabel(r"tidal width amplitude $\Delta\delta$ [mm]"); ax[1].set_ylabel(r"closure width $\delta^*$ [mm]"); ax[1].set_title("(b)", loc="left")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("/tmp/newfigs/wall_seal_regime.pdf"); print("wrote /tmp/newfigs/wall_seal_regime.pdf")
