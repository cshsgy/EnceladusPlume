#!/usr/bin/env python3
"""Evaluate the adopted (backflow-fitted) parameters with the free-lip barrier and
report spill, flux curve and timing.  PYTHONPATH=. python analysis/free_lip_check.py"""
import os, sys, time, numpy as np
sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run_fit as R
from enceladus_plume import config as C
from enceladus_plume.gas_dynamics.lookup import GasLookupTable
r = R.load_result(R.DEFAULT_RESULT); lut = GasLookupTable(R.DEFAULT_LOOKUP, clean=True)
dw, L, we = float(r["dw"]), float(r["L"]), float(r["w_eff"])
al, p2, sig, p0, A = float(r["harm_scale"]), float(r["harm_phase"]), float(r["sigma"]), float(r["phi0"]), float(r["A"])
ma_o, y_o, so = np.loadtxt(R._DATA, delimiter=",", skiprows=1).T
grid = np.linspace(0, 360, 721)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.errorbar(ma_o, y_o / A, yerr=so / A, fmt="o", ms=3.5, color="k", capsize=2, label="observed / A (backflow fit)")
for mode, c in [("backflow", "tab:red"), ("free", "tab:blue")]:
    R.BARRIER_MODE = mode
    cfg = R._cfg(); assert cfg.liquid_dynamics.surface_barrier == mode; cfg.physical.equilibrium_depth = L
    t0 = time.time(); MA, fl = R._flux_curve(cfg, L, dw, we, lut, harm_scale=al, harm_phase=p2); dt = time.time() - t0
    o = np.argsort(MA); g, fs = R._ensemble_smooth(MA[o], fl[o], sig)
    curve = np.interp((grid - p0) % 360.0, g, fs, period=360.0) * 500e3
    print(f"{mode:9s}: {dt:5.1f} s  cycle-mean total {curve.mean():8.1f} kg/s  peak {curve.max():9.1f} kg/s at MA {grid[curve.argmax()]:.0f}")
    ax.plot(grid, curve / 500e3, color=c, lw=2, label=f"{mode} barrier, same parameters")
ax.set_yscale("log"); ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90)); ax.set_xlabel("observed mean anomaly [deg]"); ax.set_ylabel("flux per unit length (model units)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("/tmp/free_lip_check.png", dpi=130); print("wrote /tmp/free_lip_check.png")
