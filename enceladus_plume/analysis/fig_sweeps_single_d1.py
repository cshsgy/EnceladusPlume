#!/usr/bin/env python3
"""Advisor-requested sweep figure for the adopted single-cosine model: how the
crack width, water level and plume flux (vs the data) change with the two physical
parameters, tidal width amplitude and source depth, each at its self-consistent
closure width (grid). Everything else at the adopted fit (sigma_phi, phi0 free
re-solved per curve? no: sigma fixed, phi0/A re-solved by weighted LSQ per curve).
    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/fig_sweeps_single_d1.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import single_d1_common as C, run_fit as R
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics
st = C.setup(); lut, cfg = st["lut"], st["cfg"]
fit = R.load_result(os.path.join(R._RESULTS, "diurnal_fit_free_single_sc_fine_d1.json"))
dw0, L0, sig0, A0 = float(fit["dw"]) * 1e3, float(fit["L"]) / 1e3, float(fit["sigma"]), float(fit["A"])
ma_o, y_o, so = st["ma_o"], st["y_o"], st["sig"]; w_o = R._weights(ma_o, so)
grid = np.linspace(0, 360, 721); CL = 500e3

def curves(dw_mm, L_km):
    we = st["weff_of"](dw_mm, L_km); L = L_km * 1e3; D = L / 10; cfg.physical.equilibrium_depth = L
    P = cfg.physical.orbital_period; t_in = np.arange(100, P + 1, 200.0)
    w_in = build_width_series(t_in, 1 + dw_mm * 1e-3 / we, we, orbital_period=P)
    w, h, t, v = liquid_dynamics(w_in, t_in, L, cfg)
    MA, fl = R._flux_curve(cfg, L, dw_mm * 1e-3, we, lut)
    o = np.argsort(MA); g, fs = R._ensemble_smooth(MA[o], fl[o], sig0)
    p0, A, c2 = R._best_phi_A(g, fs, ma_o, y_o, w_o)
    m = t >= t[-1] - P; oo = np.argsort(t[m]); MAc = (t[m][oo] - t[m][oo][0]) / P * 360
    obs = lambda x, xg: np.interp((grid - p0) % 360.0, xg, x, period=360.0)
    return obs(w[m][oo], MAc) * 1e3, obs(h[m][oo] / D, MAc), A * obs(fs, g), c2 / (len(ma_o) - 3), we * 1e3, p0

def panel(values, make, tag, fname, title):
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 9.2), sharex=True)
    cols = plt.cm.viridis(np.linspace(0, 0.95, len(values)))
    axs[2].errorbar(ma_o, y_o, yerr=so, fmt="o", ms=3.5, color="k", capsize=2, zorder=5, label="observed (Ingersoll+ 2020)")
    for c, val in zip(cols, values):
        dw_mm, L_km = make(val)
        wmm, hD, model, c2, we, p0 = curves(dw_mm, L_km)
        best = np.isclose(val, dw0 if tag == "dw" else L0, rtol=0.02)
        lab = (rf"$\Delta\delta$={val:g} mm" if tag == "dw" else rf"$L$={val:g} km") + rf" ($\delta^*$={we:.1f} mm, $\chi^2$/dof={c2:.1f})" + ("  best fit" if best else "")
        axs[0].plot(grid, wmm, color=c, lw=2.2 if best else 1.5, label=lab); axs[1].plot(grid, hD, color=c, lw=2.2 if best else 1.5); axs[2].plot(grid, model, color=c, lw=2.2 if best else 1.5)
        print(f"{tag}={val:g}: w_eff*={we:.2f} mm chi2/dof={c2:.2f} phi0={p0:.0f}", flush=True)
    axs[0].set_ylabel("crack width [mm]"); axs[1].set_ylabel("water level $h/D$"); axs[1].axhline(1, color="0.6", ls=":")
    axs[2].set_ylabel("slab density [kg km$^{-1}$]"); axs[2].set_xlabel("observed mean anomaly [deg]"); axs[2].set_xlim(0, 360); axs[2].set_xticks(range(0, 361, 90))
    axs[0].legend(fontsize=7.5, ncol=2, loc="upper left"); axs[2].legend(fontsize=7.5, loc="upper left")
    for a in axs: a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(fname, dpi=160); print("wrote", fname)

panel([3.5, 4.5, 5.4, 6.5, 8.0, 11.0], lambda v: (v, L0), "dw", "/tmp/newfigs/sweep_dw.pdf",
      rf"Tidal width amplitude varied ($L$={L0:.2f} km, $\sigma_\phi$={sig0:.0f}$^\circ$ fixed; each crack at its closure width $\delta^*$)")
panel([1.0, 1.25, 1.55, 2.0, 3.0, 5.0], lambda v: (dw0, v), "L", "/tmp/newfigs/sweep_L.pdf",
      rf"Source depth varied ($\Delta\delta$={dw0:.1f} mm, $\sigma_\phi$={sig0:.0f}$^\circ$ fixed; each crack at its closure width $\delta^*$)")
