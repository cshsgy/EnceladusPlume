#!/usr/bin/env python3
"""Does the single-cosine forcing give two peaks once the near-surface approach is
resolved?  For a few (dw, L) cases and barrier thicknesses dlt, compute the
self-consistent closure width (water first reaching D - dlt) with the single cosine,
then the free-lip flux curve at w_eff* and slightly below, reporting the number of
peaks, spill, and the minimum gas-column length reached.
    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/near_surface_resolution.py [--lut PATH]
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run_fit as R
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics, compute_overflow_rate, buffer_overflow
from enceladus_plume.gas_dynamics.lookup import GasLookupTable
from enceladus_plume.wall_geometry import closure_width
from scipy.signal import find_peaks

ap = argparse.ArgumentParser(); ap.add_argument("--lut", default=R.DEFAULT_LOOKUP); ap.add_argument("--tag", default="")
ap.add_argument("--cases", default="7.1:10.8,10:5,20:5,48:2.25")
ap.add_argument("--dlts", default="10,3,1,0.3")
args = ap.parse_args()
lut = GasLookupTable(args.lut, clean=True); print("LUT depth min", lut.depth.min(), "n", len(lut.depth))
R.BARRIER_MODE = "free"
cases = [tuple(float(x) for x in c.split(":")) for c in args.cases.split(",")]
dlts = [float(x) for x in args.dlts.split(",")]
CL = 500e3
fig, axs = plt.subplots(len(cases), len(dlts), figsize=(4.0 * len(dlts), 3.0 * len(cases)), sharex=True, squeeze=False)
for ci, (dw_mm, L_km) in enumerate(cases):
    for di, dlt in enumerate(dlts):
        cfg = R._cfg(); cfg.liquid_dynamics.barrier_delta = dlt; L = L_km * 1e3; cfg.physical.equilibrium_depth = L; D = L / 10
        P = cfg.physical.orbital_period; rho = cfg.physical.liquid_density
        f_evap = cfg.physical.latent_heat_fusion / (cfg.physical.latent_heat_vaporization + cfg.physical.latent_heat_fusion)
        we_star, ok = closure_width(cfg, dw_mm * 1e-3)
        if not ok: print(f"dw={dw_mm} L={L_km} dlt={dlt}: no closure"); continue
        ax = axs[ci, di]
        for fac, c in [(1.0, "k"), (0.97, "tab:blue"), (0.94, "tab:red")]:
            we = we_star * fac
            t_in = np.arange(100, P + 1, 200.0)
            w_in = build_width_series(t_in, 1 + dw_mm * 1e-3 / we, we, orbital_period=P)
            w, h, t, v = liquid_dynamics(w_in, t_in, L, cfg)
            _, phi, _, _ = lut.query_vectorized(R.TB, np.clip(D - h, lut.depth.min(), lut.depth.max()), np.clip(w, lut.delta.min(), lut.delta.max()))
            gas = np.nan_to_num(phi * w)
            spill = rho * w * compute_overflow_rate(t, v, h, w_in, t_in, L, cfg)
            tot = gas + buffer_overflow(t, f_evap * spill, cfg.liquid_dynamics.overflow_tau)
            m = t >= t[-1] - P; o = np.argsort(t[m]); MA = (t[m][o] - t[m][o][0]) / P * 360
            g, fs = R._ensemble_smooth(MA, tot[m][o], 0.0)
            pk, _ = find_peaks(np.concatenate([fs, fs[:len(fs)//2]]), prominence=0.05 * fs.max())
            npk = len(set(np.mod(pk, len(fs)))) if len(pk) else 0
            col_min = (D - h[m][o]).min()
            print(f"dw={dw_mm:5.1f} L={L_km:5.2f} dlt={dlt:4.1f} w_eff={we*1e3:6.2f} ({fac:.2f} w*): peaks={npk} "
                  f"gas mean {np.trapz(gas[m][o],t[m][o])/P*CL:7.0f} peak {gas[m][o].max()*CL:8.0f} | spill liq mean {np.trapz(spill[m][o],t[m][o])/P*CL:8.0f} | min column {col_min:6.2f} m  h_max/D {h[m][o].max()/D:.4f}", flush=True)
            ax.plot(g, fs * CL, color=c, lw=1.5, label=f"{fac:.2f} w* ({we*1e3:.1f} mm)")
        ax.set_yscale("log"); ax.set_title(rf"$\Delta\delta$={dw_mm:g} mm, L={L_km:g} km, cap layer {dlt:g} m", fontsize=9); ax.grid(alpha=0.3)
        if di == 0: ax.set_ylabel("flux [kg/s]")
        if ci == len(cases) - 1: ax.set_xlabel("model MA [deg]"); ax.set_xticks(range(0, 361, 90))
        ax.legend(fontsize=7)
fig.suptitle("Single cosine, free lip: flux at and just below the closure width, vs cap-layer thickness", fontsize=10)
fig.tight_layout(); out = f"/tmp/freefit/near_surface{args.tag}.png"; fig.savefig(out, dpi=120); print("wrote", out)
