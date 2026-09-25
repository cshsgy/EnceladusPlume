#!/usr/bin/env python3
"""SI figure: how the adopted single-cosine solution responds to adding a second
harmonic. (a) alpha varied at phi2 = 30 deg; (b) phi2 varied at alpha = 0.4. Dw, L and
sigma_phi are held at the adopted fit; the closure width is recomputed by bisection
for each forcing (free lip, 1 m cap, fine LUT); phi0 and A are re-solved per curve.
    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/fig_sweeps_2f_d1.py [out.pdf]
"""
import os, sys, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import single_d1_common as C, run_fit as R
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics
from enceladus_plume.wall_geometry import closure_width
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/sam2/dev/enceladus_plume_paper/Figures/sweeps_2f.pdf"
fit = R.load_result(os.path.join(R._RESULTS, "diurnal_fit_free_single_sc_fine_d1.json"))
dw0, L0, sig0 = float(fit["dw"]), float(fit["L"]), float(fit["sigma"])
grid = np.linspace(0, 360, 721)

def curves(alpha, phi2):
    st = C.setup(); cfg, lut = st["cfg"], st["lut"]; cfg.physical.equilibrium_depth = L0
    fm = "shifted-double-cosine"; fp = dict(second_harmonic_scale=alpha, second_harmonic_phase_deg=phi2)
    we, ok = closure_width(cfg, dw0, forcing_model=fm, forcing_params=fp, w_lo=1.5e-3, w_hi=0.03)
    D = L0 / 10; P = cfg.physical.orbital_period; t_in = np.arange(100, P + 1, 200.0)
    w_in = build_width_series(t_in, 1 + dw0 / we, we, orbital_period=P, forcing_model=fm, **fp)
    w, h, t, v = liquid_dynamics(w_in, t_in, L0, cfg)
    MA, fl = R._flux_curve(cfg, L0, dw0, we, lut, harm_scale=alpha, harm_phase=phi2)
    o = np.argsort(MA); g, fs = R._ensemble_smooth(MA[o], fl[o], sig0)
    ma_o, y_o, so = st["ma_o"], st["y_o"], st["sig"]
    p0, A, c2 = R._best_phi_A(g, fs, ma_o, y_o, R._weights(ma_o, so))
    m = t >= t[-1] - P; oo = np.argsort(t[m]); MAc = (t[m][oo] - t[m][oo][0]) / P * 360
    ob = lambda x, xg: np.interp((grid - p0) % 360.0, xg, x, period=360.0)
    return dict(w=ob(w[m][oo], MAc) * 1e3, h=ob(h[m][oo] / D, MAc), f=A * ob(fs, g), c2=c2 / 19, we=we * 1e3, a=alpha, p=phi2)

alphas = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]; phis = [0, 30, 60, 90, 120, 150]
jobs = [(a, 30.0) for a in alphas] + [(0.4, float(p)) for p in phis]
res = Parallel(n_jobs=len(jobs))(delayed(curves)(a, p) for a, p in jobs)
st = C.setup(); ma_o, y_o, so = st["ma_o"], st["y_o"], st["sig"]
fig, ax = plt.subplots(3, 2, figsize=(12.5, 10.5), sharex="col", sharey="row")
for col, sub, key, title in [(0, res[:6], "a", rf"(a) $\alpha$ varied, $\phi_2=30^\circ$"), (1, res[6:], "p", rf"(b) $\phi_2$ varied, $\alpha=0.4$")]:
    cols = plt.cm.viridis(np.linspace(0, 0.95, len(sub)))
    ax[2, col].errorbar(ma_o, y_o, yerr=so, fmt="o", ms=3, color="k", capsize=2, zorder=5, label="observed")
    for c, r in zip(cols, sub):
        lab = (rf"$\alpha$={r['a']:g}" if key == "a" else rf"$\phi_2$={r['p']:g}$^\circ$") + rf" ($\delta^*$={r['we']:.1f} mm, $\chi^2_\nu$={r['c2']:.1f})"
        lw = 2.4 if (key == "a" and r["a"] == 0) else 1.4
        ax[0, col].plot(grid, r["w"], color=c, lw=lw, label=lab); ax[1, col].plot(grid, r["h"], color=c, lw=lw); ax[2, col].plot(grid, r["f"], color=c, lw=lw)
        print(f"alpha={r['a']:.2f} phi2={r['p']:.0f}: w_eff*={r['we']:.2f} mm chi2/dof={r['c2']:.2f}", flush=True)
    ax[1, col].axhline(1, color="0.6", ls=":"); ax[0, col].legend(fontsize=8.5, loc="upper left", framealpha=0.9); ax[2, col].legend(fontsize=8.5, loc="upper left")
    ax[0, col].set_title(title, loc="left", fontsize=11); ax[2, col].set_xlabel("observed mean anomaly [deg]"); ax[2, col].set_xlim(0, 360); ax[2, col].set_xticks(range(0, 361, 90))
ax[0, 0].set_ylabel("crack width [mm]"); ax[1, 0].set_ylabel("water level $h/D$"); ax[2, 0].set_ylabel("slab density [kg km$^{-1}$]")
for a in ax.flat: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT); print("wrote", OUT)
