#!/usr/bin/env python3
"""Decompose the free-lip best fit (results/diurnal_fit_free.json): vapor vs spill,
water level, time at the surface, tau sensitivity, and a direct check of the
attractor closure width at the fitted (dw, L) in free mode.
PYTHONPATH=. python analysis/free_lip_decompose.py"""
import os, sys, time, numpy as np
sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run_fit as R
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics, compute_overflow_rate, buffer_overflow, BARRIER_DELTA
from enceladus_plume.gas_dynamics.lookup import GasLookupTable
from enceladus_plume.wall_geometry import evolve_geometry_coupled

R.BARRIER_MODE = "free"
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R._RESULTS, "diurnal_fit_free.json")
TAG = os.path.splitext(os.path.basename(RES))[0]
r = R.load_result(RES); lut = GasLookupTable(R.DEFAULT_LOOKUP, clean=True)
dw, L, we = float(r["dw"]), float(r["L"]), float(r["w_eff"])
al, p2, sig, p0, A = float(r["harm_scale"]), float(r["harm_phase"]), float(r["sigma"]), float(r["phi0"]), float(r["A"])
cfg = R._cfg(); cfg.physical.equilibrium_depth = L
D = L / 10; P = cfg.physical.orbital_period; CL = 500e3; rho = cfg.physical.liquid_density
f_evap = cfg.physical.latent_heat_fusion / (cfg.physical.latent_heat_vaporization + cfg.physical.latent_heat_fusion)
print(f"fit: dw={dw*1e3:.1f} mm L={L:.0f} m (D={D:.0f} m) w_eff*={we*1e3:.2f} mm alpha={al:.2f} phi2={p2:.0f} sigma={sig:.0f} phi0={p0:.0f} chi2/dof={float(r['chi2_red']):.2f}")
t_in = np.arange(100, P + 1, 200.0)
w_in = build_width_series(t_in, 1 + dw / we, we, orbital_period=P, forcing_model="shifted-double-cosine", second_harmonic_scale=al, second_harmonic_phase_deg=p2)
w, h, t, v = liquid_dynamics(w_in, t_in, L, cfg)
_, phi, _, phi_s = lut.query_vectorized(R.TB, np.clip(D - h, lut.depth.min(), lut.depth.max()), np.clip(w, lut.delta.min(), lut.delta.max()))
gas = np.nan_to_num(phi * w); evap = np.nan_to_num(phi_s * w)
spill_liq = rho * w * compute_overflow_rate(t, v, h, w_in, t_in, L, cfg)
m = t >= t[-1] - P; o = np.argsort(t[m]); tm = t[m][o]; MA = (tm - tm[0]) / P * 360
at = h[m][o] >= D - BARRIER_DELTA
print(f"h_max/D={h[m][o].max()/D:.3f}, h_min/D={h[m][o].min()/D:.3f}; time at surface {at.mean()*P/3600:.2f} h ({at.mean()*100:.1f}% of cycle)")
print(f"cycle-mean: vapor at vent {np.trapz(gas[m][o],tm)/P*CL:.0f} kg/s | evaporation at liquid {np.trapz(evap[m][o],tm)/P*CL:.0f} | spill liquid {np.trapz(spill_liq[m][o],tm)/P*CL:.0f} kg/s -> vapor {f_evap*np.trapz(spill_liq[m][o],tm)/P*CL:.0f} kg/s")
print(f"peak: vapor {gas[m][o].max()*CL:.0f} kg/s; spill liquid {spill_liq[m][o].max()*CL:.0f} kg/s")
grid = np.linspace(0, 360, 721); obs = lambda x, xg: np.interp((grid - p0) % 360.0, xg, x, period=360.0)
ma_o, y_o, so = np.loadtxt(R._DATA, delimiter=",", skiprows=1).T
fig, ax = plt.subplots(3, 1, figsize=(7.2, 9), sharex=True)
ax[0].plot(grid, obs(w[m][o], MA) * 1e3, "k-"); ax[0].set_ylabel("crack width [mm]")
ax[1].plot(grid, obs(h[m][o] / D, MA), "b-"); ax[1].axhline(1, color="0.6", ls=":"); ax[1].set_ylabel("water level $h/D$")
g, gs_ = R._ensemble_smooth(MA, gas[m][o], sig)
for tau, c in [(600, "tab:orange"), (1800, "tab:red"), (3600, "tab:green"), (3 * 3600, "tab:purple")]:
    ov = buffer_overflow(t, f_evap * spill_liq, tau); go, fo = R._ensemble_smooth(MA, ov[m][o], sig)
    tot = obs(gs_, g) + obs(fo, go)
    ax[2].plot(grid, tot * CL, color=c, lw=1.8 if tau == 1800 else 1.1, label=rf"total, $\tau$={tau/3600:g} h" + (" (fit)" if tau == 1800 else ""))
    if tau == 1800:
        ax[2].plot(grid, obs(gs_, g) * CL, "b--", lw=1.4, label="vapor evaporated in the crack")
        ax[2].plot(grid, obs(fo, go) * CL, "r:", lw=1.6, label="vapor from spilled water (x0.12)")
        print(f"tau=0.5h: cycle-mean total {tot.mean()*CL:.0f} kg/s; peak {tot.max()*CL:.0f} at MA {grid[tot.argmax()]:.0f}; spill share of cycle mean {obs(fo,go).mean()/tot.mean():.2f}, of peak {obs(fo,go)[tot.argmax()]/tot.max():.2f}")
    else:
        print(f"tau={tau/3600:g}h: peak {tot.max()*CL:.0f} kg/s at MA {grid[tot.argmax()]:.0f}")
ax[2].errorbar(ma_o, y_o / A * CL, yerr=so / A * CL, fmt="o", ms=3.5, color="k", capsize=2, label="observed / A")
ax[2].set_ylabel("mass flux, 500 km [kg s$^{-1}$]"); ax[2].set_xlabel("observed mean anomaly [deg]"); ax[2].set_xlim(0, 360); ax[2].set_xticks(range(0, 361, 90)); ax[2].legend(fontsize=7.5)
for a in ax: a.grid(alpha=0.3)
fig.suptitle(rf"Free-lip best fit: $\Delta\delta$={dw*1e3:.0f} mm, $L$={L/1e3:.1f} km, $\alpha$={al:.2f}, $\phi_2$={p2:.0f}$^\circ$, $\sigma_\phi$={sig:.0f}$^\circ$, $\chi^2$/dof={float(r['chi2_red']):.2f}", fontsize=10)
fig.tight_layout(); fig.savefig(f"/tmp/freefit/decompose_{TAG}.png", dpi=140); print(f"wrote /tmp/freefit/decompose_{TAG}.png")
# self-consistency: direct bisection closure width at the fitted (dw, L) vs the interpolated value used
from enceladus_plume.wall_geometry import closure_width
t0 = time.time(); we_direct, ok = closure_width(cfg, dw, forcing_model="shifted-double-cosine" if al > 0 else "single-cosine",
                                                 forcing_params=dict(second_harmonic_scale=al, second_harmonic_phase_deg=p2) if al > 0 else None,
                                                 w_lo=max(1e-3, 0.5*we), w_hi=min(0.08, 2*we))
print(f"direct closure width at (dw={dw*1e3:.1f} mm, L={L/1e3:.2f} km): {we_direct*1e3:.3f} mm (fit used {we*1e3:.3f} mm) [{(time.time()-t0):.0f} s]")
ma_o, y_o, so = np.loadtxt(R._DATA, delimiter=",", skiprows=1).T; w_o = R._weights(ma_o, so)
th = np.array([dw*1e3, L/1e3, p2, al, sig]); dof = float(r["dof"])
for lab, wv in [("interpolated", we), ("direct", we_direct)] + [(f"{x:+.1f} mm", we + x*1e-3) for x in (-0.5, -0.25, 0.25, 0.5)]:
    nll = R._neg_loglike(th, lambda a,b,wv=wv: wv, lut, cfg, ma_o, y_o, w_o); print(f"  chi2/dof at w_eff {lab:>12s} ({wv*1e3:.2f} mm): {2*nll/dof:.2f}", flush=True)
