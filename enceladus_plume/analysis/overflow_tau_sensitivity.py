#!/usr/bin/env python3
"""How much does the overflow-reservoir residence time tau matter?  At the adopted
best fit, compute the raw (unbuffered) overflow, its cycle budget, and the total
flux curve for several tau; also the Hertz-Knudsen numbers behind tau ~ rho*d_pond/j.
Run from the package dir: OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/overflow_tau_sensitivity.py
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_fit as R
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics, compute_overflow_rate, buffer_overflow, BARRIER_DELTA
from enceladus_plume.gas_dynamics.lookup import GasLookupTable

r = R.load_result(R.DEFAULT_RESULT); cfg = R._cfg(); lut = GasLookupTable(R.DEFAULT_LOOKUP, clean=True)
dw, L, we = float(r["dw"]), float(r["L"]), float(r["w_eff"])
alpha, phi2, sigma, phi0 = float(r["harm_scale"]), float(r["harm_phase"]), float(r["sigma"]), float(r["phi0"])
cfg.physical.equilibrium_depth = L
D = L / 10.0; P = cfg.physical.orbital_period
Lv, Lf = cfg.physical.latent_heat_vaporization, cfg.physical.latent_heat_fusion
f_evap = Lf / (Lv + Lf); rho_w = cfg.physical.liquid_density
print(f"Lv={Lv:.3g} Lf={Lf:.3g} f_evap={f_evap:.3f}  BARRIER_DELTA={BARRIER_DELTA} m  config tau={cfg.liquid_dynamics.overflow_tau} s")

t_in = np.arange(100, P + 1, 200.0)
w_in = build_width_series(t_in, 1.0 + dw / we, we, orbital_period=P, forcing_model="shifted-double-cosine",
                          second_harmonic_scale=alpha, second_harmonic_phase_deg=phi2)
w, h, t, v = liquid_dynamics(w_in, t_in, L, cfg)
_, phi, _, phi_s = lut.query_vectorized(R.TB, np.clip(D - h, lut.depth.min(), lut.depth.max()), np.clip(w, lut.delta.min(), lut.delta.max()))
gas = np.nan_to_num(phi * w)
dhdt_of = compute_overflow_rate(t, v, h, w_in, t_in, L, cfg)          # m/s lost to barrier
liq_raw = rho_w * w * dhdt_of                                          # kg/s per m, liquid crossing the lip
m = t >= (t[-1] - P); o = np.argsort(t[m]); tm = t[m][o]; MA = (tm - tm[0]) / P * 360.0
CL = 500e3
print(f"cycle-mean vapor at vent {np.trapz(gas[m][o], tm)/P*CL:.0f} kg/s")
print(f"raw liquid overflow: cycle-mean {np.trapz(liq_raw[m][o], tm)/P*CL:.2f} kg/s, peak {liq_raw[m][o].max()*CL:.1f} kg/s, "
      f"vapor from it (x f_evap) cycle-mean {np.trapz(liq_raw[m][o], tm)/P*CL*f_evap:.2f} kg/s")
at = h[m][o] >= D - BARRIER_DELTA
print(f"time at the surface barrier: {at.mean()*100:.1f}% of the cycle ({at.mean()*P/3600:.1f} h); h_max/D={h[m][o].max()/D:.4f}; "
      f"max |v| {np.abs(v[m][o]).max():.2f} m/s; mean |v| at barrier {np.abs(v[m][o][at]).mean() if at.any() else 0:.3f} m/s; max dh/dt lost {dhdt_of[m][o].max():.2e} m/s")
# naive volume argument: excess volume the piston would push above the surface
# Hertz-Knudsen numbers for tau ~ rho_w d_pond / j
Rg = 461.5; T = 273.15; psat = 611.0
j_vac = psat / np.sqrt(2 * np.pi * Rg * T)
print(f"Hertz-Knudsen into vacuum at 273 K: {j_vac:.2f} kg m^-2 s^-1 -> tau=1800 s implies d_pond={1800*j_vac/rho_w*100:.0f} cm (if j=j_vac)")
for pb in [0, 100, 300, 500, 600]:
    j = (psat - pb) / np.sqrt(2 * np.pi * Rg * T); print(f"  back pressure {pb:3d} Pa: j={j:.3f} kg/m2/s, d_pond for tau=0.5 h = {1800*j/rho_w*100:.1f} cm, tau for d_pond=1 cm = {rho_w*0.01/max(j,1e-9)/60:.1f} min")
# energy-limited: freezing crust supplies latent heat; conductive throttling through crust thickness s: j = k dT/(s Lv)
k_ice = 2.2; dT = 273 - 200
for s_cm in [0.1, 1, 10]:
    j = k_ice * dT / (s_cm * 1e-2 * Lv); print(f"  conduction-limited through {s_cm} cm crust: j={j:.4f} kg/m2/s")

taus = [60, 600, 1800, 3600, 3 * 3600, 10 * 3600]
grid = np.linspace(0, 360, 721); obs = lambda x, xg: np.interp((grid - phi0) % 360.0, xg, x, period=360.0)
fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
g0, gs0 = R._ensemble_smooth(MA, gas[m][o], sigma)
ax[0].plot(grid, obs(gs0, g0) * CL, "k--", lw=1.5, label="vapor only (no overflow)")
for tau in taus:
    ov = buffer_overflow(t, f_evap * liq_raw, tau)
    g, fs = R._ensemble_smooth(MA, (gas + ov)[m][o], sigma)
    ax[0].plot(grid, obs(fs, g) * CL, lw=1.6, label=rf"$\tau$={tau/3600:g} h")
    go, fo = R._ensemble_smooth(MA, ov[m][o], sigma)
    ax[1].plot(grid, obs(fo, go) * CL, lw=1.6, label=rf"$\tau$={tau/3600:g} h")
    print(f"tau={tau/3600:5.2f} h: overflow-vapor peak {obs(fo,go).max()*CL:.2f} kg/s; total peak change vs vapor-only {(obs(fs,g).max()-obs(gs0,g0).max())*CL:+.2f} kg/s")
gr, fr = R._ensemble_smooth(MA, (f_evap * liq_raw)[m][o], sigma)
ax[1].plot(grid, obs(fr, gr) * CL, "k:", lw=1.2, label="unbuffered")
ax[0].set_ylabel("total plume mass flux [kg s$^{-1}$]"); ax[1].set_ylabel("overflow-derived vapor [kg s$^{-1}$]")
ax[1].set_xlabel("observed mean anomaly [deg]"); ax[1].set_xlim(0, 360); ax[1].set_xticks(range(0, 361, 90))
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8); [a.grid(alpha=0.3) for a in ax]
fig.suptitle(rf"Overflow reservoir residence time $\tau$ at the best fit ($\Delta\delta$={dw*1e3:.1f} mm, $L$={L/1e3:.0f} km, $\alpha$={alpha:.2f}, $\phi_2$={phi2:.0f}$^\circ$)", fontsize=10)
fig.tight_layout(); fig.savefig("/tmp/overflow_tau_sweep.png", dpi=150); print("wrote /tmp/overflow_tau_sweep.png")
