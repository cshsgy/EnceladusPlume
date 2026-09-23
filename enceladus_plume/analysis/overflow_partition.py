#!/usr/bin/env python3
"""Diagnose what the surface barrier does at the best fit: time at the surface, piston
rise rate vs base velocity (backflow), the model-diagnosed spill, and the free-lip upper
bound in which all displaced water spills. Run from the package dir:
    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/overflow_partition.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.getcwd())
import run_fit as R
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics, compute_overflow_rate, BARRIER_DELTA, _derivative
r = R.load_result(R.DEFAULT_RESULT); cfg = R._cfg()
dw, L, we = float(r["dw"]), float(r["L"]), float(r["w_eff"]); alpha, phi2, phi0 = float(r["harm_scale"]), float(r["harm_phase"]), float(r["phi0"])
cfg.physical.equilibrium_depth = L; D = L/10; P = cfg.physical.orbital_period; g = cfg.physical.gravity
t_in = np.arange(100, P+1, 200.0)
w_in = build_width_series(t_in, 1+dw/we, we, orbital_period=P, forcing_model="shifted-double-cosine", second_harmonic_scale=alpha, second_harmonic_phase_deg=phi2)
w, h, t, v = liquid_dynamics(w_in, t_in, L, cfg)
m = t >= t[-1]-P; t, w, h, v = t[m], w[m], h[m], v[m]
MAobs = ((t-t[0])/P*360 + phi0) % 360
w_ext = np.concatenate([w_in,w_in,w_in]); t_ext = np.concatenate([t_in-P,t_in,t_in+P]); dwdt_ext = np.gradient(w_ext, t_ext)
dwdt = np.interp(t % P, t_ext, dwdt_ext)
piston = -(h+L)/w*dwdt                      # rise rate of the top if v0 = 0  (m/s)
dhdt_free = v - (h+L)/w*dwdt                # what continuity says with the model's v0
at = h >= D - BARRIER_DELTA
of = compute_overflow_rate(t, v, h, w_in, t_in, L, cfg)
rho=1000.; CL=500e3
print(f"barrier time {at.mean()*P/3600:.2f} h, observed MA range of plateau: {MAobs[at].min():.0f}-{MAobs[at].max():.0f}")
print(f"during plateau: piston (m/s) mean {piston[at].mean():+.3f} min {piston[at].min():+.3f} max {piston[at].max():+.3f}")
print(f"during plateau: v0 (m/s) mean {v[at].mean():+.3f} min {v[at].min():+.3f} max {v[at].max():+.3f}   -> continuity dh/dt mean {dhdt_free[at].mean():+.4f}")
print(f"closing (piston>0) fraction of plateau: {(piston[at]>0).mean():.2f}")
# spill upper bound: all displaced volume while at cap & closing leaves the lip
up = np.clip(piston,0,None)*at
spill_up = np.trapz(rho*w*up, t)/P*CL
print(f"UPPER BOUND spill (all piston displacement at cap exits lip): cycle-mean {spill_up:.0f} kg/s liquid, x0.12 -> {0.12*spill_up:.0f} kg/s vapor; peak {np.max(rho*w*up)*CL:.0f} kg/s")
print(f"model diagnosed spill: {np.trapz(rho*w*of,t)/P*CL:.2f} kg/s")
# gravity-driven backflow capacity through the column at the cap: g*D*(...)/(L+D) balanced by wall friction 2Cf v^2/w
for Cf in [0.002, 0.005, 0.01]:
    vb = np.sqrt(g*D*we/(2*Cf*(L+D))); print(f"  gravity/friction backflow speed for Cf={Cf}: {vb:.3f} m/s  (piston mean at cap {piston[at].mean():+.3f})")
# total displaced volume per cycle vs plume
print(f"total volume displaced by wall closure per cycle per m: {dw*(L+D):.0f} m^3 -> {dw*(L+D)*rho*CL/P:.2e} kg/s if it all left (500 km)")
# where is the water pre-plateau: h at plateau start
i0 = np.argmax(at); print(f"plateau starts at MA {MAobs[i0]:.0f}, v0 there {v[i0]:+.3f} m/s, piston {piston[i0]:+.3f} m/s")
