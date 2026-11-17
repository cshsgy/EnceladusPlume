#!/usr/bin/env python3
"""Does frictional melting at a throat at the tidal water line balance the condensation there?
For L = 1.55/5/10/20 km (D = L/10, mouth width = closure width for dw = 5.4 mm), put a Gaussian
rock throat of width delta_t (half-height 20 m) at the equilibrium water level, run the tapered
liquid solver, and compare, at the throat, the per-orbit frictional melt (Churchill friction,
continuity velocity) with the condensation (IP law at depth D, times the exposed fraction).
    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/throat_balance.py
"""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))
from joblib import Parallel, delayed
import single_d1_common as C, run_fit as R
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.tapered import liquid_dynamics_tapered
from enceladus_plume.friction import fanning_friction_factor
from enceladus_plume.physics import find_evap_surface
from enceladus_plume.wall_geometry import _make_grid
OUT = os.path.join(R._RESULTS, "throat_balance"); os.makedirs(OUT, exist_ok=True)
HT = 20.0; DW = 5.4; RHO_ICE = 917.0

def run(L_km, dt_mm):
    st = C.setup(); cfg = st["cfg"]; P = float(cfg.physical.orbital_period); phys, fp = cfg.physical, cfg.friction
    L = L_km * 1e3; D = L / 10; surface = L + D; cfg.physical.equilibrium_depth = L
    cfg.liquid_dynamics.npts_velocity = int(max(1000, L / 5.0))
    dm = st["weff_of"](DW, L_km); dt = dt_mm * 1e-3 if dt_mm > 0 else dm
    zeta = np.unique(np.concatenate([_make_grid(surface, 300), np.linspace(L - 6 * HT, L + 6 * HT, 241)]))
    e_rock = 0.5 * (dm - dt) * np.exp(-((zeta - L) / HT) ** 2)
    t_in = np.arange(100, P + 1, 200.0); w_in = build_width_series(t_in, 1 + DW * 1e-3 / dm, dm, orbital_period=P)
    t0 = time.time()
    wt, ht, tt, vt = liquid_dynamics_tapered(w_in, t_in, L, cfg, e_ice_zeta=zeta, e_ice_values=e_rock, w_floor=1e-4)
    m = tt >= tt[-1] - P; tc, hc, wc, vc = tt[m], ht[m], wt[m], vt[m]; dwc = np.gradient(wt, tt)[m]; dts = np.gradient(tc)
    rho_w = float(phys.liquid_density); Lf = float(phys.latent_heat_fusion)
    d_eff = max(dm - 2 * 0.5 * (dm - dt), 1e-4)        # gap at the throat centre
    melt = 0.0; vmax = 0.0; wet = 0.0
    for k in range(len(tc)):
        if L + hc[k] < L: continue                        # throat exposed
        wet += dts[k]
        v = (wc[k] * vc[k] - dwc[k] * L) / (wc[k] - 2 * np.interp(L, zeta, e_rock))
        Cf = fanning_friction_factor(fp.liquid_model, v, d_eff, rho=rho_w, mu=float(phys.liquid_viscosity), roughness=float(fp.roughness), C_lam=float(fp.C_lam))
        melt += Cf * rho_w * abs(v) ** 3 / Lf * dts[k]; vmax = max(vmax, abs(v))
    exposed = 1 - wet / P
    E = -find_evap_surface(R.TB, 68.0, 5.67e-8, D, 2.4, float(phys.latent_heat))   # kg/m^2/s per wall, Tw = Tb
    dep = E * P * exposed
    rec = dict(L_km=L_km, dt_mm=dt * 1e3, dm_mm=dm * 1e3, D=D, melt_mm=melt / RHO_ICE * 1e3, dep_mm=dep / RHO_ICE * 1e3,
               hmax_over_D=float(np.nanmax(hc)) / D, hmin_over_D=float(np.nanmin(hc)) / D, vmax=vmax,
               Re=rho_w * vmax * 2 * d_eff / float(phys.liquid_viscosity), exposed=exposed, secs=time.time() - t0)
    print(json.dumps(rec), flush=True); return rec

cases = [(L, dt) for L in [1.55, 5.0, 10.0, 20.0] for dt in [0.3, 0.5, 1.0, 2.0, 3.0, 0.0]]
recs = Parallel(n_jobs=len(cases))(delayed(run)(L, dt) for L, dt in cases)
json.dump(recs, open(os.path.join(OUT, "throat_balance.json"), "w"), indent=1); print("done")
