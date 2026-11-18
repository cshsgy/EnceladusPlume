#!/usr/bin/env python3
"""Coupled evolution of the ice lining with the variable-width gas column.

Per iteration: tapered liquid solve -> 6 tidal phases -> nozzle gas column from the water
level to the mouth (deposition on the exposed wall with Tw = T_sat(p), choke, throat throttling,
plume flux) -> frictional melt on the covered wall (continuity velocity, Churchill friction)
-> lining update with an adaptive number of orbits (max change 0.25 mm) -> mechanics: ceiling
(delta_r,min - g)/2, excess crushed (falls into the liquid) or squashed down the wall.
    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/coupled_evolution.py --L 10 --mech crush --tag L10_crush
"""
import argparse, os, sys, time, json, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))
from joblib import Parallel, delayed
import single_d1_common as C, run_fit as R
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.tapered import liquid_dynamics_tapered
from enceladus_plume.friction import fanning_friction_factor
from enceladus_plume.wall_geometry import _make_grid
from enceladus_plume.gas_dynamics.nozzle import solve_gas_column
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=float, default=10.0); ap.add_argument("--dw", type=float, default=5.4)
ap.add_argument("--mu", type=float, default=0.3, help="shear: friction coefficient at the wall contact"); ap.add_argument("--sigma-tide", type=float, default=0.1, help="shear: tidal normal stress across the crack [MPa]")
ap.add_argument("--slip", type=float, default=0.5, help="shear: strike-slip displacement per orbit [m]"); ap.add_argument("--eta", type=float, default=1.0, help="shear: fraction of the frictional heat that melts lining")
ap.add_argument("--mech", choices=["crush", "squash", "lidfail", "shear"], default="crush", help="lidfail: where the lining reaches the ceiling the bridged lid fails and falls (lining reset to half the ceiling, the sawtooth mean)")
ap.add_argument("--init", default=None, help="npz with e/zeta/hist to continue from"); ap.add_argument("--tsaw", type=float, default=10.0, help="cells that would bridge within this many orbits are treated as a sawtooth and do not limit the step")
ap.add_argument("--dztop", type=float, default=0.1); ap.add_argument("--ngas", type=int, default=500); ap.add_argument("--gfloor", type=float, default=0.5)
ap.add_argument("--phases", type=int, default=6); ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--demax", type=float, default=0.25, help="max lining change per iteration [mm]")
ap.add_argument("--tag", default="run"); ap.add_argument("--out", default=os.path.join(R._RESULTS, "coupled"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); RHO_ICE = 917.0

st = C.setup(); cfg = st["cfg"]; P = float(cfg.physical.orbital_period); phys, fp = cfg.physical, cfg.friction
L = a.L * 1e3; D = L / 10; surface = L + D; cfg.physical.equilibrium_depth = L
cfg.liquid_dynamics.npts_velocity = int(max(1000, L / 10.0))
dm = st["weff_of"](a.dw, a.L); gfloor = a.gfloor * 1e-3
zeta = np.unique(np.concatenate([_make_grid(surface, 400), np.linspace(L - 1.2 * D, L + D, 300)]))
t_in = np.arange(100, P + 1, 200.0); w_in = build_width_series(t_in, 1 + a.dw * 1e-3 / dm, dm, orbital_period=P)
e_max = np.full(len(zeta), 0.5 * (dm - gfloor))
print(f"[{a.tag}] L={a.L} km D={D:.0f} m mouth delta*={dm*1e3:.2f} mm mech={a.mech} g={a.gfloor} mm grid {len(zeta)}" + (f" | shear: mu={a.mu} sigma={a.sigma_tide} MPa slip={a.slip} m eta={a.eta}: Q={a.mu*a.sigma_tide*1e6*surface*a.slip:.2e} J/m/orbit = {a.eta*a.mu*a.sigma_tide*1e6*surface*a.slip/float(phys.latent_heat_fusion):.0f} kg/m/orbit meltable" if a.mech == "shear" else ""), flush=True)

def squash(e_new, e_max, zeta):
    e = e_new.copy(); dz = np.gradient(zeta); out = 0.0; order = np.argsort(-zeta)
    for j, i in enumerate(order):
        exc = e[i] - e_max[i]
        if exc <= 0: continue
        e[i] = e_max[i]
        if j + 1 < len(order): k = order[j + 1]; e[k] += exc * dz[i] / dz[k]
        else: out += exc * dz[i]
    return e, out

def gas_phase(hk, wk, e):
    """Nozzle solve at one phase: returns (Ev on zeta grid per wall <=0, flux_out, evap_in, front depth)."""
    zw = L + hk; Hg = surface - zw
    if Hg < 1.0:
        Hg = 1.0; zw = surface - 1.0
    def wf(z): return np.clip(wk - 2 * np.interp(zw + z, zeta, e), 1e-4, None)
    n = 300 if Hg <= 100 else a.ngas
    try:
        r = solve_gas_column(wf, Hg, n=n, dz_top=a.dztop, t_max=30 + 0.3 * Hg)
    except FloatingPointError as exc:
        print(f"[{a.tag}] gas phase failed (Hg={Hg:.0f} m): {exc}; deposition set to zero for this phase", flush=True)
        return np.zeros(len(zeta)), np.nan, np.nan, np.nan, False
    Ev = np.zeros(len(zeta)); above = zeta >= zw
    Ev[above] = np.nan_to_num(np.interp(zeta[above] - zw, r.z, r.Ev))
    ok = r.p > 1e-3; front = float(surface - (zw + r.z[ok].max())) if ok.any() else Hg
    return Ev, r.flux_out, r.evap_in, front, bool(r.converged)

e = np.zeros(len(zeta)); hist = []; orbit = 0.0; it0 = 0
if a.init:
    d0 = np.load(a.init); e = np.interp(zeta, d0["zeta"], d0["e"]); hist = json.loads(str(d0["hist"])); orbit = hist[-1]["orbit"]; it0 = hist[-1]["it"]
    print(f"[{a.tag}] continuing from {a.init} at orbit {orbit:.1f} (it {it0})", flush=True)
for it in range(it0 + 1, it0 + a.iters + 1):
    t0 = time.time()
    wt, ht, tt, vt = liquid_dynamics_tapered(w_in, t_in, L, cfg, e_ice_zeta=zeta, e_ice_values=e, w_floor=gfloor)
    m = tt >= tt[-1] - P; tc, hc, wc, vc = tt[m], ht[m], wt[m], vt[m]; dwc = np.gradient(wt, tt)[m]
    t_liq = time.time() - t0; hmax = float(np.nanmax(hc))
    # --- gas at a.phases phases
    tk = tc[0] + (np.arange(a.phases) + 0.5) / a.phases * P
    hk = np.interp(tk, tc, hc); wk = np.interp(tk, tc, wc)
    res = Parallel(n_jobs=a.phases)(delayed(gas_phase)(hk[k], wk[k], e) for k in range(a.phases))
    dep = np.zeros(len(zeta)); flux = []; evap = []; fronts = []
    for Ev, fo, ev, fr, conv in res:
        dep += -Ev * P / a.phases; flux.append(fo); evap.append(ev); fronts.append(fr)
    flux = [x for x in flux if np.isfinite(x)] or [np.nan]; fronts = [x for x in fronts if np.isfinite(x)] or [np.nan]
    # --- frictional melt on the covered wall
    sel = np.linspace(0, len(tc) - 1, 80).round().astype(int); dts = np.gradient(tc[sel])
    melt = np.zeros(len(zeta)); rho_w = float(phys.liquid_density); Lf = float(phys.latent_heat_fusion)
    for k in sel.tolist():
        pass
    for j, k in enumerate(sel):
        cov = zeta <= L + hc[k]
        if not cov.any(): continue
        zc = zeta[cov]; d_eff = np.clip(wc[k] - 2 * e[cov], gfloor, None)
        v = (wc[k] * vc[k] - dwc[k] * zc) / d_eff
        Cf = np.array([fanning_friction_factor(fp.liquid_model, vv, dd, rho=rho_w, mu=float(phys.liquid_viscosity), roughness=float(fp.roughness), C_lam=float(fp.C_lam)) for vv, dd in zip(v, d_eff)])
        melt[cov] += Cf * rho_w * np.abs(v) ** 3 / Lf * dts[j]
    # --- tidal shear heating at the wall contact (force control: F = sigma_tide * H on the contact zone)
    shear = np.zeros(len(zeta)); Q_total = a.mu * a.sigma_tide * 1e6 * surface * a.slip     # J per m of crack per orbit
    if a.mech == "shear":
        dzc = np.gradient(zeta); contact_time = 0.0; sig_n = []
        for j, k in enumerate(sel):
            C = wc[k] < 2 * e + gfloor
            if not C.any(): continue
            lc = float(dzc[C].sum()); contact_time += dts[j] / P
            shear[C] += Q_total * (dts[j] / P) / lc                  # J per m^2 of wall pair in this cell
            sig_n.append(a.sigma_tide * 1e6 * surface / lc)
        shear = a.eta * shear / (2 * Lf)                             # kg per wall per orbit (per wall area)
        sig_n_max = max(sig_n) / 1e6 if sig_n else 0.0
    rate = (dep - melt - shear) / RHO_ICE               # m per wall per orbit
    free = (e < e_max - 1e-6) | (rate < 0)
    if a.mech == "lidfail": free = free & (e + np.clip(rate, 0, None) * a.tsaw < e_max)   # sawtooth cells do not limit the step
    rmax = float(np.max(np.abs(rate[free]))) if free.any() else 1e-9
    cycles = float(np.clip(a.demax * 1e-3 / max(rmax, 1e-12), 1.0, 50.0))
    e_new = np.clip(e + rate * cycles, 0.0, None)
    if a.mech in ("crush", "shear"):
        removed = float(_trapz(np.clip(e_new - e_max, 0, None), zeta)) * RHO_ICE / cycles; e_new = np.minimum(e_new, e_max)
    elif a.mech == "lidfail":
        over = (e_new + np.clip(rate, 0, None) * 1.0) > e_max        # bridges within the next orbit: sawtooth regime
        removed = float(_trapz(np.where(over, np.clip(e_new - 0.5 * e_max, 0, None), 0.0), zeta)) * RHO_ICE / cycles
        e_new = np.where(over, np.minimum(e_new, 0.5 * e_max), e_new)
    else:
        e_new, V_out = squash(e_new, e_max, zeta); removed = V_out * RHO_ICE / cycles
    e = e_new; orbit += cycles
    gap_close = dm - 2 * e; wl = (zeta > L - 0.5 * D) & (zeta < L + D)
    rec = dict(it=it, orbit=orbit, cycles=cycles, hmax_over_D=hmax / D, plume_mean=float(np.mean(flux)), plume_phases=[float(x) for x in flux], evap_phases=[float(x) for x in evap],
               fronts=[float(x) for x in fronts], M_dep=float(_trapz(dep, zeta)), M_melt=float(_trapz(melt, zeta)), M_removed=removed,
               M_shear=float(_trapz(shear, zeta)), shear_contact_time=(contact_time if a.mech == "shear" else 0.0), sigma_n_max_MPa=(sig_n_max if a.mech == "shear" else 0.0),
               neck_mm=float(gap_close[wl].min() * 1e3), neck_depth=float(surface - zeta[wl][np.argmin(gap_close[wl])]),
               gap_mouth_mm=float(gap_close[-1] * 1e3), gap_mean_exposed_mm=float(gap_close[zeta > L].mean() * 1e3), t_liq=t_liq, secs=time.time() - t0)
    hist.append(rec)
    print(f"[{a.tag}] it{it:3d} orbit {orbit:6.1f} (+{cycles:4.1f}) hmax/D {hmax/D:.3f} | plume {np.mean(flux)*5e5:6.1f} kg/s/500km (phases {' '.join(f'{x*5e5:.0f}' for x in flux)}) | dep {rec['M_dep']:5.1f} melt {rec['M_melt']:5.2f} shear {rec['M_shear']:5.1f} crushed {removed:5.1f} kg/m/wall/orbit | contact {rec['shear_contact_time']:.2f} orb sig_n {rec['sigma_n_max_MPa']:6.2f} MPa | front {min(fronts):6.1f}-{max(fronts):6.1f} m | neck {rec['neck_mm']:.2f} mm @ {rec['neck_depth']:.0f} m, mouth {rec['gap_mouth_mm']:.2f}, exposed mean {rec['gap_mean_exposed_mm']:.2f} mm | liq {t_liq:.0f}s tot {rec['secs']:.0f}s", flush=True)
    np.savez(os.path.join(a.out, f"coupled_{a.tag}.npz"), zeta=zeta, e=e, e_max=e_max, dep=dep, melt=melt, shear=shear, L=L, D=D, dm=dm, gfloor=gfloor, hist=json.dumps(hist), args=json.dumps(vars(a)))
    np.savez(os.path.join(a.out, f"coupled_{a.tag}_it{it:03d}.npz"), e=e, dep=dep, melt=melt, orbit=orbit)
print(f"[{a.tag}] done", flush=True)
