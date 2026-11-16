#!/usr/bin/env python3
"""Minimalist solid-mechanics model of the wall deposit for the adopted model
(single cosine, free lip, 1 m cap, L = 1.55 km, tidal width amplitude 5.4 mm).

Model (per wall, 1-D in height zeta; rock aperture delta_r(t) prescribed = displacement control):
  * condensation adds ice on the exposed wall (existing law, exposure- and choke-gated);
  * frictional dissipation melts ice on the covered wall (existing q = Cf rho |v|^3);
  * mechanical crushing: wherever 2 e(zeta) would exceed delta_r,min - g_floor at the
    closing phase, the excess is crushed. All debris falls freely into the liquid and
    melts (the mouth is always the narrowest point, so nothing jams): f_return = 1.
  * optional --wipe: the lining is also removed wherever the water periodically reaches
    (the old stage-B assumption; thermodynamically the 272.6 K water has no superheat).
Diagnostics: contact height l_c of the crushed zone and the contact stress the shell
would need to crush it, sigma_tide * H / l_c, compared with the compressive strength.
The lined profile is then handed to the tapered liquid solver, so the water rise is
self-consistent with the taper (the test of the uniform-narrowing assumption).

    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/lining_mechanics_d1.py --delta0 4.52 --tag attr
"""
import argparse, os, sys, time, json, numpy as np
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))
import single_d1_common as C, run_fit as R
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.tapered import liquid_dynamics_tapered
from enceladus_plume.physics import vapor_pressure, evaporation_rate_simple
from enceladus_plume.wall_budget import _RG, RHO_ICE
from enceladus_plume.wall_geometry import (_condensation_flux, _make_grid,
                                           _deposition_per_cycle, _friction_melt_per_cycle)

ap = argparse.ArgumentParser()
ap.add_argument("--dw", type=float, default=5.4, help="tidal width amplitude [mm]")
ap.add_argument("--L", type=float, default=1.55, help="liquid column depth [km]")
ap.add_argument("--delta0", type=float, default=None, help="initial rock aperture at the closing phase [mm] (default: closure width)")
ap.add_argument("--gfloor", type=float, default=0.1, help="minimum gap the crushed lining leaves at the closing phase [mm]")
ap.add_argument("--cycles", type=float, default=5.0, help="orbits per iteration")
ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--nz", type=int, default=300)
ap.add_argument("--nt", type=int, default=80)
ap.add_argument("--mech", choices=["crush","squash"], default="crush", help="crush: excess lining removed (falls into the liquid); squash: excess extruded along the wall, mass conserved")
ap.add_argument("--lip", choices=["down","out"], default="down", help="squash only: excess at the lip is extruded down the crack or out of the mouth")
ap.add_argument("--wipe", action="store_true", help="also remove the lining wherever the water reaches")
ap.add_argument("--nochoke", action="store_true", help="no plume-choke cap on the deposition (upper bound)")
ap.add_argument("--sigma-tide", type=float, default=0.1, help="tidal stress [MPa] for the contact-stress diagnostic")
ap.add_argument("--sigma-c", type=float, default=5.0, help="compressive strength of ice [MPa]")
ap.add_argument("--tag", default="run")
ap.add_argument("--out", default=os.path.join(R._RESULTS, "lining_mechanics"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

st = C.setup(); cfg = st["cfg"]; lut = st["lut"]
P = float(cfg.physical.orbital_period); L = a.L * 1e3; D = L / 10.0; surface = L + D
cfg.physical.equilibrium_depth = L
phys, fp = cfg.physical, cfg.friction
Lv, Lf = float(phys.latent_heat), float(phys.latent_heat_fusion)
dstar = st["weff_of"](a.dw, a.L)
delta0 = (a.delta0 * 1e-3) if a.delta0 else dstar
gfloor = a.gfloor * 1e-3
H = surface  # shell thickness along the stripe
print(f"[{a.tag}] mech={a.mech} lip={a.lip} dw={a.dw} mm L={a.L} km delta*={dstar*1e3:.3f} mm delta_r,min={delta0*1e3:.3f} mm "
      f"gfloor={a.gfloor} mm wipe={a.wipe} choke={not a.nochoke}", flush=True)

zeta = _make_grid(surface, a.nz)
dep_flux = _condensation_flux(zeta, surface, R.TB, Lv)        # kg m^-2 s^-1, >= 0
ec = vapor_pressure(273.15) / np.sqrt(2.0 * np.pi * _RG); bv = Lv / _RG
t_in = np.arange(100, P + 1, 200.0)
w_in = build_width_series(t_in, 1 + a.dw * 1e-3 / delta0, delta0, orbital_period=P)
from enceladus_plume.friction import fanning_friction_factor
def melt_tapered(zeta, tc, hc, wc, vc, dwc, dt, e, gfloor):
    """Ice removed by wall dissipation q = Cf rho_w |v|^3 on the covered wall, with the
    velocity from continuity in the lined crack (kg/m^2 per orbit)."""
    melt = np.zeros(len(zeta)); rho_w = float(phys.liquid_density)
    for k in range(len(tc)):
        covered = zeta <= L + hc[k]
        if not np.any(covered): continue
        zc = zeta[covered]; d_eff = np.clip(wc[k] - 2 * e[covered], gfloor, None)
        v_prof = (wc[k] * vc[k] - dwc[k] * zc) / d_eff
        if fp.liquid_model == "constant":
            Cf = float(fp.liquid_Cf_constant)
        else:
            Cf = np.array([fanning_friction_factor(fp.liquid_model, vv, dd, rho=rho_w, mu=float(phys.liquid_viscosity),
                                                   roughness=float(fp.roughness), C_lam=float(fp.C_lam)) for vv, dd in zip(v_prof, d_eff)])
        melt[covered] += Cf * rho_w * np.abs(v_prof) ** 3 / Lf * dt[k]
    return melt

def squash(e_new, e_max, zeta, lip):
    """Extrude the excess lining along the wall, conserving volume per unit crack length.
    lip="down": everything is pushed down the crack (top -> bottom sweep).
    lip="out":  the contact zone attached to the mouth is squeezed toward both free ends,
                with the divide at its middle: the upper half is extruded out of the mouth.
    Returns (e, extruded_out) with extruded_out the volume lost out of the mouth."""
    e = e_new.copy(); dz = np.gradient(zeta); out = 0.0
    order = np.argsort(-zeta)                      # from the mouth downward
    if lip == "out":
        exc0 = np.clip(e - e_max, 0.0, None)
        n_top = 0
        for i in order:                            # contiguous contact zone from the mouth
            if exc0[i] > 0 or e[i] >= e_max[i] - 1e-9: n_top += 1
            else: break
        top_zone = order[:n_top]
        if n_top > 0:
            z_div = 0.5 * (zeta[top_zone].max() + zeta[top_zone].min())
            for i in top_zone:
                if zeta[i] >= z_div and exc0[i] > 0:
                    out += exc0[i] * dz[i]; e[i] = e_max[i]
    for j, i in enumerate(order):
        exc = e[i] - e_max[i]
        if exc <= 0: continue
        e[i] = e_max[i]
        if j + 1 < len(order):
            k = order[j + 1]; e[k] += exc * dz[i] / dz[k]
        else:
            out += exc * dz[i]                     # reached the floor: nowhere to go
    return e, out

e_max = np.full(len(zeta), 0.5 * (delta0 - gfloor))             # mechanical ceiling of the lining

e = np.zeros(len(zeta)); hist = []
for it in range(1, a.iters + 1):
    t0 = time.time()
    wt, ht, tt, vt = liquid_dynamics_tapered(w_in, t_in, L, cfg, e_ice_zeta=zeta, e_ice_values=e, w_floor=gfloor)
    m = tt >= (tt[-1] - P)
    tc, hc, wc, vc = tt[m], ht[m], wt[m], vt[m]; dwc = np.gradient(wt, tt)[m]
    sel = np.linspace(0, len(tc) - 1, a.nt).round().astype(int)
    tc, hc, wc, vc, dwc = tc[sel], hc[sel], wc[sel], vc[sel], dwc[sel]; dt = np.gradient(tc)
    hmax = float(np.nanmax(hc)); spill_mask = (D - hc) <= 1.0
    # choke cap from the gas table at the mouth gap (the narrowest point)
    e_top = float(np.interp(surface, zeta, e))
    w_mouth = np.clip(wc - 2 * e_top, gfloor, None)
    if a.nochoke:
        rc = np.zeros(len(tc))
    else:
        rc, *_ = lut.query_vectorized(R.TB, np.clip(D - hc, lut.depth.min(), lut.depth.max()), np.clip(w_mouth, lut.delta.min(), lut.delta.max()))
        rc = np.where(np.isfinite(rc), rc, np.nanmean(rc[np.isfinite(rc)]) if np.any(np.isfinite(rc)) else 0.77)
    phi_t = (1.0 - rc) * evaporation_rate_simple(ec, bv, R.TB)
    dep = _deposition_per_cycle(zeta, dep_flux, tc, hc, wc, phi_t, rc, e, L, dt)      # kg/m^2 per orbit
    melt = melt_tapered(zeta, tc, hc, wc, vc, dwc, dt, e, gfloor)
    e_new = e + (dep - melt) / RHO_ICE * a.cycles
    e_new = np.clip(e_new, 0.0, None)
    if a.mech == "crush":
        crushed = np.clip(e_new - e_max, 0.0, None); e_new = np.minimum(e_new, e_max); V_out = 0.0
    else:
        e_new, V_out = squash(e_new, e_max, zeta, a.lip); crushed = np.zeros(len(zeta))
    if a.wipe:
        wiped = zeta <= L + hmax; crushed[wiped] += e_new[wiped]; e_new[wiped] = 0.0
    contact = e_new >= e_max - 1e-6
    l_c = float(_trapz(contact.astype(float), zeta))
    sigma_contact = a.sigma_tide * 1e6 * H / l_c if l_c > 0 else np.inf
    M_dep, M_melt, M_crush = (float(_trapz(x, zeta)) for x in (dep, melt, crushed * RHO_ICE / a.cycles))
    M_crush += V_out * RHO_ICE / a.cycles   # squash: mass extruded out of the crack
    M_lining = float(_trapz(e_new, zeta)) * RHO_ICE
    front = float(zeta[contact].min()) if contact.any() else surface
    step = float(np.nanmax(np.abs(e_new - e))); e = e_new
    gap_min = delta0 - 2 * e
    rec = dict(it=it, orbit=it * a.cycles, hmax=hmax, spill_frac=float(spill_mask.mean()), M_dep=M_dep, M_melt=M_melt, M_crush=M_crush,
               l_c=l_c, sigma_contact_MPa=sigma_contact / 1e6, M_lining=M_lining, front_depth=surface - front, gap_mouth_mm=float(gap_min[-1] * 1e3),
               gap_mean_mm=float(gap_min.mean() * 1e3), gap_liquid_mean_mm=float(gap_min[zeta <= L].mean() * 1e3), step=step, secs=time.time() - t0)
    hist.append(rec)
    print(f"[{a.tag}] it{it:3d} orbit {rec['orbit']:5.0f}: hmax {hmax:7.2f} m  dep {M_dep:5.1f} melt {M_melt:5.1f} out {M_crush:5.1f} lining {M_lining:7.0f} kg/m/wall | front {surface-front:7.1f} m "
          f"l_c {l_c:7.1f} m  sigma_contact {sigma_contact/1e6:8.2f} MPa | gap mouth {rec['gap_mouth_mm']:.2f} mean {rec['gap_mean_mm']:.2f} liquid {rec['gap_liquid_mean_mm']:.2f} mm | step {step*1e3:.3f} mm {rec['secs']:.0f}s", flush=True)
    np.savez(os.path.join(a.out, f"lining_{a.tag}.npz"), zeta=zeta, e=e, e_max=e_max, dep=dep, melt=melt, crushed=crushed, dep_flux=dep_flux,
             delta0=delta0, dstar=dstar, dw_mm=a.dw, L_km=a.L, gfloor=gfloor, hist=json.dumps(hist), args=json.dumps(vars(a)))
    if step < 1e-6 and it > 3:
        print(f"[{a.tag}] converged", flush=True); break
print(f"[{a.tag}] done", flush=True)
