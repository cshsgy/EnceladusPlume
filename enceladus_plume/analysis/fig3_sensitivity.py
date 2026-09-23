#!/usr/bin/env python3
"""Sensitivity of the paper's Fig. 3 curves (crack width, water level h/D, plume
mass flux vs observed mean anomaly) to the two forcing-shape parameters of the
double-cosine tide, Eq. (2): the 2f/1f amplitude ratio alpha and the 2f phase phi2.
All other parameters are held at the adopted best fit (results/diurnal_fit.json):
Delta-delta, L, w_eff* (attractor closure width), sigma_phi, phi0 and A.

Run from the package dir:
    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/fig3_sensitivity.py [--out DIR]
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_fit as R
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics, compute_overflow_rate, buffer_overflow
from enceladus_plume.gas_dynamics.lookup import GasLookupTable


def cycle(cfg, lut, L, dw, we, alpha, phi2, sigma, phi0):
    """Width, h/D and ensemble-smoothed flux over the last cycle, in observed MA."""
    D = L / 10.0; P = cfg.physical.orbital_period
    f_evap = cfg.physical.latent_heat_fusion / (cfg.physical.latent_heat_vaporization + cfg.physical.latent_heat_fusion)
    rho_w = cfg.physical.liquid_density
    t_in = np.arange(100, P + 1, 200.0)
    w_in = build_width_series(t_in, 1.0 + dw / we, we, orbital_period=P,
                              forcing_model="shifted-double-cosine",
                              second_harmonic_scale=alpha, second_harmonic_phase_deg=phi2)
    cfg.physical.equilibrium_depth = L
    w, h, t, v = liquid_dynamics(w_in, t_in, L, cfg)
    _, phi, _, _ = lut.query_vectorized(R.TB, np.clip(D - h, lut.depth.min(), lut.depth.max()),
                                        np.clip(w, lut.delta.min(), lut.delta.max()))
    gas = np.nan_to_num(phi * w)
    ov = np.nan_to_num(f_evap * rho_w * w * compute_overflow_rate(t, v, h, w_in, t_in, L, cfg))
    flux = gas + buffer_overflow(t, ov, cfg.liquid_dynamics.overflow_tau)
    m = t >= (t[-1] - P); o = np.argsort(t[m])
    MA = (t[m][o] - t[m][o][0]) / P * 360.0
    g, fs = R._ensemble_smooth(MA, flux[m][o], sigma)
    grid = np.linspace(0, 360, 721)
    obs = lambda x, xg: np.interp((grid - phi0) % 360.0, xg, x, period=360.0)
    return grid, obs(w[m][o], MA), obs(h[m][o] / D, MA), obs(fs, g)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp")
    ap.add_argument("--alphas", default="0,0.2,0.4,0.6,0.82,1.0,1.2")
    ap.add_argument("--phi2s", default="0,30,60,90,123,150")
    args = ap.parse_args()
    r = R.load_result(R.DEFAULT_RESULT); cfg = R._cfg(); lut = GasLookupTable(R.DEFAULT_LOOKUP, clean=True)
    dw, L, we = float(r["dw"]), float(r["L"]), float(r["w_eff"])
    a0, p0, sigma, phi0 = float(r["harm_scale"]), float(r["harm_phase"]), float(r["sigma"]), float(r["phi0"])
    ma_o, y_o, sig = np.loadtxt(R._DATA, delimiter=",", skiprows=1).T
    print(f"best fit: dw={dw*1e3:.1f} mm L={L/1e3:.1f} km w_eff*={we*1e3:.2f} mm alpha={a0:.2f} phi2={p0:.0f} sigma={sigma:.0f} phi0={phi0:.0f}")

    def sweep(name, values, fixed_label, make_args, tag):
        fig, axs = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
        cols = plt.cm.viridis(np.linspace(0.0, 0.95, len(values)))
        # data, normalized to the best-fit flux peak so all curves share the frame
        _, _, _, f_best = cycle(cfg, lut, L, dw, we, a0, p0, sigma, phi0)
        scale = f_best.max()
        axs[2].errorbar(ma_o, y_o / (float(r["A"]) * scale), yerr=sig / (float(r["A"]) * scale), fmt="o", ms=3.5,
                        color="k", lw=1, capsize=2, zorder=5, label="observed (Ingersoll+ 2020) / A")
        for c, val in zip(cols, values):
            al, ph = make_args(val)
            grid, w, hD, f = cycle(cfg, lut, L, dw, we, al, ph, sigma, phi0)
            lab = f"{tag}={val:g}" + ("  (best fit)" if np.isclose(val, a0 if tag == r"$\alpha$" else p0, atol=0.011) else "")
            axs[0].plot(grid, w * 1e3, color=c, lw=1.8, label=lab)
            axs[1].plot(grid, hD, color=c, lw=1.8)
            axs[2].plot(grid, f / scale, color=c, lw=1.8)
            i = np.argmax(f); print(f"  {tag}={val:g}: flux peak {f.max()/scale:.2f} at MA {grid[i]:.0f}, "
                                    f"cycle-mean {f.mean()/scale:.2f}, max h/D {hD.max():.2f} at MA {grid[np.argmax(hD)]:.0f}, min width {w.min()*1e3:.2f} mm")
        axs[0].set_ylabel("crack width [mm]"); axs[1].set_ylabel("water level $h/D$"); axs[2].set_ylabel("plume mass flux\n(norm. to best-fit peak)")
        axs[1].axhline(1.0, color="0.6", ls=":", lw=1); axs[1].text(2, 1.02, "surface", fontsize=8, color="0.4")
        axs[2].set_xlabel("observed mean anomaly [deg]"); axs[2].set_xlim(0, 360); axs[2].set_xticks(range(0, 361, 90))
        axs[0].legend(fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.0)); axs[2].legend(fontsize=8, loc="upper left")
        for a in axs: a.grid(alpha=0.3)
        fig.suptitle(name + f"\n(other parameters fixed at the best fit: {fixed_label})", fontsize=10)
        fig.tight_layout(); out = os.path.join(args.out, f"fig3_sweep_{tag.strip('$\\')}.png"); fig.savefig(out, dpi=150); print("wrote", out)

    fixed = (rf"$\Delta\delta$={dw*1e3:.1f} mm, $L$={L/1e3:.0f} km, $\delta_{{\rm eff}}^*$={we*1e3:.1f} mm, "
             rf"$\sigma_\phi$={sigma:.0f}$^\circ$, $\phi_0$={phi0:.0f}$^\circ$")
    alphas = [float(x) for x in args.alphas.split(",")]; phi2s = [float(x) for x in args.phi2s.split(",")]
    sweep(rf"Fig. 3 vs the 2f/1f amplitude ratio $\alpha$ ($\phi_2$={p0:.0f}$^\circ$ fixed)", alphas, fixed,
          lambda v: (v, p0), r"$\alpha$")
    sweep(rf"Fig. 3 vs the 2f phase $\phi_2$ ($\alpha$={a0:.2f} fixed)", phi2s, fixed,
          lambda v: (a0, v), r"$\phi_2$")


if __name__ == "__main__":
    main()
