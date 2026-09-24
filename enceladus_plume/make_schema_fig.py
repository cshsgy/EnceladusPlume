#!/usr/bin/env python
"""Build the Fig. 3 schematic composite (schematic_diurnal.pdf).

Top: the hand-drawn three-phase cutaway (Fig_schema.png, phases A/B/C).
Bottom: the diurnal profile the phases refer to -- now the *fitted* ensemble
model curve (identical to Fig. 10) together with the digitized observed data,
plus the water-level and crack-width context curves.  Everything is drawn in
the observed mean-anomaly frame (the model curves are shifted by the fitted
phase offset phi0) so the A/B/C markers, the ensemble curve, and the data all
share one phase axis.

Run from the package dir with the fit result and r-table lookup available:
    OMP_NUM_THREADS=1 PYTHONPATH=$PWD python make_schema_fig.py \
        --result /tmp/diurnal_fit_mle.npz --lookup /tmp/encfig_lut.npz
"""
import argparse
import os
import tempfile

import numpy as np

from run_fit import _DATA, _cfg, _ensemble_smooth, _flux_curve, load_result, DEFAULT_RESULT, DEFAULT_LOOKUP  # noqa: E402
from enceladus_plume.utils import build_width_series  # noqa: E402
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics  # noqa: E402
from enceladus_plume.gas_dynamics.lookup import GasLookupTable  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.join(_HERE, "..", "writing", "manuscript", "Figures", "Fig_schema.png")
PAPER_PNG = "/home/sam2/dev/enceladus_plume_paper/Figures/Fig_schema.png"
OUT = "/home/sam2/dev/enceladus_plume_paper/Figures/schematic_diurnal.pdf"


def build(result_path, lookup_path, out=OUT, simple=False, flybys=None, barrier=None, barrier_delta=None):
    """simple=True draws only the fitted ensemble curve vs the data (plus A/B/C markers);
    flybys is an optional dict {label: mean anomaly deg} marked as ticks along the top."""
    import run_fit as _R
    if barrier: _R.BARRIER_MODE = barrier
    if barrier_delta is not None: _R.BARRIER_DELTA_M = barrier_delta
    r = load_result(result_path)
    lut = GasLookupTable(lookup_path, clean=True)
    cfg = _cfg()
    dw, L, we = float(r["dw"]), float(r["L"]), float(r["w_eff"])
    phi0, A = float(r["phi0"]), float(r["A"])
    sigma = float(r["sigma"])
    cfg.physical.equilibrium_depth = L

    # --- ensemble model curve, exactly as Fig. 10 (plot_ensemble) ---
    MA_m, flux_m = _flux_curve(cfg, L, dw, we, lut,
                               harm_scale=float(r["harm_scale"]),
                               harm_phase=float(r["harm_phase"]))
    o = np.argsort(MA_m)
    MA_m, flux_m = MA_m[o], flux_m[o]
    g, fse = _ensemble_smooth(MA_m, flux_m, sigma)
    gcur = np.linspace(0, 360, 721)
    ens = A * np.interp((gcur - phi0) % 360.0, g, fse, period=360.0)

    # observed data (digitized, Ingersoll+ 2020)
    ma_o, y_o, sig = np.loadtxt(_DATA, delimiter=",", skiprows=1).T

    # --- water level h/D and crack width over one cycle at the FITTED params ---
    D = L / 10.0
    P = cfg.physical.orbital_period
    t_in = np.arange(100, P + 1, 200.0)
    R = 1.0 + dw / we
    w_in = build_width_series(t_in, R, we, orbital_period=P,
                              forcing_model="shifted-double-cosine",
                              second_harmonic_scale=float(r["harm_scale"]),
                              second_harmonic_phase_deg=float(r["harm_phase"]))
    w, h, t, _v = liquid_dynamics(w_in, t_in, L, cfg)
    m = t >= (t[-1] - P)
    oc = np.argsort(t[m] - t[m][0])
    MA_cyc = (t[m][oc] - t[m][oc][0]) / P * 360.0
    hD = np.clip(h[m][oc] / D, 0.0, 1.0)
    wn = w[m][oc]
    # shift model-phase -> observed mean-anomaly frame (same offset as the flux)
    obs = (MA_cyc + phi0) % 360.0
    so = np.argsort(obs)
    obs, hD, wn = obs[so], hD[so], wn[so]

    # --- normalize to the ensemble peak so plume/data/level/width share [0,1] ---
    scale = ens.max()
    ens_n = ens / scale
    y_n, sig_n = y_o / scale, sig / scale
    wn_n = wn / wn.max()

    # --- A/B/C phase markers: secondary peak, inter-peak trough, main peak ---
    def _argpeak(lo, hi, want_min=False):
        s = (gcur >= lo) & (gcur <= hi)
        idx = np.argmin(ens[s]) if want_min else np.argmax(ens[s])
        return float(gcur[s][idx])
    C = _argpeak(150, 240)               # main peak (largest)
    Asec = _argpeak(10, 80)              # secondary peak
    B = _argpeak(Asec + 10, C - 30, want_min=True)  # inter-peak trough
    print(f"markers: A(secondary)={Asec:.0f}  B(trough)={B:.0f}  C(main)={C:.0f}  phi0={phi0:.0f}")

    # --- compose ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.gridspec import GridSpec

    png = PAPER_PNG if os.path.exists(PAPER_PNG) else PNG
    img = mpimg.imread(png)
    fig = plt.figure(figsize=(7.0, 8.0))
    gs = GridSpec(2, 1, height_ratios=[1.5, 1.0], hspace=0.16)
    ax0 = fig.add_subplot(gs[0]); ax0.imshow(img); ax0.axis("off")

    ax = fig.add_subplot(gs[1])
    ax.errorbar(ma_o, y_n, yerr=sig_n, fmt="o", ms=3.5, color="k", lw=1,
                capsize=2, zorder=5, label="observed (Ingersoll+ 2020)")
    ax.plot(gcur, ens_n, "-", color="tab:red", lw=2.2,
            label="plume strength (fitted ensemble)")
    if not simple:
        ax.plot(obs, hD, "--", color="tab:blue", lw=1.8, label="water level $h/D$")
        ax.plot(obs, wn_n, ":", color="tab:brown", lw=1.8,
                label="crack width (normalized)")
    if flybys:
        # flybys: {label: [MA, MA, ...]} -- one tick per MA, one text per label (centered)
        for lab, xs in flybys.items():
            xs = np.atleast_1d(xs)
            for x in xs:
                ax.plot([x], [1.02], marker="v", color="tab:green", ms=7, clip_on=False, zorder=6)
            ax.text(float(np.mean(xs)), 0.955, lab, ha="center", va="top", fontsize=7.5, color="tab:green")
        ax.plot([], [], marker="v", ls="none", color="tab:green", label="Cassini CDA plume flybys")
    for lab, x in [("A", Asec), ("B", B), ("C", C)]:
        ax.axvline(x, color="0.35", ls=(0, (1, 1)), lw=1.0)
        ax.text(x, 1.10, lab, ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90)); ax.set_ylim(0, 1.14)
    ax.set_xlabel("mean anomaly [deg]"); ax.set_ylabel("normalized")
    if simple:
        ax.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    else:
        ax.legend(fontsize=8.0, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0, framealpha=0.9)
    ax.grid(alpha=0.3)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", default=DEFAULT_RESULT)
    ap.add_argument("--lookup", default=DEFAULT_LOOKUP)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--simple", action="store_true",
                    help="only the fitted curve vs data (no water-level / width curves)")
    ap.add_argument("--barrier", default=None); ap.add_argument("--barrier-delta", type=float, default=None)
    ap.add_argument("--flybys", default=None,
                    help="comma list label:MA[;MA...] to mark, e.g. 'E5 E17:208;209,E7:283'")
    args = ap.parse_args()
    fb = None
    if args.flybys:   # e.g. "E21 E2:84;97,E18 E5 E17:200;208;209,E7 E3:283;291"
        fb = {kv.split(":")[0]: [float(x) for x in kv.split(":")[1].split(";")]
              for kv in args.flybys.split(",")}
    build(args.result, args.lookup, out=args.out, simple=args.simple, flybys=fb, barrier=args.barrier, barrier_delta=args.barrier_delta)
