#!/usr/bin/env python3
"""Figure for the wall-lining mechanics runs (analysis/lining_mechanics_d1.py):
(a) lining thickness profiles e(zeta) at the latest iteration, (b) depth of the lining
front and liquid-column mean gap vs orbit, (c) contact stress required to keep crushing /
squashing the lining (displacement control) vs orbit, against ice strengths.
    PYTHONPATH=. python analysis/fig_lining_mechanics_d1.py crush_attr squash_down_attr squash_out_attr
"""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.getcwd()); import run_fit as R
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = os.path.join(R._RESULTS, "lining_mechanics")
tags = sys.argv[1:] or [os.path.basename(f)[7:-4] for f in sorted(glob.glob(os.path.join(src, "lining_*.npz")))]
out = os.environ.get("ENC_PAPER_FIGS", "/tmp/newfigs"); os.makedirs(out, exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4)); ax1b = ax[1].twinx(); cols = plt.cm.tab10(np.arange(len(tags)))
for c, tag in zip(cols, tags):
    d = np.load(os.path.join(src, f"lining_{tag}.npz")); h = json.loads(str(d["hist"]))
    zeta, e, emax = d["zeta"], d["e"], d["e_max"]; surface = zeta.max(); L = float(d["L_km"]) * 1e3
    depth = surface - zeta; o = np.array([r["orbit"] for r in h])
    lab = f"{tag} ({o[-1]:.0f} orbits)"
    ax[0].plot(e * 1e3, np.clip(depth, 0.1, None), color=c, lw=1.8, label=lab)
    ax[1].plot(o, [r["front_depth"] for r in h], color=c, lw=1.8, label=lab)
    ax1b.plot(o, [r["gap_liquid_mean_mm"] for r in h], color=c, lw=1.2, ls=":")
    ax[2].plot(o, [r["sigma_contact_MPa"] for r in h], color=c, lw=1.8, marker="o", ms=3, label=lab)
ax[0].plot(emax * 1e3, np.clip(surface - zeta, 0.1, None), "k--", lw=1, label=r"ceiling $(\delta_{r,min}-g)/2$")
ax[0].axhline(surface - L, color="0.5", lw=0.8); ax[0].text(0.05, surface - L, " equilibrium water level", va="bottom", fontsize=8, color="0.4")
ax[0].set_xlabel("lining thickness per wall [mm]"); ax[0].set_ylabel("depth below the mouth [m]"); ax[0].set_yscale("log"); ax[0].set_ylim(2000, 0.1); ax[0].set_title("(a) lining profile", loc="left"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
ax[1].axhline(surface - L, color="0.5", lw=0.8, ls="--"); ax[1].text(5, surface - L, "water level", fontsize=8, color="0.4", va="bottom"); ax[1].set_xlabel("orbits"); ax[1].set_ylabel("depth of the lining front [m] (solid)"); ax1b.set_ylabel("liquid-column mean gap at closure [mm] (dotted)"); ax[1].set_ylim(0, 1800); ax[1].invert_yaxis(); ax1b.set_ylim(0, 5); ax[1].set_title("(b) front propagation", loc="left"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)
ax[2].axhspan(5, 10, color="tab:red", alpha=0.15, label="brittle crushing 5-10 MPa"); ax[2].axhspan(1.0, 2.5, color="tab:orange", alpha=0.2, label="Glen creep, ~273 K, O(1) strain/orbit")
ax[2].set_yscale("log"); ax[2].set_xlabel("orbits"); ax[2].set_ylabel(r"$\sigma_{tide} H / \ell_c$  [MPa]  ($\sigma_{tide}$=0.1 MPa)"); ax[2].set_title("(c) stress needed for displacement control", loc="left"); ax[2].grid(alpha=0.3, which="both"); ax[2].legend(fontsize=7)
fig.suptitle(r"Wall-lining mechanics, adopted model ($\Delta\delta$=5.4 mm, $L$=1.55 km, $\delta_{r,min}$=4.52 mm, $g$=0.5 mm, displacement control)", fontsize=10)
fig.tight_layout(); f = os.path.join(out, "lining_mechanics.pdf"); fig.savefig(f); fig.savefig(f.replace(".pdf", ".png"), dpi=130); print("wrote", f)
