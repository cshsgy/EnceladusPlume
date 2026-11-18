#!/usr/bin/env python3
"""Figure for the coupled lining/gas evolution runs: (a) gap at the closing phase vs depth at
several times, (b) neck width and exposed-mean gap vs orbit, (c) plume (orbit mean, 500 km) and
vapor-front depth vs orbit.  PYTHONPATH=. python analysis/fig_coupled.py L5_crush L5_squash ...
"""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.getcwd()); import run_fit as R
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = os.path.join(R._RESULTS, "coupled"); out = os.environ.get("ENC_FIGS", "/home/sam/.local/share/slack-agents/qisha/workspace/figs")
tags = sys.argv[1:]; n = len(tags)
fig, ax = plt.subplots(3, n, figsize=(4.6 * n, 11), squeeze=False)
for j, tag in enumerate(tags):
    d = np.load(os.path.join(src, f"coupled_{tag}.npz")); h = json.loads(str(d["hist"])); zeta = d["zeta"]; L = float(d["L"]); D = float(d["D"]); dm = float(d["dm"]); surface = L + D
    its = sorted(glob.glob(os.path.join(src, f"coupled_{tag}_it*.npz"))); pick = its[:: max(1, len(its) // 5)] + its[-1:]
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(pick)))
    for f, c in zip(pick, cols):
        e = np.load(f); ax[0, j].plot((dm - 2 * e["e"]) * 1e3, np.clip(surface - zeta, 0.1, None), color=c, lw=1.5, label=f"{float(e['orbit']):.0f} orbits")
    ax[0, j].axhline(D, color="0.4", ls="--", lw=.8); ax[0, j].text(0.2, D, " equilibrium water level", fontsize=7, va="bottom", color="0.4")
    ax[0, j].axhline(D - float(h[-1]["hmax_over_D"]) * D + 0.1, color="tab:blue", ls=":", lw=.8)
    ax[0, j].set_yscale("log"); ax[0, j].set_ylim(1.3 * D + L * 0.2, 0.1); ax[0, j].set_xlim(0, dm * 1e3 * 1.05); ax[0, j].set_xlabel("gap at the closing phase [mm]"); ax[0, j].set_ylabel("depth below the mouth [m]")
    ax[0, j].set_title(f"{tag}: L = {L/1e3:g} km, mouth {dm*1e3:.1f} mm", fontsize=9); ax[0, j].legend(fontsize=7); ax[0, j].grid(alpha=.3, which="both")
    o = [r["orbit"] for r in h]
    wl = (zeta > L - 0.5 * D) & (zeta < L + 0.7 * D); neck = []
    for f in its:
        e = np.load(f); neck.append(float((dm - 2 * e["e"][wl]).min() * 1e3))
    ax[1, j].plot([float(np.load(f)["orbit"]) for f in its], neck, "o-", color="tab:red", ms=3, label="narrowest gap within 0.3 D of the water line")
    ax[1, j].plot(o, [r["gap_mean_exposed_mm"] for r in h], "s-", color="tab:orange", ms=3, label="mean gap of the exposed wall")
    ax[1, j].plot(o, [r["gap_mouth_mm"] for r in h], "^-", color="tab:brown", ms=3, label="gap at the mouth")
    ax[1, j].set_ylim(0, dm * 1e3 * 1.05); ax[1, j].set_xlabel("orbits"); ax[1, j].set_ylabel("gap at the closing phase [mm]"); ax[1, j].grid(alpha=.3); ax[1, j].legend(fontsize=7)
    ax[2, j].plot(o, [r["plume_mean"] * 5e5 for r in h], "o-", color="tab:green", ms=3, label="plume, orbit mean, 500 km [kg/s]")
    ax[2, j].set_ylabel("plume [kg/s]"); ax[2, j].set_xlabel("orbits"); ax[2, j].grid(alpha=.3); ax[2, j].set_ylim(0, None)
    b = ax[2, j].twinx(); b.plot(o, [max(r["fronts"]) for r in h], "v--", color="tab:purple", ms=3, label="vapor front at low water [m]"); b.plot(o, [r["hmax_over_D"] * D for r in h], ":", color="tab:blue", label="water rise h_max [m]")
    b.set_ylabel("depth / height [m]"); b.set_ylim(0, 1.3 * D); hl, ll = ax[2, j].get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); ax[2, j].legend(hl + h2, ll + l2, fontsize=7)
fig.suptitle("Coupled lining evolution (gas: variable-width column; liquid: tapered solver; melt: Churchill friction; mechanics: ceiling + crush or squash)", fontsize=10)
fig.tight_layout(); f = os.path.join(out, "coupled_evolution.png"); fig.savefig(f, dpi=120); print("wrote", f)
