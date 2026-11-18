#!/usr/bin/env python3
"""Per-orbit deposition and frictional melt vs depth in the current state of the coupled runs,
with the gap profile; marks the equilibrium water level and the tidal band."""
import os, sys, json, numpy as np
sys.path.insert(0, os.getcwd()); import run_fit as R
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = os.path.join(R._RESULTS, "coupled"); out = "/home/sam/.local/share/slack-agents/qisha/workspace/figs/coupled_rates.png"
tags = sys.argv[1:]; fig, ax = plt.subplots(1, len(tags), figsize=(6 * len(tags), 5.2), squeeze=False)
for a, tag in zip(ax[0], tags):
    d = np.load(os.path.join(src, f"coupled_{tag}.npz")); h = json.loads(str(d["hist"])); z = d["zeta"]; L = float(d["L"]); D = float(d["D"]); dm = float(d["dm"]); surf = L + D
    dep = surf - z; ok = dep > 0.05
    a.loglog(np.clip(d["dep"] / 917 * 1e3, 1e-6, None)[ok], dep[ok], color="tab:blue", lw=1.5, label="condensation [mm per wall per orbit]")
    a.loglog(np.clip(d["melt"] / 917 * 1e3, 1e-6, None)[ok], dep[ok], color="tab:red", lw=1.5, label="frictional melt [mm per wall per orbit]")
    a.loglog(((dm - 2 * d["e"]) * 1e3)[ok], dep[ok], color="0.3", lw=1, ls="--", label="gap at the closing phase [mm]")
    a.axhline(D, color="tab:cyan", lw=1); a.text(1.2e-6, D * 0.93, "equilibrium water level", color="tab:cyan", fontsize=8)
    a.axhspan(D * 0.0 + 1e-3, D * 1.6, color="tab:cyan", alpha=0.0)
    a.axhline(1.6 * D, color="tab:cyan", lw=1, ls=":"); a.text(1.2e-6, 1.6 * D * 0.93, "low-water line", color="tab:cyan", fontsize=8)
    a.set_ylim(1.3 * L + D, 0.05); a.set_xlim(1e-6, 30); a.set_xlabel("rate [mm per wall per orbit]  /  gap [mm]"); a.set_ylabel("depth below the mouth [m]")
    a.set_title(f"{tag}: L = {L/1e3:g} km, orbit {h[-1]['orbit']:.0f}, plume {h[-1]['plume_mean']*5e5:.0f} kg/s", fontsize=9); a.grid(alpha=.3, which="both"); a.legend(fontsize=7, loc="lower left")
fig.suptitle("Where the lining grows and where it melts, in the current state of the coupled runs", fontsize=10); fig.tight_layout(); fig.savefig(out, dpi=130); print("wrote", out)
