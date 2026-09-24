#!/usr/bin/env python3
"""SI Figure S1: digitization sensitivity of the adopted single-cosine fit.
Reads results/freefit_diagnostics/sensitivity_single_d1.npz; writes the figure into the paper repo."""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.load("results/freefit_diagnostics/sensitivity_single_d1.npz"); res = d["res"]; x0 = d["x0"]
out = sys.argv[1] if len(sys.argv) > 1 else "/home/sam2/dev/enceladus_plume_paper/Figures/sensitivity.pdf"
dof = 21
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
sc = ax[0].scatter(res[:, 0], res[:, 1], c=res[:, 3] / dof, cmap="viridis", s=40, vmin=1.5, vmax=8)
ax[0].plot(x0[0], x0[1], "r*", ms=15, label="adopted fit")
ax[0].axhspan(4, 12, color="0.85", zorder=0, label="south polar shell 4--12 km (Hemingway \\& Mittal 2019)")
ax[0].set_xlabel(r"tidal width amplitude $\Delta\delta$ [mm]"); ax[0].set_ylabel(r"depth of the liquid column $L$ [km]"); ax[0].set_ylim(0.8, 13); ax[0].set_yscale("log")
ax[0].set_title("(a)", loc="left"); ax[0].legend(fontsize=8, loc="upper right"); ax[0].grid(alpha=0.3, which="both")
cb = fig.colorbar(sc, ax=ax[0]); cb.set_label(r"reduced $\chi^2$ of the refit")
ax[1].hist(res[:, 1], bins=np.geomspace(0.9, 4, 16), color="tab:blue", alpha=0.8)
ax[1].axvline(x0[1], color="r", lw=2, label=rf"adopted fit ($L$={x0[1]:.2f} km)")
q = np.percentile(res[:, 1], [16, 50, 84]); ax[1].axvspan(q[0], q[2], color="tab:blue", alpha=0.15, label=f"16--84%: {q[0]:.2f}--{q[2]:.2f} km")
ax[1].set_xscale("log"); ax[1].set_xlabel(r"refitted depth $L$ [km]"); ax[1].set_ylabel("realizations"); ax[1].set_title("(b)", loc="left"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(out); print("wrote", out)
print("median", np.round(np.median(res, axis=0), 3), "68%", np.round(np.percentile(res, [16, 84], axis=0), 3))
