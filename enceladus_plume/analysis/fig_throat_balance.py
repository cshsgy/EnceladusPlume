#!/usr/bin/env python3
"""Melt vs condensation at a water-line throat, from analysis/throat_balance.py (reads the log lines)."""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
recs = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
Ls = sorted(set(r["L_km"] for r in recs)); out = sys.argv[2]
fig, ax = plt.subplots(1, len(Ls), figsize=(4.2 * len(Ls), 4.4), sharey=True)
for a, L in zip(np.atleast_1d(ax), Ls):
    rr = sorted([r for r in recs if r["L_km"] == L], key=lambda r: r["dt_mm"]); x = [r["dt_mm"] for r in rr]
    a.loglog(x, [max(r["melt_mm"], 1e-6) for r in rr], "o-", color="tab:red", lw=2, label="frictional melt at the throat")
    a.loglog(x, [r["dep_mm"] for r in rr], "s--", color="tab:blue", lw=2, label="condensation at the throat (exposed part of the orbit)")
    for r in rr: a.annotate(f"h$_{{max}}$/D={r['hmax_over_D']:.2f}\nRe={r['Re']:.0f}", (r["dt_mm"], max(r["melt_mm"], 1e-6)), xytext=(3, -14), textcoords="offset points", fontsize=6.5, color="tab:red")
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.set_title(f"L = {L:g} km (D = {rr[0]['D']:.0f} m, mouth {rr[0]['dm_mm']:.1f} mm)", fontsize=9); a.set_xlabel("throat width at the water line [mm]"); a.grid(alpha=.3, which="both")
np.atleast_1d(ax)[0].set_ylabel("mm per wall per orbit"); np.atleast_1d(ax)[0].legend(fontsize=7)
fig.suptitle("Throat at the equilibrium water level (half-height 20 m): can tidal frictional melting balance the condensation?", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=130); print("wrote", out)
