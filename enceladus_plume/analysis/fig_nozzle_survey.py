#!/usr/bin/env python3
"""Figures for the nozzle survey.
  fig 1 (profiles): for a chosen Hg and mouth width, uniform vs throats: p, M, T, wall deposit rate, seal rate vs depth.
  fig 2 (phase maps): for each Hg and mouth width, maps over (throat depth, throat width) of
     plume throttle out/out_uniform, sonic-point depth, and the location of the fastest relative sealing.
    PYTHONPATH=. python analysis/fig_nozzle_survey.py
"""
import os, sys, glob, numpy as np
sys.path.insert(0, os.getcwd()); import run_fit as R
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
src = os.path.join(R._RESULTS, "nozzle_survey"); out = os.environ.get("ENC_PAPER_FIGS", "/tmp/newfigs"); os.makedirs(out, exist_ok=True)
P = 118800.0
D = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(os.path.join(src, "*.npz"))}
print(len(D), "cases loaded")

def load(tag): return D.get(tag)

# ---- fig 1: profiles
for Hg, dm in [(100.0, 4.5), (500.0, 10.0), (1500.0, 20.0)]:
    u = load(f"u_H{Hg:g}_m{dm:g}")
    if u is None: continue
    fig, ax = plt.subplots(1, 5, figsize=(19, 4.6), sharey=True)
    sets = [("uniform", u, "k")] + [(f"throat {dt:g} mm at {zt:g} m", load(f"t_H{Hg:g}_m{dm:g}_z{zt:g}_w{dt:g}"), c)
                                    for (zt, dt), c in zip([(5.0, 1.0), (5.0, 0.3), (50.0, 1.0), (50.0, 0.3)], ["tab:blue", "tab:cyan", "tab:red", "tab:orange"])]
    for lab, d, c in sets:
        if d is None: continue
        dep = d["d"]; ok = d["p"] > 1e-6
        ax[0].plot(d["p"], dep, color=c, lw=1.6, label=lab)
        ax[1].plot(d["M"], dep, color=c, lw=1.6)
        ax[2].plot(d["T"], dep, color=c, lw=1.6)
        ax[3].plot(-d["Ev"] / 917.0 * P * 1e3, dep, color=c, lw=1.6)
        ax[4].plot(d["seal_rate"] * P, dep, color=c, lw=1.6, label=f"{lab}: out/evap={float(d['flux_out'])/max(float(d['evap_in']),1e-30):.2f}")
    for a_ in ax: a_.grid(alpha=0.3, which="both")
    ax[0].set_yscale("log"); ax[0].set_ylim(Hg, 0.05); ax[0].set_ylabel("depth below the mouth [m]")
    ax[0].set_xscale("log"); ax[0].set_xlim(1e-2, 1e3); ax[0].set_xlabel("p [Pa]"); ax[0].set_title("(a) pressure", loc="left"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("Mach"); ax[1].set_xlim(0, 3); ax[1].axvline(1, color="0.5", lw=0.8, ls="--"); ax[1].set_title("(b) Mach", loc="left")
    ax[2].set_xlabel("gas T [K]"); ax[2].set_xlim(0, 290); ax[2].set_title("(c) gas temperature", loc="left")
    ax[3].set_xscale("log"); ax[3].set_xlim(1e-4, 30); ax[3].set_xlabel("wall deposit [mm per wall per orbit]"); ax[3].set_title("(d) condensation on the wall", loc="left")
    ax[4].set_xscale("log"); ax[4].set_xlim(1e-4, 30); ax[4].set_xlabel(r"relative sealing rate $-\,\partial\ln\delta/\partial t$ [per orbit]"); ax[4].set_title("(e) where the gap closes fastest", loc="left"); ax[4].legend(fontsize=7)
    fig.suptitle(f"Gas column of length {Hg:g} m, mouth width {dm:g} mm: uniform crack vs an internal throat (Gaussian, half-height 3 m)", fontsize=10)
    fig.tight_layout(); f = os.path.join(out, f"nozzle_profiles_H{Hg:g}_m{dm:g}.png"); fig.savefig(f, dpi=130); fig.savefig(f[:-4] + ".pdf"); print("wrote", f); plt.close(fig)

# ---- fig 2: phase maps
HG = [20.0, 100.0, 500.0, 1500.0]; DMs = [4.5, 10.0, 20.0]; ZT = [2.0, 5.0, 20.0, 50.0, 200.0]; DT = [0.3, 0.5, 1.0, 2.0, 4.5]
for dm in DMs:
    fig, ax = plt.subplots(3, len(HG), figsize=(4.2 * len(HG), 11))
    for j, Hg in enumerate(HG):
        u = load(f"u_H{Hg:g}_m{dm:g}"); out_u = float(u["flux_out"]) if u is not None else np.nan
        thr = np.full((len(DT), len(ZT)), np.nan); zs = thr.copy(); loc = thr.copy()
        for iz, zt in enumerate(ZT):
            for iw, dt in enumerate(DT):
                d = load(f"t_H{Hg:g}_m{dm:g}_z{zt:g}_w{dt:g}")
                if d is None: continue
                thr[iw, iz] = float(d["flux_out"]) / max(out_u, 1e-30)
                zs[iw, iz] = Hg - float(d["z_sonic"]) if np.isfinite(float(d["z_sonic"])) else np.nan
                loc[iw, iz] = d["d"][np.argmax(d["seal_rate"])]
        for i, (arr, title, kw) in enumerate([(thr, "plume throttle: out / out(uniform)", dict(cmap="viridis", vmin=0, vmax=1.5)),
                                              (zs, "depth of the sonic point [m]", dict(cmap="magma", norm=LogNorm(0.1, 2000))),
                                              (loc, "depth where the gap closes fastest [m]", dict(cmap="magma", norm=LogNorm(0.05, 2000)))]):
            im = ax[i, j].imshow(arr, origin="lower", aspect="auto", **kw)
            ax[i, j].set_xticks(range(len(ZT))); ax[i, j].set_xticklabels([f"{z:g}" for z in ZT]); ax[i, j].set_yticks(range(len(DT))); ax[i, j].set_yticklabels([f"{w:g}" for w in DT])
            ax[i, j].set_xlabel("throat depth [m]"); ax[i, j].set_ylabel("throat width [mm]"); ax[i, j].set_title(f"Hg={Hg:g} m: {title}", fontsize=8)
            for iz in range(len(ZT)):
                for iw in range(len(DT)):
                    if np.isfinite(arr[iw, iz]): ax[i, j].text(iz, iw, f"{arr[iw, iz]:.2g}", ha="center", va="center", fontsize=7, color="w" if i else "k")
            plt.colorbar(im, ax=ax[i, j], fraction=0.046)
    fig.suptitle(f"Throat survey, mouth width {dm:g} mm (uniform-crack plume out/evap: " + ", ".join(f"Hg={Hg:g}: {float(load(f'u_H{Hg:g}_m{dm:g}')['flux_out'])/max(float(load(f'u_H{Hg:g}_m{dm:g}')['evap_in']),1e-30):.2f}" for Hg in HG if load(f"u_H{Hg:g}_m{dm:g}") is not None) + ")", fontsize=9)
    fig.tight_layout(); f = os.path.join(out, f"nozzle_maps_m{dm:g}.png"); fig.savefig(f, dpi=120); fig.savefig(f[:-4] + ".pdf"); print("wrote", f); plt.close(fig)

# ---- fig 3: uniform crack: out/evap over (Hg, mouth width)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
HGall = [2.0, 20.0, 100.0, 500.0, 1500.0]; DMall = [1.0, 2.0, 4.5, 10.0, 20.0]
A = np.full((len(DMall), len(HGall)), np.nan); B = A.copy()
for j, Hg in enumerate(HGall):
    for i, dm in enumerate(DMall):
        d = load(f"u_H{Hg:g}_m{dm:g}")
        if d is None: continue
        A[i, j] = float(d["flux_out"]) / max(float(d["evap_in"]), 1e-30); B[i, j] = float(d["deposit"]) * P / 2.0
for k, (arr, title, kw) in enumerate([(A, "plume fraction out / evap", dict(cmap="viridis", vmin=0, vmax=1)), (B, "wall deposit [kg per m per wall per orbit]", dict(cmap="magma", norm=LogNorm(1, 300)))]):
    im = ax[k].imshow(arr, origin="lower", aspect="auto", **kw); plt.colorbar(im, ax=ax[k], fraction=0.046)
    ax[k].set_xticks(range(len(HGall))); ax[k].set_xticklabels([f"{h:g}" for h in HGall]); ax[k].set_yticks(range(len(DMall))); ax[k].set_yticklabels([f"{w:g}" for w in DMall])
    ax[k].set_xlabel("gas-column length Hg [m]"); ax[k].set_ylabel("crack width [mm]"); ax[k].set_title(title, fontsize=9)
    for j in range(len(HGall)):
        for i in range(len(DMall)):
            if np.isfinite(arr[i, j]): ax[k].text(j, i, f"{arr[i, j]:.2g}", ha="center", va="center", fontsize=8, color="w" if k else "k")
fig.suptitle("Uniform crack: what escapes and what is deposited, vs column length and width", fontsize=10)
fig.tight_layout(); f = os.path.join(out, "nozzle_uniform.png"); fig.savefig(f, dpi=130); fig.savefig(f[:-4] + ".pdf"); print("wrote", f)
