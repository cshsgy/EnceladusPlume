#!/usr/bin/env python3
"""Regime survey of the variable-width gas column (gas_dynamics/nozzle.py).
Cases: gas-column length Hg x mouth width, uniform; plus a throat (Gaussian dip of
width dt at depth zt below the mouth, 1/e half-height ht) in the gas column.
Outputs results/nozzle_survey/<tag>.npz with the profiles and the seal-rate profile.
    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/nozzle_survey.py --workers 60
"""
import argparse, os, sys, time, itertools, numpy as np
sys.path.insert(0, os.getcwd()); import run_fit as R
from joblib import Parallel, delayed
from enceladus_plume.gas_dynamics.nozzle import solve_gas_column

ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=60)
ap.add_argument("--out", default=os.path.join(R._RESULTS, "nozzle_survey")); ap.add_argument("--quick", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

HG = [2.0, 20.0, 100.0, 500.0, 1500.0]
DM = [1.0, 2.0, 4.5, 10.0, 20.0]                     # mouth width, mm
ZT = [2.0, 5.0, 20.0, 50.0, 200.0]                   # throat depth below the mouth, m
DT = [0.3, 0.5, 1.0, 2.0, 4.5]                       # throat width, mm
HT = 3.0                                             # throat 1/e half-height, m
cases = [dict(tag=f"u_H{Hg:g}_m{dm:g}", Hg=Hg, dm=dm, zt=None, dt=None) for Hg in HG for dm in DM]
cases += [dict(tag=f"t_H{Hg:g}_m{dm:g}_z{zt:g}_w{dt:g}", Hg=Hg, dm=dm, zt=zt, dt=dt)
          for Hg in HG for dm in [4.5, 10.0, 20.0] for zt in ZT for dt in DT if zt + 3 * HT < Hg and dt < dm]
if a.quick: cases = cases[:4]
print(f"{len(cases)} cases", flush=True)

def run(c):
    f = os.path.join(a.out, c["tag"] + ".npz")
    if os.path.exists(f): return c["tag"], "cached"
    dm = c["dm"] * 1e-3
    if c["zt"] is None:
        wf = lambda z: np.full_like(z, dm)
    else:
        zt, dt = c["Hg"] - c["zt"], c["dt"] * 1e-3
        wf = lambda z: dm - (dm - dt) * np.exp(-((z - zt) / HT) ** 2)
    n = 300 if c["Hg"] <= 100 else 500
    t0 = time.time()
    try:
        r = solve_gas_column(wf, c["Hg"], n=n, dz_top=0.1, t_max=20.0 + 0.1 * c["Hg"])
    except Exception as e:
        return c["tag"], f"FAILED {e}"
    np.savez(f, **{k: getattr(r, k) for k in ["z", "w", "d", "rho", "v", "p", "T", "M", "Ev"]},
             flux_out=r.flux_out, evap_in=r.evap_in, deposit=r.deposit, converged=r.converged, steps=r.steps,
             seal_rate=r.seal_rate(), z_sonic=r.z_sonic, Hg=c["Hg"], dm=c["dm"], zt=c["zt"] or -1, dt=c["dt"] or -1, secs=time.time() - t0)
    return c["tag"], f"out/evap {r.flux_out/max(r.evap_in,1e-30):.3f} z_sonic {r.z_sonic:.1f} conv {r.converged} {time.time()-t0:.0f}s"

for tag, msg in Parallel(n_jobs=a.workers, verbose=0, return_as="generator")(delayed(run)(c) for c in cases):
    print(f"{tag:28s} {msg}", flush=True)
print("done", flush=True)
