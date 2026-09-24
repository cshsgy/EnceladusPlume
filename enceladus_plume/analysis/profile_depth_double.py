#!/usr/bin/env python3
"""Does the second harmonic admit a deeper source?  At fixed L, search (dw, phi2,
alpha, sigma) with the self-consistent free-lip model (1 m cap, fine LUT): parallel
Sobol round then a Nelder-Mead polish per depth (polishes run in parallel across L).
    ENC_LUT=results/gas_lut_fine.npz ENC_BARRIER_DELTA=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
      python analysis/profile_depth_double.py --Ls 3,5,8,12
"""
import argparse, os, sys, time, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))
os.environ.setdefault("ENC_LUT", "results/gas_lut_fine.npz"); os.environ.setdefault("ENC_BARRIER_DELTA", "1")
import fit_selfconsistent as F
from joblib import Parallel, delayed
from scipy.stats import qmc
from scipy.optimize import minimize

ap = argparse.ArgumentParser(); ap.add_argument("--Ls", default="3,5,8,12"); ap.add_argument("--n", type=int, default=200); ap.add_argument("--nm", type=int, default=40)
ap.add_argument("--jobs", type=int, default=60); ap.add_argument("--out", default="results/freefit_diagnostics/profile_depth_double.npz")
args = ap.parse_args()
Ls = [float(x) for x in args.Ls.split(",")]
lo = np.array([3.0, 0.0, 0.3, 0.0]); hi = np.array([60.0, 180.0, 1.5, 45.0])   # dw, phi2, alpha, sigma
def full(x, L): return [x[0], L, x[1], x[2], x[3]]
out = {}
for L in Ls:
    X = lo + qmc.Sobol(d=4, scramble=True, seed=int(L * 10)).random(args.n) * (hi - lo)
    t0 = time.time(); res = Parallel(n_jobs=args.jobs)(delayed(F.evaluate)(full(x, L)) for x in X)
    chi2 = np.array([r[0] for r in res]); k = np.argsort(chi2)
    print(f"L={L:5.1f} km: Sobol {args.n} in {(time.time()-t0)/60:.1f} min; best chi2/dof(19)={chi2[k[0]]/19:.2f} at dw={X[k[0],0]:.1f} phi2={X[k[0],1]:.0f} alpha={X[k[0],2]:.2f} sigma={X[k[0],3]:.0f}", flush=True)
    out[L] = (X, chi2)
def polish(L):
    X, chi2 = out[L]; x0 = X[np.argmin(chi2)]
    loc = minimize(lambda x: F.evaluate(full(x, L))[0], x0, method="Nelder-Mead", bounds=list(zip(lo, hi)), options=dict(xatol=1e-2, fatol=0.2, maxiter=args.nm))
    c, w = F.evaluate(full(loc.x, L)); return L, loc.x, c, w
t0 = time.time(); pol = Parallel(n_jobs=len(Ls))(delayed(polish)(L) for L in Ls)
print(f"polish done in {(time.time()-t0)/60:.1f} min")
summary = []
for L, x, c, w in pol:
    print(f"  L={L:5.1f} km: chi2/dof(19)={c/19:.2f}  dw={x[0]:.2f} mm phi2={x[1]:.0f} alpha={x[2]:.2f} sigma={x[3]:.1f}  w_eff*={w*1e3:.2f} mm", flush=True)
    summary.append([L, x[0], x[1], x[2], x[3], w, c])
np.savez(args.out, summary=np.array(summary), **{f"X_{L:g}": out[L][0] for L in Ls}, **{f"chi2_{L:g}": out[L][1] for L in Ls})
print("saved", args.out)
