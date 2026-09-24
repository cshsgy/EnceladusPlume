#!/usr/bin/env python3
"""Text S4 redo: digitization sensitivity of the single-cosine free-lip fit (1 m cap,
fine LUT, gridded closure width). Perturb the digitized points within their per-bin
scatter (N realizations), refit (dw, L, sigma) by Nelder-Mead from the adopted fit.
    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/sensitivity_single_d1.py [--N 30 --pool 30]
"""
import argparse, os, sys, time, numpy as np, multiprocessing as mp
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))
import single_d1_common as C
from scipy.optimize import minimize
_ST = None
def _init():
    global _ST; _ST = C.setup()
def _one(task):
    i, yp, x0 = task
    f = lambda th: C.chi2_single(th, _ST, y=yp)
    loc = minimize(f, x0, method="Nelder-Mead", bounds=C.BOUNDS3, options=dict(xatol=1e-2, fatol=0.2, maxiter=60))
    return i, [*loc.x, float(loc.fun)]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, default=30); ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--out", default="results/freefit_diagnostics/sensitivity_single_d1.npz"); args = ap.parse_args()
    import run_fit as R
    fit = R.load_result(os.path.join(R._RESULTS, "diurnal_fit_free_single_sc_fine_d1.json"))
    x0 = [float(fit["dw"]) * 1e3, float(fit["L"]) / 1e3, float(fit["sigma"])]
    ma, y0, sig = np.loadtxt(R._DATA, delimiter=",", skiprows=1).T
    rng = np.random.default_rng(0)
    tasks = [(i, np.clip(y0 + rng.normal(0, sig), 1.0, None), x0) for i in range(args.N)]
    t0 = time.time()
    with mp.Pool(args.pool, initializer=_init) as pool:
        out = pool.map(_one, tasks)
    res = np.array([r for _, r in sorted(out)])
    np.savez(args.out, res=res, labels=["dw_mm", "L_km", "sigma", "chi2"], x0=x0)
    dof = len(ma) - 3
    print(f"N={args.N} realizations, {(time.time()-t0)/60:.1f} min; start {np.round(x0,2)}")
    for j, l in enumerate(["dw [mm]", "L [km]", "sigma [deg]"]):
        q = np.percentile(res[:, j], [16, 50, 84]); print(f"  {l:12s} median {q[1]:.2f}  (+{q[2]-q[1]:.2f} / -{q[1]-q[0]:.2f})  range {res[:,j].min():.2f}-{res[:,j].max():.2f}")
    print(f"  chi2/dof median {np.median(res[:,3])/dof:.2f}, range {res[:,3].min()/dof:.2f}-{res[:,3].max()/dof:.2f}")
    print("SENSITIVITY DONE", flush=True)
