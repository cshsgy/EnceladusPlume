#!/usr/bin/env python3
"""Self-consistent free-lip fit: the closure width w_eff* is recomputed by bisection
for the *fitted forcing* (alpha, phi2) at every likelihood evaluation, so the crack
is genuinely on the attractor (water first reaching the surface cap) for the
forcing that drives it. Each evaluation costs ~10 liquid solves (~1 min), so the
global search is a parallel successive-shrinking Sobol search (joblib), followed by
a short serial Nelder-Mead polish.

    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/fit_selfconsistent.py [--single] [--jobs 60]
Writes results/diurnal_fit_free_sc.json (or _single_sc.json) and a log of all evaluations.
"""
import argparse, copy, json, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
import run_fit as R
from scipy.stats import qmc

BOUNDS = np.array(R.MLE_BOUNDS, dtype=float)          # dw_mm, L_km, phi2, alpha, sigma
_STATE = {}


def _setup():
    if "lut" not in _STATE:
        from enceladus_plume.gas_dynamics.lookup import GasLookupTable
        _STATE["lut"] = GasLookupTable(R.DEFAULT_LOOKUP, clean=True)
        _STATE["obs"] = np.loadtxt(R._DATA, delimiter=",", skiprows=1).T
        _STATE["weff0"] = R.build_weff_interp(R._cfg(), verbose=False)   # single-cosine grid: warm start
    return _STATE


def closure_width_sc(cfg, dw, alpha, phi2, we0, tol=2e-5):
    """Bisection closure width for the double-cosine forcing, bracketed around we0."""
    from enceladus_plume.wall_geometry import closure_width
    fp = dict(second_harmonic_scale=float(alpha), second_harmonic_phase_deg=float(phi2))
    fm = "shifted-double-cosine" if alpha > 0 else "single-cosine"
    lo, hi = 0.5 * we0, 2.0 * we0
    for _ in range(3):
        we, ok = closure_width(cfg, dw, forcing_model=fm, forcing_params=fp if alpha > 0 else None,
                               w_lo=max(lo, 1e-3), w_hi=min(hi, 0.08), tol=tol)
        if not ok:
            return float("nan")
        if we <= max(lo, 1e-3) * 1.01:      # hit the lower bracket: widen downward
            lo *= 0.5
        elif we >= min(hi, 0.08) * 0.99:    # hit the upper bracket: widen upward
            hi *= 2.0
        else:
            return we
    return we


def evaluate(theta, mode="free"):
    """Return (chi2, w_eff) for theta = (dw_mm, L_km, phi2, alpha, sigma)."""
    st = _setup()
    dw_mm, L_km, phi2, alpha, sigma = [float(x) for x in theta]
    cfg = R._cfg(); cfg.liquid_dynamics.surface_barrier = mode
    cfg.physical.equilibrium_depth = L_km * 1e3
    we0 = st["weff0"](dw_mm, L_km)
    try:
        we = closure_width_sc(cfg, dw_mm * 1e-3, alpha, phi2, we0)
    except Exception as e:  # pragma: no cover
        return 1e12, float("nan")
    if not np.isfinite(we):
        return 1e12, float("nan")
    ma_o, y_o, sig = st["obs"]; w_o = R._weights(ma_o, sig)
    nll = R._neg_loglike(np.array([dw_mm, L_km, phi2, alpha, sigma]), lambda a, b: we,
                         st["lut"], cfg, ma_o, y_o, w_o)
    return 2.0 * float(nll), float(we)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--n1", type=int, default=300); ap.add_argument("--n2", type=int, default=200); ap.add_argument("--n3", type=int, default=150)
    ap.add_argument("--nm", type=int, default=60, help="Nelder-Mead iterations for the final serial polish")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from joblib import Parallel, delayed
    tag = "diurnal_fit_free_single_sc" if args.single else "diurnal_fit_free_sc"
    out = args.out or os.path.join(R._RESULTS, tag + ".json")
    logf = open(f"/tmp/freefit/{tag}_evals.csv", "a")
    single = args.single
    ndim = 3 if single else 5
    idx = [0, 1, 4] if single else [0, 1, 2, 3, 4]
    lo, hi = BOUNDS[idx, 0].copy(), BOUNDS[idx, 1].copy()

    def full(x):
        th = np.zeros(5); th[idx] = x; return th

    def run_round(X, label):
        t0 = time.time()
        res = Parallel(n_jobs=args.jobs)(delayed(evaluate)(full(x)) for x in X)
        chi2 = np.array([r[0] for r in res]); we = np.array([r[1] for r in res])
        for x, c, w in zip(X, chi2, we):
            logf.write(",".join(f"{v:.6g}" for v in [*full(x), w * 1e3 if np.isfinite(w) else -1, c]) + "\n")
        logf.flush()
        k = np.argsort(chi2)[:10]
        print(f"  {label}: {len(X)} evals in {(time.time()-t0)/60:.1f} min; best chi2={chi2[k[0]]:.1f} at {np.round(full(X[k[0]]),2)} w_eff={we[k[0]]*1e3:.2f} mm", flush=True)
        return X, chi2, we, k

    def sobol(n, lo, hi, seed):
        s = qmc.Sobol(d=ndim, scramble=True, seed=seed).random(n)
        return lo + s * (hi - lo)

    print(f"self-consistent free-lip fit ({'single' if single else 'double'} cosine), bounds lo={lo} hi={hi}", flush=True)
    X, c, w, k = run_round(sobol(args.n1, lo, hi, 1), "round 1 (global)")
    for rnd, n in [(2, args.n2), (3, args.n3)]:
        top = X[k[:10 if rnd == 2 else 5]]
        blo, bhi = top.min(0), top.max(0); span = np.maximum(bhi - blo, 0.05 * (hi - lo))
        blo = np.maximum(lo, blo - 0.25 * span); bhi = np.minimum(hi, bhi + 0.25 * span)
        X2, c2, w2, _ = run_round(sobol(n, blo, bhi, rnd), f"round {rnd} (box {np.round(blo,2)}..{np.round(bhi,2)})")
        X = np.vstack([X, X2]); c = np.concatenate([c, c2]); w = np.concatenate([w, w2]); k = np.argsort(c)[:10]
    best = X[k[0]]
    # serial Nelder-Mead polish
    from scipy.optimize import minimize
    t0 = time.time()
    loc = minimize(lambda x: evaluate(full(x))[0], best, method="Nelder-Mead", bounds=list(zip(lo, hi)),
                   options=dict(xatol=1e-2, fatol=0.2, maxiter=args.nm))
    xb = loc.x if loc.fun < c[k[0]] else best
    chi2, we = evaluate(full(xb))
    print(f"  NM polish [{(time.time()-t0)/60:.1f} min]: chi2 {c[k[0]]:.1f} -> {chi2:.1f}", flush=True)
    th = full(xb); dw_mm, L_km, phi2, alpha, sigma = th
    st = _setup(); ma_o, y_o, sig = st["obs"]; w_o = R._weights(ma_o, sig)
    cfg = R._cfg(); cfg.liquid_dynamics.surface_barrier = "free"; cfg.physical.equilibrium_depth = L_km * 1e3
    MA, fl = R._flux_curve(cfg, L_km * 1e3, dw_mm * 1e-3, we, st["lut"], harm_scale=alpha, harm_phase=phi2)
    o = np.argsort(MA); g, fs = R._ensemble_smooth(MA[o], fl[o], sigma); p0, A, _ = R._best_phi_A(g, fs, ma_o, y_o, w_o)
    dof = len(ma_o) - ndim
    res = dict(dw=dw_mm * 1e-3, L=L_km * 1e3, w_eff=we, harm_scale=float(alpha), harm_phase=float(phi2), sigma=float(sigma),
               phi0=float(p0), A=float(A), chi2=float(chi2), dof=int(dof), chi2_red=float(chi2 / dof),
               barrier="free", closure="self-consistent bisection with fitted forcing")
    R.save_result(out, res)
    print("=== SELF-CONSISTENT FIT ===")
    for kk, v in res.items(): print(f"  {kk} = {v}")
    print(f"  saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
