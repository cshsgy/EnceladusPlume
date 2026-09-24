#!/usr/bin/env python3
"""Posterior of the single-cosine free-lip model (1 m cap, fine LUT, gridded
self-consistent closure width): affine-invariant ensemble MCMC over (dw, L, sigma_phi)
against the exact forward model, parallel over walkers.
    OMP_NUM_THREADS=1 PYTHONPATH=. python analysis/run_mcmc_single_d1.py [--nsteps 800 --nw 64 --pool 60]
"""
import argparse, os, sys, time, numpy as np, multiprocessing as mp
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))
import single_d1_common as C
_ST = None; _LO = np.array([b[0] for b in C.BOUNDS3]); _HI = np.array([b[1] for b in C.BOUNDS3])
def _init():
    global _ST; _ST = C.setup()
def logprob(th):
    th = np.asarray(th, float)
    if np.any(th < _LO) or np.any(th > _HI): return -np.inf
    return -0.5 * C.chi2_single(th, _ST)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nsteps", type=int, default=800); ap.add_argument("--nw", type=int, default=64); ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--out", default="results/freefit_diagnostics/posterior_single_d1.npz"); args = ap.parse_args()
    import run_fit as R
    fit = R.load_result(os.path.join(R._RESULTS, "diurnal_fit_free_single_sc_fine_d1.json"))
    mle = np.array([float(fit["dw"]) * 1e3, float(fit["L"]) / 1e3, float(fit["sigma"])])
    nd, nw, nsteps, a = 3, args.nw, args.nsteps, 2.0
    rng = np.random.default_rng(0)
    pos = np.clip(mle * (1 + rng.normal(0, [0.03, 0.03, 0.15], size=(nw, nd))), _LO + 1e-6, _HI - 1e-6)
    pool = mp.Pool(args.pool, initializer=_init)
    lp = np.array(pool.map(logprob, list(pos))); print("init logP: best", lp.max(), "median", np.median(lp), flush=True)
    half = nw // 2; idx = np.arange(nw); chain = np.zeros((nsteps, nw, nd)); lpc = np.zeros((nsteps, nw)); t0 = time.time(); nacc = 0
    for s in range(nsteps):
        for k in (0, 1):
            S = idx[k * half:(k + 1) * half]; Cc = idx[(1 - k) * half:(2 - k) * half]
            props = []; zs = np.empty(half)
            for m, i in enumerate(S):
                j = Cc[rng.integers(half)]; z = ((a - 1.0) * rng.random() + 1.0) ** 2 / a
                props.append(pos[j] + z * (pos[i] - pos[j])); zs[m] = z
            lpp = np.array(pool.map(logprob, props))
            acc = np.log(rng.random(half)) < (nd - 1) * np.log(zs) + lpp - lp[S]
            for m, i in enumerate(S):
                if acc[m]: pos[i] = props[m]; lp[i] = lpp[m]; nacc += 1
        chain[s] = pos; lpc[s] = lp
        if (s + 1) % 25 == 0:
            el = (time.time() - t0) / 60
            np.savez(args.out, chain=chain[:s + 1], logp=lpc[:s + 1], labels=["dw [mm]", "L [km]", "sigma_phi [deg]"], mle=mle)
            print(f"step {s+1}/{nsteps} meanlogP={lp.mean():.1f} bestlogP={lp.max():.1f} acc={nacc/((s+1)*nw):.2f} {el:.1f}min ETA {el/(s+1)*(nsteps-s-1):.0f}min", flush=True)
    pool.close(); pool.join()
    samples = chain[nsteps // 3:].reshape(-1, nd); q = np.percentile(samples, [16, 50, 84], axis=0)
    np.savez(args.out, chain=chain, logp=lpc, samples=samples, med=q[1], lo=q[1] - q[0], up=q[2] - q[1], labels=["dw [mm]", "L [km]", "sigma_phi [deg]"], mle=mle)
    print("MCMC DONE  median:", np.round(q[1], 3), " -", np.round(q[1] - q[0], 3), " +", np.round(q[2] - q[1], 3), flush=True)
if __name__ == "__main__":
    main()
