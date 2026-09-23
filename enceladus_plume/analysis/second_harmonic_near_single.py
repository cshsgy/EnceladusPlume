import os, sys, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'analysis')
os.environ.setdefault("ENC_LUT", "results/gas_lut_fine.npz"); os.environ.setdefault("ENC_BARRIER_DELTA", "1")
import fit_selfconsistent as F, run_fit as R
from joblib import Parallel, delayed
from scipy.stats import qmc
from scipy.optimize import minimize
s0 = R.load_result("results/diurnal_fit_free_single_sc_fine_d1.json")
x0 = np.array([float(s0["dw"])*1e3, float(s0["L"])/1e3, 0.0, 0.0, float(s0["sigma"])])
c0, w0 = F.evaluate(x0); print(f"single solution re-evaluated: chi2={c0:.1f} (/19 -> {c0/19:.2f})", flush=True)
lo = np.array([x0[0]*0.9, x0[1]*0.9, 0.0, 0.0, 5.0]); hi = np.array([x0[0]*1.1, x0[1]*1.1, 180.0, 0.6, 30.0])
X = lo + qmc.Sobol(d=5, scramble=True, seed=7).random(240) * (hi - lo)
res = Parallel(n_jobs=60)(delayed(F.evaluate)(x) for x in X)
chi2 = np.array([r[0] for r in res]); k = np.argsort(chi2)
print("box search around the single solution with alpha in [0,0.6]: best 5:")
for i in k[:5]: print(f"   chi2={chi2[i]:.1f} (/19={chi2[i]/19:.2f}) at {np.round(X[i],2)} w_eff={res[i][1]*1e3:.2f}")
start = X[k[0]] if chi2[k[0]] < c0 else x0
loc = minimize(lambda x: F.evaluate(x)[0], start, method="Nelder-Mead", bounds=list(zip(lo, hi)), options=dict(xatol=1e-2, fatol=0.2, maxiter=50))
c, w = F.evaluate(loc.x)
print(f"NM from best: chi2={c:.1f} (/19={c/19:.2f}) at {np.round(loc.x,3)} w_eff={w*1e3:.2f} mm")
np.savez("/tmp/freefit/double_from_single.npz", X=X, chi2=chi2, xbest=loc.x, cbest=c)
