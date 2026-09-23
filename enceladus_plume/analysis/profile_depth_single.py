import os, sys, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'analysis')
os.environ.setdefault("ENC_LUT", "results/gas_lut_fine.npz"); os.environ.setdefault("ENC_BARRIER_DELTA", "1")
import fit_selfconsistent as F
from joblib import Parallel, delayed
Ls = [1.0, 1.55, 2.0, 3.5, 5.0, 8.0, 12.0]
dws = np.geomspace(3.0, 24.0, 33)
sig = 15.85
grid = [(dw, L) for L in Ls for dw in dws]
res = Parallel(n_jobs=63)(delayed(F.evaluate)([dw, L, 0.0, 0.0, sig]) for dw, L in grid)
np.savez("/tmp/freefit/Lprofile_single_fine.npz", grid=np.array(grid), chi2=np.array([r[0] for r in res]), weff=np.array([r[1] for r in res]))
print("single cosine, 1 m cap, fine LUT, sigma fixed 15.9; 33 log-spaced dw in 3-24 mm per L")
for L in Ls:
    rows = [(dw, c, w) for (dw, l), (c, w) in zip(grid, res) if l == L]
    dw, c, w = min(rows, key=lambda r: r[1])
    ok = sum(1 for r in rows if r[1]/21 < 3.0)
    print(f"  L={L:5.2f} km: best dw={dw:5.2f} mm chi2/dof={c/21:6.2f} w_eff*={w*1e3:.2f} mm; #dw with chi2/dof<3: {ok}/33; profile: " + " ".join(f"{r[1]/21:.0f}" for r in rows))
