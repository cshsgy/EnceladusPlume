import os, sys, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'analysis')
os.environ.setdefault("ENC_LUT", "results/gas_lut_fine.npz"); os.environ.setdefault("ENC_BARRIER_DELTA", "1")
import fit_selfconsistent as F
from joblib import Parallel, delayed
# refine (dw, sigma) jointly at three depths around the coarse-profile minima
cases = {1.0: np.linspace(3.2, 4.4, 13), 3.5: np.linspace(6.6, 8.6, 13), 8.0: np.linspace(10.5, 13.5, 13)}
sigs = [0.0, 5.0, 10.0, 16.0, 22.0, 30.0, 40.0]
grid = [(dw, L, s) for L, dws in cases.items() for dw in dws for s in sigs]
res = Parallel(n_jobs=63)(delayed(F.evaluate)([dw, L, 0.0, 0.0, s]) for dw, L, s in grid)
np.savez("/tmp/freefit/Lprofile_sigma.npz", grid=np.array(grid), chi2=np.array([r[0] for r in res]), weff=np.array([r[1] for r in res]))
print("single cosine, 1 m cap, fine LUT: joint (dw, sigma) refinement at fixed L (chi2/dof, 21 dof)")
for L in cases:
    rows = [(dw, s, c, w) for (dw, l, s), (c, w) in zip(grid, res) if l == L]
    dw, s, c, w = min(rows, key=lambda r: r[2])
    print(f"  L={L:4.1f} km: best chi2/dof={c/21:5.2f} at dw={dw:.2f} mm sigma={s:.0f} (w_eff*={w*1e3:.2f} mm)")
    for s in sigs:
        print(f"     sigma={s:4.0f}: " + " ".join(f"{r[2]/21:4.1f}" for r in rows if r[1] == s))
