"""Shared setup for the single-cosine, free-lip, 1-m-cap, fine-LUT model with the
gridded self-consistent closure width (results/weff_grid_single_d1.npz)."""
import os, sys, numpy as np
sys.path.insert(0, os.getcwd())
import run_fit as R
from enceladus_plume.gas_dynamics.lookup import GasLookupTable
from scipy.interpolate import RegularGridInterpolator

GRID = os.path.join(R._RESULTS, "weff_grid_single_d1.npz")
LUT = os.path.join(R._RESULTS, "gas_lut_fine.npz")
BOUNDS3 = [(3.0, 24.0), (1.0, 12.0), (0.0, 45.0)]      # dw_mm, L_km, sigma_phi

def setup():
    d = np.load(GRID)
    v = d["w_eff"].copy()
    if np.isnan(v).any():
        from scipy.interpolate import NearestNDInterpolator
        gi, gj = np.meshgrid(np.arange(v.shape[0]), np.arange(v.shape[1]), indexing="ij"); ok = ~np.isnan(v)
        v[~ok] = NearestNDInterpolator(np.column_stack([gi[ok], gj[ok]]), v[ok])(gi[~ok], gj[~ok])
    rgi = RegularGridInterpolator((np.log(d["dw_mm"]), np.log(d["L_km"])), np.log(v), method="linear", bounds_error=False, fill_value=None)
    def weff_of(dw_mm, L_km):
        return float(np.exp(rgi([[np.log(dw_mm), np.log(L_km)]])[0]))
    lut = GasLookupTable(LUT, clean=True)
    cfg = R._cfg(); cfg.liquid_dynamics.surface_barrier = "free"; cfg.liquid_dynamics.barrier_delta = float(d["barrier_delta_m"])
    ma_o, y_o, sig = np.loadtxt(R._DATA, delimiter=",", skiprows=1).T
    return dict(weff_of=weff_of, lut=lut, cfg=cfg, ma_o=ma_o, y_o=y_o, sig=sig)

def chi2_single(x3, st, y=None):
    """chi2 (not halved) for theta3 = (dw_mm, L_km, sigma) with observations y (default digitized)."""
    dw, L, s = [float(v) for v in x3]
    y = st["y_o"] if y is None else y
    w_o = R._weights(st["ma_o"], st["sig"])
    return 2.0 * R._neg_loglike(np.array([dw, L, 0.0, 0.0, s]), st["weff_of"], st["lut"], st["cfg"], st["ma_o"], y, w_o)
