#!/usr/bin/env python3
"""Fit the forward model to the observed diurnal plume profile.

Fits the model total mass-flux-versus-mean-anomaly curve (gas evaporation plus
buffered overflow, via ``peaks._flux_curve``) to the digitized Ingersoll et al.
(2020) slab-density profile (``data/observed_diurnal.csv``).

The crack is placed on the self-sealing attractor, so the effective width is not
free but set by the overflow seal depth ``w_eff*(dw, L)`` from
``evolve_geometry_coupled``. Free parameters:

    dw   : absolute tidal swing            [m]
    L    : effective source depth          [m]
    phi0 : global mean-anomaly phase offset [deg]
    A    : amplitude (slab density per unit model flux) -- a nuisance scale

``phi0`` is scanned and ``A`` is solved in closed form (weighted linear least
squares) at every (dw, L) grid node, so the nonlinear search is only over the
2-D (dw, L) grid; a parabolic refine and Delta-chi^2 = 1 give uncertainties.

Usage:  python run_fit.py [--lookup PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np

from enceladus_plume.config import load_config
from enceladus_plume.utils import build_width_series
from enceladus_plume.liquid_dynamics.solver import (
    liquid_dynamics, compute_overflow_rate, buffer_overflow,
)
from enceladus_plume.wall_geometry import evolve_geometry_coupled
from enceladus_plume.gas_dynamics.interpolator import generate_r_table
from enceladus_plume.gas_dynamics.lookup import GasLookupTable

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data", "observed_diurnal.csv")
TB = 272.6

# (dw, L) search grid. w_eff* (overflow seal depth) is computed per cell.
DW_GRID = np.array([10, 14, 20, 25]) * 1e-3                      # m
L_GRID = np.array([3, 5, 10, 20]) * 1e3                          # m
PHI_STEP = 2.0                                                   # deg

# Second-harmonic (double-cosine) forcing: amplitude scale and phase (deg). The
# tidal stress across the tiger stripes is not a pure single sinusoid; a 2f term
# (shifted-double-cosine forcing, MeyerZuWestram2024) is fit per cell. scale=0
# recovers the single-cosine exactly and is always included as the baseline.
HARM_SCALES = np.array([0.0, 2.0, 4.0])
HARM_PHASES = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])

# Up-weight the two observed peaks (secondary ~40 deg, main ~190 deg) so the fit
# is driven to reproduce both peaks rather than the bulk of the profile. The
# weight is 1/sigma^2 times a boost that is BOOST+1 at the peak centres and 1 away.
PEAK_CENTERS = np.array([40.0, 190.0])   # deg mean anomaly
PEAK_WIDTH = 18.0                        # deg
PEAK_BOOST = 4.0


def _weights(ma_o, sig):
    """Inverse-variance weights, up-weighted near the two observed peaks."""
    boost = np.ones_like(ma_o)
    for c in PEAK_CENTERS:
        d = (ma_o - c + 180.0) % 360.0 - 180.0
        boost += PEAK_BOOST * np.exp(-0.5 * (d / PEAK_WIDTH) ** 2)
    return boost / sig ** 2


# Continuous MLE bounds: (dw[mm], L[km], phi2[deg], 2f-scale, sigma_phi[deg]).
MLE_BOUNDS = [(6.0, 30.0), (2.0, 22.0), (0.0, 180.0), (0.0, 5.0), (0.0, 35.0)]


def _cfg():
    cfg = load_config()
    cfg.liquid_dynamics.n_periods = 2
    cfg.liquid_dynamics.max_step = 300.0
    cfg.liquid_dynamics.npts_velocity = 40   # h_max/D identical to 150 here, ~2x faster
    return cfg


def _flux_curve(cfg, L, dw, we, lookup, harm_scale=0.0, harm_phase=0.0):
    """Total mass flux (gas + buffered overflow) over the last cycle vs MA.

    Uses the shifted-double-cosine forcing; ``harm_scale=0`` recovers the
    single-cosine exactly. Mirrors make_figures._flux_curve otherwise.
    """
    D = L / 10.0
    P = cfg.physical.orbital_period
    f_evap = cfg.physical.latent_heat_fusion / (
        cfg.physical.latent_heat + cfg.physical.latent_heat_fusion)
    rho_w = cfg.physical.liquid_density
    t_in = np.arange(100, P + 1, 200.0)
    R = 1.0 + dw / we
    w_in = build_width_series(t_in, R, we, orbital_period=P,
                              forcing_model="shifted-double-cosine",
                              second_harmonic_scale=harm_scale,
                              second_harmonic_phase_deg=harm_phase)
    w, h, t, v = liquid_dynamics(w_in, t_in, L, cfg)
    _, phi, _, _ = lookup.query_vectorized(
        TB, np.clip(D - h, lookup.depth.min(), lookup.depth.max()),
        np.clip(w, lookup.delta.min(), lookup.delta.max()))
    gas = np.nan_to_num(phi * w)
    ov = np.nan_to_num(f_evap * rho_w * w * compute_overflow_rate(t, v, h, w_in, t_in, L, cfg))
    ov = buffer_overflow(t, ov, cfg.liquid_dynamics.overflow_tau)
    flux = gas + ov
    m = t >= (t[-1] - P)
    o = np.argsort(t[m] - t[m][0])
    return (t[m][o] - t[m][o][0]) / P * 360.0, flux[m][o]


# Secondary-peak prominence term: the relative strength of the secondary peak
# (its amplitude above the adjacent baseline) is more diagnostic than the
# absolute slab density, so we add an explicit, heavily-weighted penalty matching
# the model's secondary prominence to the observed one. This prevents the fit from
# washing the secondary out (e.g. via a large ensemble sigma) while still matching
# the pointwise values.
SEC_PEAK_MA = 37.5    # observed secondary-peak mean anomaly (deg)
SEC_BASE_MA = 67.5    # adjacent baseline (inter-peak dip) mean anomaly (deg)
PROM_ON = True
PROM_BOOST = 80.0     # weight of the prominence term relative to a peak-point weight


def _best_phi_A(MA_m, flux_m, ma_o, y_o, w_o):
    """Best phase offset (scan) and closed-form amplitude; returns (phi0,A,chi2).

    The objective is the peak-weighted pointwise chi^2 plus, when ``PROM_ON``, a
    secondary-peak prominence term ``W (A[m_sec - m_base] - obs_prom)^2``.
    """
    if PROM_ON:
        i_sec = int(np.argmin(np.abs(ma_o - SEC_PEAK_MA)))
        i_base = int(np.argmin(np.abs(ma_o - SEC_BASE_MA)))
        obs_prom = float(y_o[i_sec] - y_o[i_base])
        Wp = PROM_BOOST * float(w_o[i_sec])
    best = (np.nan, np.nan, np.inf)
    for phi0 in np.arange(0.0, 360.0, PHI_STEP):
        m = np.interp((ma_o - phi0) % 360.0, MA_m, flux_m, period=360.0)
        # weighted linear LSQ for A: minimize sum w (A m - o)^2
        denom = np.sum(w_o * m * m)
        if denom <= 0:
            continue
        A = np.sum(w_o * m * y_o) / denom
        if A <= 0:
            continue
        chi2 = float(np.sum(w_o * (A * m - y_o) ** 2))
        if PROM_ON:
            mod_prom = A * float(m[i_sec] - m[i_base])
            chi2 += Wp * (mod_prom - obs_prom) ** 2
        if chi2 < best[2]:
            best = (float(phi0), float(A), chi2)
    return best


def fit(lookup, cfg=None):
    cfg = cfg or _cfg()
    ma_o, y_o, sig = np.loadtxt(_DATA, delimiter=",", skiprows=1).T
    w_o = _weights(ma_o, sig)

    # harmonic combos: (scale, phase); scale=0 (single-cosine baseline) once.
    combos = [(0.0, 0.0)] + [(float(s), float(p)) for s in HARM_SCALES if s > 0
                             for p in HARM_PHASES]
    chi2 = np.full((len(DW_GRID), len(L_GRID)), np.inf)
    phi = np.full_like(chi2, np.nan)
    amp = np.full_like(chi2, np.nan)
    weff = np.full_like(chi2, np.nan)
    hsc = np.full_like(chi2, np.nan)   # best 2f amplitude scale per cell
    hph = np.full_like(chi2, np.nan)   # best 2f phase per cell
    chi2_single = np.full_like(chi2, np.inf)  # scale=0 baseline per cell
    total = len(DW_GRID) * len(L_GRID)
    t0 = time.time()
    done = 0
    for i, dw in enumerate(DW_GRID):
        for j, L in enumerate(L_GRID):
            cfg.physical.equilibrium_depth = float(L)
            # w_eff* from single-cosine attractor (swing is preserved by the
            # forcing normalization, so the seal depth is ~harmonic-independent).
            res = evolve_geometry_coupled(cfg, float(dw), n_e=7,
                                          w_eff_max=0.06, w_floor=2e-3)
            if res.overflow and np.isfinite(res.w_eff_overflow):
                we = float(res.w_eff_overflow)
                weff[i, j] = we
                for (sc, ph) in combos:
                    MA_m, flux_m = _flux_curve(cfg, float(L), float(dw), we, lookup,
                                               harm_scale=sc, harm_phase=ph)
                    o = np.argsort(MA_m)
                    p0, A, c2 = _best_phi_A(MA_m[o], flux_m[o], ma_o, y_o, w_o)
                    if sc == 0.0:
                        chi2_single[i, j] = c2
                    if c2 < chi2[i, j]:
                        chi2[i, j], phi[i, j], amp[i, j] = c2, p0, A
                        hsc[i, j], hph[i, j] = sc, ph
            done += 1
            el = time.time() - t0
            eta = el / done * (total - done)
            print(f"  [{done:2d}/{total}] dw={dw*1e3:4.0f}mm L={L/1e3:4.1f}km | "
                  f"elapsed {el/60:4.1f}m | ETA {eta/60:4.1f}m | "
                  f"min chi2={np.nanmin(chi2):.1f}", flush=True)

    bi, bj = np.unravel_index(np.argmin(chi2), chi2.shape)
    dof = len(ma_o) - 4
    result = dict(
        dw=float(DW_GRID[bi]), L=float(L_GRID[bj]), w_eff=float(weff[bi, bj]),
        phi0=float(phi[bi, bj]), A=float(amp[bi, bj]),
        harm_scale=float(hsc[bi, bj]), harm_phase=float(hph[bi, bj]),
        chi2=float(chi2[bi, bj]), dof=int(dof), chi2_red=float(chi2[bi, bj] / dof),
        chi2_single_best=float(np.nanmin(chi2_single)),
        DW_GRID=DW_GRID, L_GRID=L_GRID, chi2_grid=chi2, bi=int(bi), bj=int(bj),
    )
    # Delta chi^2 = 1 marginal ranges along each grid axis through the optimum
    def _range(vals, cs):
        m = np.isfinite(cs)
        if m.sum() < 2:
            return (np.nan, np.nan)
        lo = vals[m][cs[m] <= cs.min() + 1.0]
        return (float(lo.min()), float(lo.max()))
    result["dw_1sig"] = _range(DW_GRID, chi2[:, bj])
    result["L_1sig"] = _range(L_GRID, chi2[bi, :])
    return result


def _ensemble_smooth(MA, flux, sigma_deg):
    """Ensemble-average the flux over a Gaussian spread of forcing phase.

    The observed emission sums over ~500 km of tiger-stripe segments that differ
    in local forcing phase; averaging the single-crack curve over that spread is
    a periodic Gaussian convolution in mean anomaly (width ``sigma_deg``).
    Returns a uniform (grid, smoothed-flux). sigma_deg<=0 returns the raw curve.
    """
    g = np.linspace(0.0, 360.0, 720, endpoint=False)
    f = np.interp(g, MA, flux, period=360.0)
    if sigma_deg <= 0:
        return g, f
    dx = g[1] - g[0]
    half = int(np.ceil(4.0 * sigma_deg / dx))
    k = np.arange(-half, half + 1) * dx
    w = np.exp(-0.5 * (k / sigma_deg) ** 2); w /= w.sum()
    fs = np.convolve(np.concatenate([f, f, f]), w, mode="same")[len(f):2 * len(f)]
    return g, fs


def fit_ensemble(result, lookup, cfg=None):
    """Fit the along-strike phase-spread width to the base best-fit model.

    Holds (dw, L, w_eff, harmonic) at the single-crack best fit and scans the
    ensemble spread sigma (deg), re-optimising phi0 and A. Returns the best
    sigma, its chi2, and the single-crack (sigma=0) chi2 for comparison.
    """
    cfg = cfg or _cfg()
    ma_o, y_o, sig = np.loadtxt(_DATA, delimiter=",", skiprows=1).T
    w_o = _weights(ma_o, sig)
    dw, L, we = float(result["dw"]), float(result["L"]), float(result["w_eff"])
    hs, hp = float(result.get("harm_scale", 0.0)), float(result.get("harm_phase", 0.0))
    cfg.physical.equilibrium_depth = L
    MA_m, flux_m = _flux_curve(cfg, L, dw, we, lookup, harm_scale=hs, harm_phase=hp)
    o = np.argsort(MA_m); MA_m, flux_m = MA_m[o], flux_m[o]
    best = (0.0, np.inf, np.nan, np.nan)
    c0 = np.inf
    for sigma in np.arange(0.0, 40.1, 1.0):
        g, fs = _ensemble_smooth(MA_m, flux_m, sigma)
        p0, A, c2 = _best_phi_A(g, fs, ma_o, y_o, w_o)
        if sigma == 0.0:
            c0 = c2
        if c2 < best[1]:
            best = (float(sigma), c2, float(p0), float(A))
    dof = len(ma_o) - 5  # + sigma vs the single-crack's 4
    return dict(sigma=best[0], chi2=best[1], chi2_red=best[1] / dof, dof=int(dof),
                phi0=best[2], A=best[3], chi2_single=c0, chi2_single_red=c0 / (dof + 1),
                dw=dw, L=L, w_eff=we, harm_scale=hs, harm_phase=hp)


def plot_ensemble(result, lookup, cfg=None, out=None):
    """Fig. 10: single-crack vs ensemble-averaged fit to the observed profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = cfg or _cfg()
    e = fit_ensemble(result, lookup, cfg)
    ma_o, y_o, sig = np.loadtxt(_DATA, delimiter=",", skiprows=1).T
    dw, L, we = e["dw"], e["L"], e["w_eff"]
    cfg.physical.equilibrium_depth = L
    MA_m, flux_m = _flux_curve(cfg, L, dw, we, lookup,
                               harm_scale=e["harm_scale"], harm_phase=e["harm_phase"])
    o = np.argsort(MA_m); MA_m, flux_m = MA_m[o], flux_m[o]
    # single-crack (sigma=0): reuse its own best phi0/A
    _, fs0 = _ensemble_smooth(MA_m, flux_m, 0.0)
    from numpy import interp
    w_o = _weights(ma_o, sig)
    p00, A0, _ = _best_phi_A(MA_m, flux_m, ma_o, y_o, w_o)
    gcur = np.linspace(0, 360, 721)
    single = A0 * np.interp((gcur - p00) % 360.0, MA_m, flux_m, period=360.0)
    g, fse = _ensemble_smooth(MA_m, flux_m, e["sigma"])
    ens = e["A"] * np.interp((gcur - e["phi0"]) % 360.0, g, fse, period=360.0)
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.errorbar(ma_o, y_o, yerr=sig, fmt="o", ms=4, color="k", lw=1, capsize=2,
                label="observed (digitized, Ingersoll+ 2020)")
    ax.plot(gcur, single, "--", color="tab:blue", lw=1.5,
            label=f"single crack ($\\chi^2$/dof={e['chi2_single_red']:.2f})")
    ax.plot(gcur, ens, "-", color="tab:red", lw=1.9,
            label=(f"ensemble, $\\sigma_\\phi$={e['sigma']:.0f}$^\\circ$ "
                   f"($\\chi^2$/dof={e['chi2_red']:.2f})"))
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90))
    ax.set_xlabel("mean anomaly [deg]"); ax.set_ylabel("slab density [kg km$^{-1}$]")
    ax.set_title("Ensemble averaging over tiger-stripe segments")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    out = out or os.path.normpath(os.path.join(
        _HERE, "..", "writing", "manuscript", "Figures", "diurnal_fit_ensemble.pdf"))
    fig.tight_layout(); fig.savefig(out)
    print(f"wrote {out}")
    print(f"  ensemble sigma_phi = {e['sigma']:.0f} deg   "
          f"chi2/dof {e['chi2_single_red']:.2f} (single) -> {e['chi2_red']:.2f} (ensemble)")
    return e


def build_weff_interp(cfg, verbose=True):
    """Precompute the on-attractor seal depth w_eff*(dw, L) and interpolate it.

    w_eff* is the deterministic steady-state result of the sealing iteration
    (not a free parameter), so we sample it on a coarse (dw, L) grid once and
    interpolate; each likelihood evaluation is then a single flux solve.
    Returns weff_of(dw_mm, L_km) -> w_eff [m].
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    cache = os.path.join(tempfile.gettempdir(), "weff_grid.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        pts, vals = d["pts"], d["vals"]
        lin = LinearNDInterpolator(pts, vals); near = NearestNDInterpolator(pts, vals)

        def weff_of(dw_mm, L_km):
            v = lin(dw_mm, L_km)
            return float(v) if np.isfinite(v) else float(near(dw_mm, L_km))
        print("  w_eff grid loaded from cache", flush=True)
        return weff_of
    dws = np.array([6, 10, 14, 20, 26]) * 1e-3
    Ls = np.array([3, 5, 8, 12, 18, 22]) * 1e3
    pts, vals = [], []
    t0 = time.time()
    for dw in dws:
        for L in Ls:
            cfg.physical.equilibrium_depth = float(L)
            r = evolve_geometry_coupled(cfg, float(dw), n_e=7, w_eff_max=0.06, w_floor=2e-3)
            if r.overflow and np.isfinite(r.w_eff_overflow):
                pts.append([dw * 1e3, L / 1e3]); vals.append(r.w_eff_overflow)
        if verbose:
            print(f"  w_eff grid dw={dw*1e3:.0f}mm done | elapsed {(time.time()-t0)/60:.1f}m",
                  flush=True)
    pts, vals = np.array(pts), np.array(vals)
    np.savez(cache, pts=pts, vals=vals)
    lin = LinearNDInterpolator(pts, vals)
    near = NearestNDInterpolator(pts, vals)

    def weff_of(dw_mm, L_km):
        v = lin(dw_mm, L_km)
        if not np.isfinite(v):
            v = near(dw_mm, L_km)
        return float(v)
    return weff_of


def _neg_loglike(theta, weff_of, lookup, cfg, ma_o, y_o, w_o):
    """0.5 * weighted chi^2 (Gaussian negative log-likelihood up to a constant)."""
    dw_mm, L_km, phi2, scale, sigma = theta
    we = weff_of(dw_mm, L_km)
    if not np.isfinite(we) or we <= 0:
        return 1e12
    L = L_km * 1e3
    cfg.physical.equilibrium_depth = L
    try:
        MA, fl = _flux_curve(cfg, L, dw_mm * 1e-3, we, lookup,
                             harm_scale=max(scale, 0.0), harm_phase=phi2)
    except Exception:
        return 1e12
    o = np.argsort(MA)
    g, fs = _ensemble_smooth(MA[o], fl[o], max(sigma, 0.0))
    _, _, c2 = _best_phi_A(g, fs, ma_o, y_o, w_o)
    return 0.5 * float(c2)


def fit_mle(lookup, cfg=None, seed=0):
    """Continuous maximum-likelihood fit (global + local) over the free params."""
    from scipy.optimize import differential_evolution, minimize
    cfg = cfg or _cfg()
    ma_o, y_o, sig = np.loadtxt(_DATA, delimiter=",", skiprows=1).T
    w_o = _weights(ma_o, sig)
    weff_of = build_weff_interp(cfg)
    args = (weff_of, lookup, cfg, ma_o, y_o, w_o)

    neval = {"n": 0}
    t0 = time.time()

    def obj(th):
        neval["n"] += 1
        v = _neg_loglike(th, *args)
        if neval["n"] % 25 == 0:
            print(f"  eval {neval['n']:4d} | best 2*NLL so far tracked by DE | "
                  f"elapsed {(time.time()-t0)/60:.1f}m", flush=True)
        return v

    print("  global search (differential_evolution)...", flush=True)
    de = differential_evolution(obj, MLE_BOUNDS, maxiter=12, popsize=6, tol=1e-2,
                                rng=seed, polish=False, init="sobol")
    print(f"  DE done: 2*NLL={2*de.fun:.1f} at {np.round(de.x,2)}", flush=True)
    loc = minimize(obj, de.x, method="Nelder-Mead",
                   bounds=MLE_BOUNDS, options=dict(xatol=1e-2, fatol=1e-2, maxiter=400))
    x = loc.x if loc.fun < de.fun else de.x
    chi2 = 2.0 * _neg_loglike(x, *args)
    dof = len(ma_o) - 5
    dw_mm, L_km, phi2, scale, sigma = x
    we = weff_of(dw_mm, L_km)
    cfg.physical.equilibrium_depth = L_km * 1e3
    MA, fl = _flux_curve(cfg, L_km * 1e3, dw_mm * 1e-3, we, lookup,
                         harm_scale=scale, harm_phase=phi2)
    o = np.argsort(MA); g, fs = _ensemble_smooth(MA[o], fl[o], sigma)
    p0, A, _ = _best_phi_A(g, fs, ma_o, y_o, w_o)
    return dict(dw=dw_mm * 1e-3, L=L_km * 1e3, w_eff=we, harm_scale=float(scale),
                harm_phase=float(phi2), sigma=float(sigma), phi0=float(p0), A=float(A),
                chi2=float(chi2), dof=int(dof), chi2_red=float(chi2 / dof))


# ---------------------------------------------------------------------------
# Posterior (MCMC) via a fast flux emulator
# ---------------------------------------------------------------------------
_EMU_DW = np.array([7, 10, 14, 20, 27]) * 1e-3
_EMU_L = np.array([4, 6, 9, 14, 20]) * 1e3
_EMU_PHI2 = np.array([0.0, 45.0, 90.0, 135.0])
_EMU_SCALE = np.array([0.0, 1.5, 3.0, 4.5])
_EMU_MA = np.arange(0.0, 360.0, 1.0)


def build_flux_emulator(lookup, cfg=None, cache=None):
    """Precompute flux(MA; dw, L, phi2, scale) on a 4-D grid and interpolate it.

    Returns a RegularGridInterpolator that maps (dw_mm, L_km, phi2, scale) to a
    flux curve on ``_EMU_MA`` -- so each MCMC likelihood is a fast interpolation
    rather than a liquid solve. Cached to ``cache`` (npz).
    """
    from scipy.interpolate import RegularGridInterpolator
    cfg = cfg or _cfg()
    if cache and os.path.exists(cache):
        arr = np.load(cache)["flux"]
    else:
        # w_eff* per (dw, L) node (nearest-valid fallback if a node has no overflow)
        we_grid = np.full((len(_EMU_DW), len(_EMU_L)), np.nan)
        for i, dw in enumerate(_EMU_DW):
            for j, L in enumerate(_EMU_L):
                cfg.physical.equilibrium_depth = float(L)
                r = evolve_geometry_coupled(cfg, float(dw), n_e=7, w_eff_max=0.06, w_floor=2e-3)
                if r.overflow and np.isfinite(r.w_eff_overflow):
                    we_grid[i, j] = r.w_eff_overflow
            print(f"  emu w_eff dw={dw*1e3:.0f}mm done", flush=True)
        # fill NaN with row/column nearest
        for i in range(len(_EMU_DW)):
            row = we_grid[i]
            if np.all(np.isnan(row)):
                continue
            m = np.isnan(row)
            row[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), row[~m])
        arr = np.zeros((len(_EMU_DW), len(_EMU_L), len(_EMU_PHI2), len(_EMU_SCALE), len(_EMU_MA)))
        n = 0
        for i, dw in enumerate(_EMU_DW):
            for j, L in enumerate(_EMU_L):
                we = float(we_grid[i, j])
                cfg.physical.equilibrium_depth = float(L)
                for k, phi2 in enumerate(_EMU_PHI2):
                    for l, sc in enumerate(_EMU_SCALE):
                        MA, fl = _flux_curve(cfg, float(L), float(dw), we, lookup,
                                             harm_scale=float(sc), harm_phase=float(phi2))
                        o = np.argsort(MA)
                        arr[i, j, k, l] = np.interp(_EMU_MA, MA[o], fl[o], period=360.0)
                        n += 1
                print(f"  emu flux {n}/{arr[...,0].size} (dw={dw*1e3:.0f}mm L={L/1e3:.0f}km)", flush=True)
        if cache:
            np.savez(cache, flux=arr)
    rgi = RegularGridInterpolator((_EMU_DW * 1e3, _EMU_L / 1e3, _EMU_PHI2, _EMU_SCALE), arr,
                                  bounds_error=False, fill_value=None)
    return rgi


def make_log_prob(rgi, ma_o, y_o, w_o):
    """Flat-prior Gaussian log-posterior over (dw_mm, L_km, phi2, scale, sigma)."""
    lo = np.array([b[0] for b in MLE_BOUNDS]); hi = np.array([b[1] for b in MLE_BOUNDS])

    def log_prob(theta):
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        dw_mm, L_km, phi2, scale, sigma = theta
        flux = rgi([[dw_mm, L_km, phi2, scale]])[0]
        g, fs = _ensemble_smooth(_EMU_MA, flux, sigma)
        _, _, c2 = _best_phi_A(g, fs, ma_o, y_o, w_o)
        return -0.5 * c2
    return log_prob


def run_mcmc(log_prob, p0, nsteps, seed=0):
    """Affine-invariant (stretch-move) ensemble sampler; p0 is (nwalkers, ndim)."""
    rng = np.random.default_rng(seed)
    pos = np.array(p0, float); nw, nd = pos.shape
    lp = np.array([log_prob(p) for p in pos])
    chain = np.zeros((nsteps, nw, nd)); a = 2.0
    for s in range(nsteps):
        for k in range(nw):
            j = rng.integers(nw - 1); j += 1 if j >= k else 0
            z = ((a - 1.0) * rng.random() + 1.0) ** 2 / a
            prop = pos[j] + z * (pos[k] - pos[j])
            lpp = log_prob(prop)
            if np.log(rng.random()) < (nd - 1) * np.log(z) + lpp - lp[k]:
                pos[k], lp[k] = prop, lpp
        chain[s] = pos
        if (s + 1) % 200 == 0:
            print(f"  mcmc step {s+1}/{nsteps}  mean logP={lp.mean():.1f}", flush=True)
    return chain


def plot_corner(samples, labels, truths=None, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nd = samples.shape[1]
    fig, ax = plt.subplots(nd, nd, figsize=(1.9 * nd, 1.9 * nd))
    q = np.percentile(samples, [16, 50, 84], axis=0)
    for i in range(nd):
        for j in range(nd):
            a = ax[i, j]
            if j > i:
                a.axis("off"); continue
            if i == j:
                a.hist(samples[:, i], bins=30, color="tab:blue", histtype="stepfilled", alpha=0.6)
                a.axvline(q[1, i], color="k", lw=1); a.axvline(q[0, i], color="k", ls="--", lw=0.8)
                a.axvline(q[2, i], color="k", ls="--", lw=0.8)
                a.set_title(f"{labels[i]}\n${q[1,i]:.1f}^{{+{q[2,i]-q[1,i]:.1f}}}_{{-{q[1,i]-q[0,i]:.1f}}}$",
                            fontsize=8)
            else:
                a.hist2d(samples[:, j], samples[:, i], bins=30, cmap="Blues")
                if truths is not None:
                    a.plot(truths[j], truths[i], "rx", ms=6)
            if i == nd - 1:
                a.set_xlabel(labels[j], fontsize=8)
            else:
                a.set_xticklabels([])
            if j == 0 and i > 0:
                a.set_ylabel(labels[i], fontsize=8)
            else:
                a.set_yticklabels([])
            a.tick_params(labelsize=6)
    out = out or os.path.normpath(os.path.join(
        _HERE, "..", "writing", "manuscript", "Figures", "diurnal_fit_posterior.pdf"))
    fig.tight_layout(); fig.savefig(out); print(f"wrote {out}")


def fit_mcmc(lookup, cfg=None, nsteps=1500, nwalkers=32, seed=0, emu_cache=None):
    """Posterior over the fit parameters via emulator + ensemble MCMC."""
    cfg = cfg or _cfg()
    ma_o, y_o, sig = np.loadtxt(_DATA, delimiter=",", skiprows=1).T
    w_o = _weights(ma_o, sig)
    emu_cache = emu_cache or os.path.join(tempfile.gettempdir(), "flux_emu.npz")
    rgi = build_flux_emulator(lookup, cfg, cache=emu_cache)
    log_prob = make_log_prob(rgi, ma_o, y_o, w_o)
    # start walkers in a small ball around the MLE
    mle = np.array([9.3, 6.9, 82.0, 3.48, 22.0])
    lo = np.array([b[0] for b in MLE_BOUNDS]); hi = np.array([b[1] for b in MLE_BOUNDS])
    rng = np.random.default_rng(seed)
    p0 = np.clip(mle + rng.normal(0, [1.5, 1.5, 15, 0.4, 3], size=(nwalkers, 5)), lo + 1e-6, hi - 1e-6)
    print(f"  emulator ready; running MCMC ({nwalkers} walkers x {nsteps} steps)...", flush=True)
    chain = run_mcmc(log_prob, p0, nsteps)
    burn = nsteps // 3
    samples = chain[burn:].reshape(-1, 5)
    q = np.percentile(samples, [16, 50, 84], axis=0)
    return dict(samples=samples, med=q[1], lo=q[1] - q[0], up=q[2] - q[1],
                labels=["dw [mm]", "L [km]", "phi2 [deg]", "2f scale", "sigma [deg]"])


def plot_overlay(result, lookup, cfg=None, out=None):
    """Data-vs-model overlay for the best fit (recomputes one flux curve)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = cfg or _cfg()
    ma_o, y_o, sig = np.loadtxt(_DATA, delimiter=",", skiprows=1).T
    dw, L, we = float(result["dw"]), float(result["L"]), float(result["w_eff"])
    phi0, A = float(result["phi0"]), float(result["A"])
    hs, hp = float(result.get("harm_scale", 0.0)), float(result.get("harm_phase", 0.0))
    sigma = float(result.get("sigma", 0.0))
    cfg.physical.equilibrium_depth = L
    MA_m, flux_m = _flux_curve(cfg, L, dw, we, lookup, harm_scale=hs, harm_phase=hp)
    o = np.argsort(MA_m)
    gm, fsm = _ensemble_smooth(MA_m[o], flux_m[o], sigma)
    grid = np.linspace(0, 360, 721)
    model = A * np.interp((grid - phi0) % 360.0, gm, fsm, period=360.0)
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.errorbar(ma_o, y_o, yerr=sig, fmt="o", ms=4, color="k", lw=1,
                capsize=2, label="observed (digitized, Ingersoll+ 2020)")
    slab = (f"best fit: $\\Delta w$={dw*1e3:.0f} mm, $L$={L/1e3:.0f} km, "
            f"2f={hs:.1f}@{hp:.0f}$^\\circ$"
            + (f", $\\sigma_\\phi$={sigma:.0f}$^\\circ$" if sigma > 0 else ""))
    ax.plot(grid, model, "-", color="tab:red", lw=1.8, label=slab)
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90))
    ax.set_xlabel("mean anomaly [deg]"); ax.set_ylabel("slab density [kg km$^{-1}$]")
    ax.set_title("Diurnal profile: model fit to observed emission")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    out = out or os.path.normpath(os.path.join(
        _HERE, "..", "writing", "manuscript", "Figures", "diurnal_fit.pdf"))
    fig.tight_layout(); fig.savefig(out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookup", default=os.path.join(tempfile.gettempdir(), "encfig_lut.npz"))
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "diurnal_fit.npz"))
    ap.add_argument("--plot-only", action="store_true",
                    help="skip fitting; load --out and just (re)draw the overlay")
    ap.add_argument("--ensemble", action="store_true",
                    help="skip fitting; load --out, fit the ensemble spread, draw Fig. 10")
    ap.add_argument("--mle", action="store_true",
                    help="continuous max-likelihood fit (global DE + local refine)")
    ap.add_argument("--mcmc", action="store_true",
                    help="posterior via emulator + ensemble MCMC; writes corner plot")
    args = ap.parse_args()
    if args.ensemble:
        r = dict(np.load(args.out))
        plot_ensemble(r, GasLookupTable(args.lookup, clean=True))
        return
    if args.plot_only:
        r = dict(np.load(args.out))
        plot_overlay(r, GasLookupTable(args.lookup, clean=True))
        return
    if not os.path.exists(args.lookup):
        generate_r_table(np.geomspace(1e-3, 0.08, 24), np.geomspace(0.5, 3000.0, 24),
                         Tb_arr=np.array([272.0, 273.1501]), output_path=args.lookup, n_jobs=-1)
    lut = GasLookupTable(args.lookup, clean=True)
    if args.mcmc:
        r = fit_mcmc(lut)
        plot_corner(r["samples"], r["labels"], truths=[13.7, 19.4, 105.0, 0.59, 27.0])
        print("\n=== POSTERIOR (median +/- 68% credible) ===")
        for lab, m, dn, up in zip(r["labels"], r["med"], r["lo"], r["up"]):
            print(f"  {lab:14}= {m:7.2f}  (+{up:.2f} / -{dn:.2f})")
        # dn is now the lower *error* (median - 16th pct)
        np.savez(os.path.join(tempfile.gettempdir(), "diurnal_posterior.npz"),
                 samples=r["samples"], med=r["med"], lo=r["lo"], up=r["up"])
        return
    if args.mle:
        r = fit_mle(lut)
        np.savez(args.out, **{k: v for k, v in r.items()})
        print("\n=== MLE FIT (continuous) ===")
        print(f"  dw    = {r['dw']*1e3:.1f} mm")
        print(f"  L     = {r['L']/1e3:.1f} km")
        print(f"  w_eff*= {r['w_eff']*1e3:.2f} mm (on attractor, interpolated)")
        print(f"  2f: scale={r['harm_scale']:.2f}  phase={r['harm_phase']:.0f} deg")
        print(f"  sigma_phi (ensemble) = {r['sigma']:.1f} deg")
        print(f"  phi0  = {r['phi0']:.0f} deg   A = {r['A']:.3e}")
        print(f"  chi2/dof = {r['chi2']:.1f}/{r['dof']} = {r['chi2_red']:.2f}")
        print(f"  saved -> {args.out}")
        plot_overlay(r, lut, out=os.path.normpath(os.path.join(
            _HERE, "..", "writing", "manuscript", "Figures", "diurnal_fit_mle.pdf")))
        return
    r = fit(lut)
    np.savez(args.out, **{k: v for k, v in r.items()})
    print("\n=== BEST FIT (on-attractor) ===")
    print(f"  dw   = {r['dw']*1e3:.1f} mm   (Delta-chi2=1: "
          f"{r['dw_1sig'][0]*1e3:.0f}-{r['dw_1sig'][1]*1e3:.0f} mm)")
    print(f"  L    = {r['L']/1e3:.1f} km   (Delta-chi2=1: "
          f"{r['L_1sig'][0]/1e3:.1f}-{r['L_1sig'][1]/1e3:.1f} km)")
    print(f"  w_eff*= {r['w_eff']*1e3:.2f} mm (on attractor)")
    print(f"  phi0 = {r['phi0']:.0f} deg")
    print(f"  A    = {r['A']:.3e} (slab per model-flux unit)")
    print(f"  2f harmonic scale={r['harm_scale']:g} phase={r['harm_phase']:g} deg")
    print(f"  chi2/dof = {r['chi2']:.1f}/{r['dof']} = {r['chi2_red']:.2f}  "
          f"(single-cosine best chi2/dof = {r['chi2_single_best']/r['dof']:.2f})")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
