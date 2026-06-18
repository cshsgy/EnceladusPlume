#!/usr/bin/env python3
"""Unified liquid + gas pipeline with parameter sweep.

Usage:
    python run_pipeline.py --config config/run_config.yaml [-v]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from enceladus_plume.config import load_config, Config
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics, compute_overflow_rate
from enceladus_plume.gas_dynamics.interpolator import build_lookup_from_bounds
from enceladus_plume.gas_dynamics.lookup import GasLookupTable
from enceladus_plume.utils import build_width_series

logger = logging.getLogger(__name__)


def _run_liquid_case(
    wmin: float, wmaxmin: float, cfg: Config, forcing_model: str, forcing_params: dict,
) -> dict:
    """Run liquid dynamics for one (wmin, wmaxmin) and return results dict."""
    P = cfg.physical.orbital_period
    L = cfg.physical.equilibrium_depth
    t_in = np.arange(100, P + 1, 200.0)
    w_in = build_width_series(
        t_in,
        wmaxmin,
        wmin,
        orbital_period=P,
        forcing_model=forcing_model,
        **forcing_params,
    )

    w_rec, h_rec, t_rec, v_rec = liquid_dynamics(w_in, t_in, L, cfg)

    D = L / 10.0
    depth_rec = D - h_rec

    return dict(
        wmin=wmin, wmaxmin=wmaxmin,
        t_rec=t_rec, h_rec=h_rec, w_rec=w_rec, v_rec=v_rec,
        depth_rec=depth_rec, w_in=w_in, t_in=t_in,
    )


def main():
    parser = argparse.ArgumentParser(description="Liquid + gas pipeline")
    parser.add_argument("--config", required=True, help="Path to run_config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        run_cfg = yaml.safe_load(f)

    base_path = run_cfg.get("base_config")
    cfg = load_config(base_path)
    forcing_cfg = dict(run_cfg.get("forcing", {}))
    forcing_model = forcing_cfg.pop("model", "single-cosine")

    liq_overrides = run_cfg.get("liquid_dynamics", {})
    for key, val in liq_overrides.items():
        if hasattr(cfg.liquid_dynamics, key):
            setattr(cfg.liquid_dynamics, key, val)

    out_dir = Path(run_cfg.get("output", {}).get("directory", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    wmin_list = run_cfg["sweep"]["wmin"]
    wmaxmin_list = run_cfg["sweep"]["wmaxmin"]
    cases = list(itertools.product(wmin_list, wmaxmin_list))
    logger.info(
        "Sweep: %d cases (%d wmin x %d wmaxmin), forcing=%s",
        len(cases),
        len(wmin_list),
        len(wmaxmin_list),
        forcing_model,
    )

    # ------------------------------------------------------------------
    # Step 1: liquid dynamics for all cases
    # ------------------------------------------------------------------
    import pickle
    liquid_cache = out_dir / "liquid_cache.pkl"
    skip_liquid = run_cfg.get("skip_liquid", False)

    if skip_liquid and liquid_cache.exists():
        logger.info("=== Step 1: Loading cached liquid dynamics ===")
        with open(liquid_cache, "rb") as fh:
            liquid_results = pickle.load(fh)
        logger.info("  Loaded %d cached results from %s", len(liquid_results), liquid_cache)
    else:
        logger.info("=== Step 1: Running liquid dynamics ===")
        liquid_results: list[dict] = []
        for i, (wmin, wmaxmin) in enumerate(cases):
            t0 = time.time()
            logger.info("  [%d/%d] wmin=%.3f  wmaxmin=%.1f ...",
                         i + 1, len(cases), wmin, wmaxmin)
            res = _run_liquid_case(wmin, wmaxmin, cfg, forcing_model, forcing_cfg)
            elapsed = time.time() - t0
            logger.info("    done in %.1f s  (h range [%.0f, %.0f] m)",
                         elapsed, np.min(res["h_rec"]), np.max(res["h_rec"]))
            liquid_results.append(res)
        with open(liquid_cache, "wb") as fh:
            pickle.dump(liquid_results, fh)
        logger.info("  Cached liquid results to %s", liquid_cache)

    # ------------------------------------------------------------------
    # Step 2: analyze global ranges for lookup table
    # ------------------------------------------------------------------
    logger.info("=== Step 2: Analyzing depth/width ranges ===")
    all_depths = np.concatenate([r["depth_rec"] for r in liquid_results])
    all_widths = np.concatenate([r["w_rec"] for r in liquid_results])

    depth_min, depth_max = float(np.nanmin(all_depths)), float(np.nanmax(all_depths))
    width_min, width_max = float(np.nanmin(all_widths)), float(np.nanmax(all_widths))

    depth_min = max(depth_min, 0.1)
    width_min = max(width_min, 1e-4)

    logger.info("  depth range: [%.1f, %.1f] m", depth_min, depth_max)
    logger.info("  width range: [%.4f, %.4f] m", width_min, width_max)

    # ------------------------------------------------------------------
    # Step 3: build lookup table
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Building gas dynamics lookup table ===")
    lt_cfg = run_cfg.get("lookup_table", {})
    n_grid = lt_cfg.get("n_grid", 25)
    Tb_list = lt_cfg.get("Tb", [272.0, 273.1501])
    n_jobs = lt_cfg.get("jobs", -1)
    depth_margin = lt_cfg.get("depth_margin", 0.1)
    width_margin = lt_cfg.get("width_margin", 0.1)
    width_spacing = lt_cfg.get("width_spacing", "log")
    depth_spacing = lt_cfg.get("depth_spacing", "log")

    lookup_path = str(out_dir / "lookup_table.npz")

    if Path(lookup_path).exists() and not run_cfg.get("force_rebuild_lookup", False):
        logger.info("  Reusing existing lookup table: %s", lookup_path)
    else:
        t0 = time.time()
        build_lookup_from_bounds(
            depth_range=(depth_min, depth_max),
            width_range=(width_min, width_max),
            n_grid=n_grid,
            Tb=np.array(Tb_list),
            output_path=lookup_path,
            n_jobs=n_jobs,
            depth_margin=depth_margin,
            width_margin=width_margin,
            width_spacing=width_spacing,
            depth_spacing=depth_spacing,
        )
        logger.info("  Lookup table built in %.1f s", time.time() - t0)

    # ------------------------------------------------------------------
    # Step 4: vectorized gas interpolation
    # ------------------------------------------------------------------
    logger.info("=== Step 4: Vectorized gas interpolation + overflow ===")
    lookup = GasLookupTable(lookup_path)
    Tb_interp = float(np.mean(Tb_list))

    Lv = float(cfg.physical.latent_heat)
    Lf = float(cfg.physical.latent_heat_fusion)
    f_evap = Lf / (Lv + Lf)
    rho_w = float(cfg.physical.liquid_density)
    L = float(cfg.physical.equilibrium_depth)
    logger.info("  Overflow evap fraction f_evap = Lf/(Lv+Lf) = %.4f", f_evap)

    summary_rows: list[dict] = []

    for res in liquid_results:
        wmin = res["wmin"]
        wmaxmin = res["wmaxmin"]
        logger.info("  Gas interp + overflow: wmin=%.3f  wmaxmin=%.1f ...",
                     wmin, wmaxmin)

        r_arr, phi_arr, rho_arr, phi0_arr = lookup.query_vectorized(
            Tb_interp, res["depth_rec"], res["w_rec"])

        gas_mass_flux = phi_arr * res["w_rec"]

        overflow_dhdt = compute_overflow_rate(
            res["t_rec"], res["v_rec"], res["h_rec"],
            res["w_in"], res["t_in"], L, cfg)
        overflow_evap_flux = f_evap * rho_w * res["w_rec"] * overflow_dhdt
        total_mass_flux = gas_mass_flux + overflow_evap_flux

        res["r"] = r_arr
        res["phi_top"] = phi_arr
        res["rho_top"] = rho_arr
        res["phi0"] = phi0_arr
        res["gas_mass_flux"] = gas_mass_flux
        res["overflow_evap_flux"] = overflow_evap_flux
        res["mass_flux"] = total_mass_flux
        res["overflow_dhdt"] = overflow_dhdt

        case_name = f"wmin{wmin:.3f}_ratio{wmaxmin:.1f}"
        case_path = str(out_dir / f"{case_name}.npz")
        np.savez(
            case_path,
            wmin=wmin, wmaxmin=wmaxmin,
            t=res["t_rec"], h=res["h_rec"], w=res["w_rec"],
            depth=res["depth_rec"],
            r=r_arr, phi_top=phi_arr, rho_top=rho_arr, phi0=phi0_arr,
            gas_mass_flux=gas_mass_flux,
            overflow_evap_flux=overflow_evap_flux,
            mass_flux=total_mass_flux,
            overflow_dhdt=overflow_dhdt,
        )

        valid = np.isfinite(total_mass_flux)
        overflow_total = float(np.nansum(overflow_evap_flux))
        summary_rows.append(dict(
            wmin=wmin,
            wmaxmin=wmaxmin,
            file=case_name + ".npz",
            h_min=float(np.nanmin(res["h_rec"])),
            h_max=float(np.nanmax(res["h_rec"])),
            phi_top_mean=float(np.nanmean(phi_arr[valid])) if np.any(valid) else np.nan,
            mass_flux_mean=float(np.nanmean(total_mass_flux[valid])) if np.any(valid) else np.nan,
            overflow_frac=float(np.nansum(overflow_evap_flux) /
                                max(np.nansum(total_mass_flux), 1e-30)),
            pct_valid=float(np.sum(valid) / len(valid) * 100),
        ))

    # ------------------------------------------------------------------
    # Step 5: save summary
    # ------------------------------------------------------------------
    logger.info("=== Step 5: Saving summary ===")
    summary_path = str(out_dir / "summary.npz")
    np.savez(
        summary_path,
        wmin=np.array([r["wmin"] for r in summary_rows]),
        wmaxmin=np.array([r["wmaxmin"] for r in summary_rows]),
        files=np.array([r["file"] for r in summary_rows]),
        h_min=np.array([r["h_min"] for r in summary_rows]),
        h_max=np.array([r["h_max"] for r in summary_rows]),
        phi_top_mean=np.array([r["phi_top_mean"] for r in summary_rows]),
        mass_flux_mean=np.array([r["mass_flux_mean"] for r in summary_rows]),
        overflow_frac=np.array([r["overflow_frac"] for r in summary_rows]),
        pct_valid=np.array([r["pct_valid"] for r in summary_rows]),
    )
    logger.info("Summary saved to %s", summary_path)

    # ------------------------------------------------------------------
    # Step 6: generate plots
    # ------------------------------------------------------------------
    logger.info("=== Step 6: Generating plots ===")
    _generate_plots(liquid_results, wmin_list, wmaxmin_list, cfg, out_dir)

    logger.info("=== Pipeline complete: %d cases, results in %s ===",
                len(cases), out_dir)


# ======================================================================
# Plotting
# ======================================================================

_CMAP_WMIN = {0.10: "#d62728", 0.20: "#ff7f0e", 0.50: "#2ca02c", 1.00: "#1f77b4"}


def _lookup_result(results: list[dict], wmin: float, wmaxmin: float) -> dict:
    for r in results:
        if abs(r["wmin"] - wmin) < 1e-6 and abs(r["wmaxmin"] - wmaxmin) < 1e-6:
            return r
    raise KeyError(f"No result for wmin={wmin}, wmaxmin={wmaxmin}")


def _generate_plots(
    results: list[dict],
    wmin_list: list[float],
    wmaxmin_list: list[float],
    cfg,
    out_dir: Path,
) -> None:
    P = cfg.physical.orbital_period
    L = cfg.physical.equilibrium_depth
    D = L / 10.0

    nrow = len(wmin_list)
    ncol = len(wmaxmin_list)

    # --- Figure 1: liquid dynamics h(t) grid ---
    fig1, axes1 = plt.subplots(
        nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
        squeeze=False, constrained_layout=True,
    )
    fig1.suptitle("Water level h(t) — last orbital period", fontsize=14, fontweight="bold")

    for i, wmin in enumerate(wmin_list):
        for j, wmaxmin in enumerate(wmaxmin_list):
            ax = axes1[i, j]
            res = _lookup_result(results, wmin, wmaxmin)
            t_rec = res["t_rec"]
            h_rec = res["h_rec"]

            last_mask = t_rec > (cfg.liquid_dynamics.n_periods - 1) * P
            if np.sum(last_mask) > 10:
                t_phase = (t_rec[last_mask] - t_rec[last_mask][0]) / P
                h_lp = h_rec[last_mask]
            else:
                t_phase = t_rec / (P * cfg.liquid_dynamics.n_periods)
                h_lp = h_rec

            color = _CMAP_WMIN.get(wmin, "#333333")
            ax.plot(t_phase, h_lp, color=color, linewidth=0.9)
            ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")

            h_lo, h_hi = float(np.nanmin(h_lp)), float(np.nanmax(h_lp))
            margin = max(0.15 * (h_hi - h_lo), 5.0)
            ax.set_ylim(h_lo - margin, h_hi + margin)

            if h_hi > D * 0.8:
                ax.axhline(D, color="red", linewidth=0.6, linestyle=":", alpha=0.5)
            if h_lo < -L + L * 0.05:
                ax.axhline(-L, color="blue", linewidth=0.6, linestyle=":", alpha=0.5)

            amp = h_hi - h_lo
            ax.set_title(
                f"$w_{{min}}$={wmin} m, ratio={wmaxmin}\namp={amp:.0f} m",
                fontsize=9, fontweight="bold",
            )
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if i == nrow - 1:
                ax.set_xlabel("Phase", fontsize=8)
            if j == 0:
                ax.set_ylabel("h (m)", fontsize=8)

    p1 = out_dir / "plot_liquid_grid.png"
    fig1.savefig(str(p1), dpi=180)
    plt.close(fig1)
    logger.info("  Saved %s", p1)

    # --- Figure 2: mass flux grid (gas + overflow) ---
    fig2, axes2 = plt.subplots(
        nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
        squeeze=False, constrained_layout=True,
    )
    fig2.suptitle("Mass flux $\\dot{m}(t)$ — last orbital period", fontsize=14, fontweight="bold")

    for i, wmin in enumerate(wmin_list):
        for j, wmaxmin in enumerate(wmaxmin_list):
            ax = axes2[i, j]
            res = _lookup_result(results, wmin, wmaxmin)
            t_rec = res["t_rec"]
            mf_total = res["mass_flux"]
            mf_gas = res.get("gas_mass_flux", mf_total)
            mf_over = res.get("overflow_evap_flux")

            last_mask = t_rec > (cfg.liquid_dynamics.n_periods - 1) * P
            if np.sum(last_mask) > 10:
                t_phase = (t_rec[last_mask] - t_rec[last_mask][0]) / P
                mf_t_lp = mf_total[last_mask]
                mf_g_lp = mf_gas[last_mask]
                mf_o_lp = mf_over[last_mask] if mf_over is not None else None
            else:
                t_phase = t_rec / (P * cfg.liquid_dynamics.n_periods)
                mf_t_lp = mf_total
                mf_g_lp = mf_gas
                mf_o_lp = mf_over

            color = _CMAP_WMIN.get(wmin, "#333333")
            ax.plot(t_phase, mf_t_lp, color=color, linewidth=0.9, label="total")

            has_overflow = mf_o_lp is not None and np.nanmax(mf_o_lp) > 1e-10
            if has_overflow:
                ax.fill_between(t_phase, mf_g_lp, mf_t_lp,
                                color="#d62728", alpha=0.3, label="overflow evap")
                ax.plot(t_phase, mf_g_lp, color=color, linewidth=0.5,
                        linestyle="--", alpha=0.6, label="gas only")

            valid = np.isfinite(mf_t_lp)
            if np.any(valid):
                mf_lo = float(np.nanmin(mf_t_lp[valid]))
                mf_hi = float(np.nanmax(mf_t_lp[valid]))
                mf_margin = max(0.15 * (mf_hi - mf_lo), 0.01)
                ax.set_ylim(max(mf_lo - mf_margin, 0), mf_hi + mf_margin)
                mf_mean = float(np.nanmean(mf_t_lp[valid]))
                ax.axhline(mf_mean, color=color, linewidth=0.7, linestyle=":",
                           alpha=0.6, label=f"mean={mf_mean:.3f}")

            if has_overflow or (i == 0 and j == 0):
                ax.legend(fontsize=5, loc="best")

            ax.set_title(
                f"$w_{{min}}$={wmin} m, ratio={wmaxmin}",
                fontsize=9, fontweight="bold",
            )
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if i == nrow - 1:
                ax.set_xlabel("Phase", fontsize=8)
            if j == 0:
                ax.set_ylabel("Mass flux (kg/m/s)", fontsize=8)

    p2 = out_dir / "plot_massflux_grid.png"
    fig2.savefig(str(p2), dpi=180)
    plt.close(fig2)
    logger.info("  Saved %s", p2)

    # --- Figure 3: summary heatmaps (total flux, overflow fraction, amplitude) ---
    nw = len(wmin_list)
    nr = len(wmaxmin_list)
    mf_grid = np.full((nw, nr), np.nan)
    amp_grid = np.full((nw, nr), np.nan)
    ov_grid = np.full((nw, nr), np.nan)

    for res in results:
        i = wmin_list.index(res["wmin"])
        j = wmaxmin_list.index(res["wmaxmin"])
        valid = np.isfinite(res["mass_flux"])
        if np.any(valid):
            mf_grid[i, j] = float(np.nanmean(res["mass_flux"][valid]))
            gas_sum = float(np.nansum(res.get("gas_mass_flux", res["mass_flux"])[valid]))
            ov_sum = float(np.nansum(res.get("overflow_evap_flux",
                                             np.zeros_like(res["mass_flux"]))[valid]))
            total_sum = gas_sum + ov_sum
            ov_grid[i, j] = ov_sum / max(total_sum, 1e-30) * 100.0

        last_mask = res["t_rec"] > (cfg.liquid_dynamics.n_periods - 1) * P
        if np.sum(last_mask) > 10:
            h_lp = res["h_rec"][last_mask]
        else:
            h_lp = res["h_rec"]
        amp_grid[i, j] = float(np.nanmax(h_lp) - np.nanmin(h_lp))

    fig3, (ax_mf, ax_ov, ax_amp) = plt.subplots(
        1, 3, figsize=(20, 5), constrained_layout=True)

    im1 = ax_mf.imshow(
        mf_grid, origin="lower", aspect="auto",
        extent=[wmaxmin_list[0] - 0.5, wmaxmin_list[-1] + 0.5,
                wmin_list[0] - 0.025, wmin_list[-1] + 0.025],
        cmap="YlOrRd",
    )
    for i in range(nw):
        for j in range(nr):
            val = mf_grid[i, j]
            if np.isfinite(val):
                ax_mf.text(wmaxmin_list[j], wmin_list[i], f"{val:.3f}",
                           ha="center", va="center", fontsize=8, fontweight="bold")
    ax_mf.set_xticks(wmaxmin_list)
    ax_mf.set_yticks(wmin_list)
    ax_mf.set_xlabel("$w_{max}/w_{min}$", fontsize=11)
    ax_mf.set_ylabel("$w_{min}$ (m)", fontsize=11)
    ax_mf.set_title("Mean total mass flux (kg/m/s)", fontsize=12, fontweight="bold")
    fig3.colorbar(im1, ax=ax_mf, shrink=0.8)

    im_ov = ax_ov.imshow(
        ov_grid, origin="lower", aspect="auto",
        extent=[wmaxmin_list[0] - 0.5, wmaxmin_list[-1] + 0.5,
                wmin_list[0] - 0.025, wmin_list[-1] + 0.025],
        cmap="Reds", vmin=0,
    )
    for i in range(nw):
        for j in range(nr):
            val = ov_grid[i, j]
            if np.isfinite(val):
                ax_ov.text(wmaxmin_list[j], wmin_list[i], f"{val:.1f}%",
                           ha="center", va="center", fontsize=8, fontweight="bold")
    ax_ov.set_xticks(wmaxmin_list)
    ax_ov.set_yticks(wmin_list)
    ax_ov.set_xlabel("$w_{max}/w_{min}$", fontsize=11)
    ax_ov.set_ylabel("$w_{min}$ (m)", fontsize=11)
    ax_ov.set_title("Overflow evap fraction (%)", fontsize=12, fontweight="bold")
    fig3.colorbar(im_ov, ax=ax_ov, shrink=0.8)

    im2 = ax_amp.imshow(
        amp_grid, origin="lower", aspect="auto",
        extent=[wmaxmin_list[0] - 0.5, wmaxmin_list[-1] + 0.5,
                wmin_list[0] - 0.025, wmin_list[-1] + 0.025],
        cmap="YlGnBu",
    )
    for i in range(nw):
        for j in range(nr):
            val = amp_grid[i, j]
            if np.isfinite(val):
                ax_amp.text(wmaxmin_list[j], wmin_list[i], f"{val:.0f}",
                            ha="center", va="center", fontsize=8, fontweight="bold")
    ax_amp.set_xticks(wmaxmin_list)
    ax_amp.set_yticks(wmin_list)
    ax_amp.set_xlabel("$w_{max}/w_{min}$", fontsize=11)
    ax_amp.set_ylabel("$w_{min}$ (m)", fontsize=11)
    ax_amp.set_title("Water level amplitude (m)", fontsize=12, fontweight="bold")
    fig3.colorbar(im2, ax=ax_amp, shrink=0.8)

    p3 = out_dir / "plot_summary_heatmap.png"
    fig3.savefig(str(p3), dpi=180)
    plt.close(fig3)
    logger.info("  Saved %s", p3)

    # --- Figure 4: phi_top time series grid ---
    fig4, axes4 = plt.subplots(
        nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
        squeeze=False, constrained_layout=True,
    )
    fig4.suptitle("Top vapor flux $\\phi_{top}(t)$ — last orbital period",
                  fontsize=14, fontweight="bold")

    for i, wmin in enumerate(wmin_list):
        for j, wmaxmin in enumerate(wmaxmin_list):
            ax = axes4[i, j]
            res = _lookup_result(results, wmin, wmaxmin)
            t_rec = res["t_rec"]
            phi = res["phi_top"]

            last_mask = t_rec > (cfg.liquid_dynamics.n_periods - 1) * P
            if np.sum(last_mask) > 10:
                t_phase = (t_rec[last_mask] - t_rec[last_mask][0]) / P
                phi_lp = phi[last_mask]
            else:
                t_phase = t_rec / (P * cfg.liquid_dynamics.n_periods)
                phi_lp = phi

            color = _CMAP_WMIN.get(wmin, "#333333")
            ax.plot(t_phase, phi_lp, color=color, linewidth=0.9)

            valid = np.isfinite(phi_lp)
            if np.any(valid):
                p_lo = float(np.nanmin(phi_lp[valid]))
                p_hi = float(np.nanmax(phi_lp[valid]))
                p_margin = max(0.15 * (p_hi - p_lo), 0.005)
                ax.set_ylim(p_lo - p_margin, p_hi + p_margin)

            ax.set_title(
                f"$w_{{min}}$={wmin} m, ratio={wmaxmin}",
                fontsize=9, fontweight="bold",
            )
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if i == nrow - 1:
                ax.set_xlabel("Phase", fontsize=8)
            if j == 0:
                ax.set_ylabel("$\\phi_{top}$ (kg/m$^2$/s)", fontsize=8)

    p4 = out_dir / "plot_phitop_grid.png"
    fig4.savefig(str(p4), dpi=180)
    plt.close(fig4)
    logger.info("  Saved %s", p4)


if __name__ == "__main__":
    main()
