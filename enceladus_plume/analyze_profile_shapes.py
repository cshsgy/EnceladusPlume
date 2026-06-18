#!/usr/bin/env python3
"""Summarize two-peak mass-flux shape diagnostics for a pipeline output directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from enceladus_plume.config import load_config


def _moving_average(values: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def _select_last_period(t: np.ndarray, values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    if len(t) == 0:
        return t, values
    start = max(float(t[-1]) - period, float(t[0]))
    mask = t >= start
    t_last = t[mask]
    v_last = values[mask]
    if len(t_last) < 3:
        return t, values
    phase_deg = (t_last - t_last[0]) / period * 360.0
    return phase_deg, v_last


def _find_peaks(phase_deg: np.ndarray, values: np.ndarray, min_sep_deg: float = 25.0) -> list[tuple[float, float]]:
    if len(values) < 3:
        return []
    smoothed = _moving_average(values)
    candidates: list[tuple[float, float]] = []
    for i in range(1, len(smoothed) - 1):
        if not np.isfinite(smoothed[i]):
            continue
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            candidates.append((float(phase_deg[i]), float(smoothed[i])))

    candidates.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[float, float]] = []
    for phase, amp in candidates:
        if all(abs(phase - prev_phase) >= min_sep_deg for prev_phase, _ in selected):
            selected.append((phase, amp))
    selected.sort(key=lambda item: item[0])
    return selected


def _summarize_case(path: Path, period: float) -> dict[str, float | str]:
    data = np.load(path)
    phase_deg, mass_flux = _select_last_period(data["t"], data["mass_flux"], period)
    peaks = _find_peaks(phase_deg, mass_flux)
    if not peaks:
        return {
            "file": path.name,
            "wmin": float(data["wmin"]),
            "wmaxmin": float(data["wmaxmin"]),
            "major_peak_deg": np.nan,
            "major_peak_amp": np.nan,
            "first_peak_deg": np.nan,
            "first_peak_amp": np.nan,
            "peak_sep_deg": np.nan,
            "first_to_major_ratio": np.nan,
        }

    major_peak = max(peaks, key=lambda item: item[1])
    earlier_peaks = [peak for peak in peaks if peak[0] < major_peak[0]]
    first_peak = max(earlier_peaks, key=lambda item: item[1]) if earlier_peaks else (np.nan, np.nan)

    return {
        "file": path.name,
        "wmin": float(data["wmin"]),
        "wmaxmin": float(data["wmaxmin"]),
        "major_peak_deg": float(major_peak[0]),
        "major_peak_amp": float(major_peak[1]),
        "first_peak_deg": float(first_peak[0]),
        "first_peak_amp": float(first_peak[1]),
        "peak_sep_deg": float(major_peak[0] - first_peak[0]) if np.isfinite(first_peak[0]) else np.nan,
        "first_to_major_ratio": float(first_peak[1] / major_peak[1]) if np.isfinite(first_peak[1]) and major_peak[1] else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Pipeline output directory containing wmin*_ratio*.npz files")
    parser.add_argument("--target-separation-deg", type=float, default=150.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cfg = load_config()
    period = float(cfg.physical.orbital_period)

    rows = [_summarize_case(path, period) for path in sorted(output_dir.glob("wmin*_ratio*.npz"))]
    if not rows:
        raise SystemExit(f"No case files found in {output_dir}")

    def _score(row: dict[str, float | str]) -> float:
        sep = row["peak_sep_deg"]
        ratio = row["first_to_major_ratio"]
        score = 0.0
        if np.isfinite(sep):
            score += abs(float(sep) - args.target_separation_deg)
        else:
            score += 1.0e6
        if np.isfinite(ratio):
            score += abs(float(ratio) - 0.5) * 100.0
        else:
            score += 1.0e5
        return score

    rows.sort(key=_score)

    print(f"Output directory: {output_dir}")
    print("Best matches by two-peak shape:")
    for row in rows[:10]:
        print(
            f"- {row['file']}: "
            f"wmin={row['wmin']:.3f}, ratio={row['wmaxmin']:.1f}, "
            f"first_peak={row['first_peak_deg']:.1f} deg, "
            f"major_peak={row['major_peak_deg']:.1f} deg, "
            f"sep={row['peak_sep_deg']:.1f} deg, "
            f"first/major={row['first_to_major_ratio']:.2f}"
        )


if __name__ == "__main__":
    main()
