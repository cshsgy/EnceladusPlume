"""Tests for the mass-flux peak predictor (secondary vs main)."""

import numpy as np
import pytest

from enceladus_plume.config import load_config
from enceladus_plume.gas_dynamics.interpolator import generate_r_table
from enceladus_plume.gas_dynamics.lookup import GasLookupTable
from enceladus_plume.peaks import predict_peaks

pytestmark = pytest.mark.slow  # builds a lookup + runs liquid solves


@pytest.fixture(scope="module")
def lookup(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("lut") / "lut.npz")
    generate_r_table(np.geomspace(1e-3, 0.08, 8), np.geomspace(0.5, 700.0, 8),
                     Tb_arr=np.array([272.0, 273.1501]), output_path=path, n_jobs=-1)
    return GasLookupTable(path, clean=True)


def _cfg():
    cfg = load_config()
    cfg.liquid_dynamics.n_periods = 2
    cfg.liquid_dynamics.max_step = 400.0
    cfg.liquid_dynamics.npts_velocity = 120
    cfg.physical.equilibrium_depth = 5000.0
    return cfg


def test_overflow_case_has_two_peaks(lookup):
    pr = predict_peaks(_cfg(), delta_w=0.01, w_eff=0.007, lookup=lookup)
    assert pr.has_two_peaks
    assert pr.hmax_over_D > 0.9
    # widening near max-width phase; approach later, near the max-water phase
    assert 150.0 <= pr.phi_widening <= 230.0
    assert 250.0 <= pr.phi_approach <= 345.0
    assert pr.phi_approach > pr.phi_widening
    assert pr.ratio > 0.0
    assert 40.0 <= pr.peak_separation <= 160.0


def test_no_close_approach_single_peak(lookup):
    # large width, weak swing -> water stays deep -> only the widening peak
    pr = predict_peaks(_cfg(), delta_w=0.03, w_eff=0.014, lookup=lookup)
    assert pr.hmax_over_D < 0.7
    assert not pr.has_two_peaks
    assert np.isnan(pr.phi_approach)
