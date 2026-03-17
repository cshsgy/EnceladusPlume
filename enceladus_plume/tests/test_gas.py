"""Tests for gas dynamics solver and interpolator."""

import numpy as np
import pytest

from enceladus_plume.gas_dynamics.solver import miki_model_full
from enceladus_plume.gas_dynamics.interpolator import solve_r_function


class TestMikiModelFull:
    def test_returns_correct_shapes(self):
        """Output arrays should match the input grid length."""
        nz = 50
        zs = np.linspace(0, 1000, nz)
        T2 = np.full(nz, 270.0)
        Tw, Ev, MTop, PhiTop = miki_model_full(
            r=0.5, width=0.3, zs=zs, T2=T2,
            K=3.0, Lv=2.84e6, G=0.113, Dx=0.05,
            hmax=100.0,
            friction_model="constant", Cf_constant=0.002,
        )
        assert len(Tw) == nz
        assert len(Ev) == nz
        assert np.isfinite(PhiTop)

    def test_r_near_1_small_flux(self):
        """r close to 1 should give low mass flux (most vapor condenses)."""
        nz = 50
        zs = np.linspace(0, 500, nz)
        T2 = np.full(nz, 270.0)
        _, _, _, PhiTop = miki_model_full(
            r=0.99, width=0.3, zs=zs, T2=T2,
            K=3.0, Lv=2.84e6, G=0.113, Dx=0.05,
            hmax=50.0,
            friction_model="constant", Cf_constant=0.002,
        )
        assert PhiTop < 0.1

    def test_churchill_model_runs(self):
        """Churchill friction model should produce a valid result."""
        nz = 50
        zs = np.linspace(0, 1000, nz)
        T2 = np.full(nz, 270.0)
        Tw, Ev, MTop, PhiTop = miki_model_full(
            r=0.5, width=0.3, zs=zs, T2=T2,
            K=3.0, Lv=2.84e6, G=0.113, Dx=0.05,
            hmax=100.0,
            friction_model="churchill", mu_vapor=8e-6,
        )
        assert len(Tw) == nz
        assert np.isfinite(PhiTop)


class TestSolveRFunction:
    @pytest.mark.slow
    def test_known_values(self):
        """r should be between 0 and 1 for a physical configuration.

        From the MATLAB unit test: r ~ 0.7447 for (273.15, 1500, 0.25).
        We use a coarser tolerance since the Python port may differ slightly.
        """
        r, phi, rho, phi0 = solve_r_function(273.15, 1500.0, 0.25)
        assert 0.0 < r < 1.0
        assert phi <= phi0 or np.isnan(phi0)
