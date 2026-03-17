"""Tests for the Darcy-Weisbach friction module."""

import math
import numpy as np
import pytest

from enceladus_plume.friction import (
    hydraulic_diameter,
    reynolds_number,
    darcy_constant,
    darcy_laminar,
    darcy_churchill,
    fanning_friction_factor,
)


class TestHydraulicDiameter:
    def test_parallel_plates(self):
        assert hydraulic_diameter(0.5) == pytest.approx(1.0)
        assert hydraulic_diameter(0.01) == pytest.approx(0.02)


class TestReynoldsNumber:
    def test_basic(self):
        Re = reynolds_number(rho=1000.0, v=1.0, D_h=0.02, mu=1e-3)
        assert Re == pytest.approx(20000.0)

    def test_zero_velocity(self):
        assert reynolds_number(1000.0, 0.0, 0.02, 1e-3) == 0.0


class TestDarcyConstant:
    def test_conversion(self):
        """f_D = 8 * Cf."""
        assert darcy_constant(0.004) == pytest.approx(0.032)


class TestDarcyLaminar:
    def test_parallel_plate_limit(self):
        """At Re = 960, f_D = 96/960 = 0.1."""
        assert darcy_laminar(960.0) == pytest.approx(0.1)

    def test_very_small_Re(self):
        """Should not blow up for tiny Re."""
        f = darcy_laminar(1e-20)
        assert np.isfinite(f) and f > 0


class TestDarcyChurchill:
    def test_laminar_limit(self):
        """At low Re, Churchill should converge to C_lam / Re."""
        Re = 100.0
        f = darcy_churchill(Re, roughness=0.0, D_h=0.02, C_lam=96.0)
        f_lam = 96.0 / Re
        assert abs(f - f_lam) / f_lam < 0.01  # within 1%

    def test_turbulent_reasonable(self):
        """At high Re, friction factor should be a small positive number."""
        f = darcy_churchill(1e5, roughness=0.0, D_h=0.02, C_lam=96.0)
        assert 0.01 < f < 0.05

    def test_roughness_increases_friction(self):
        """Non-zero roughness should increase friction at high Re."""
        f_smooth = darcy_churchill(1e5, roughness=0.0, D_h=0.02)
        f_rough = darcy_churchill(1e5, roughness=1e-4, D_h=0.02)
        assert f_rough > f_smooth


class TestFanningFrictionFactor:
    def test_constant_model(self):
        Cf = fanning_friction_factor("constant", 1.0, 0.3, Cf_constant=0.005)
        assert Cf == pytest.approx(0.005)

    def test_laminar_model(self):
        Cf = fanning_friction_factor(
            "laminar", 0.01, 0.1, rho=1000.0, mu=1.8e-3)
        Re = 1000.0 * 0.01 * 0.2 / 1.8e-3  # ~1111
        expected_Cf = (96.0 / Re) / 8.0
        assert Cf == pytest.approx(expected_Cf, rel=1e-6)

    def test_churchill_model(self):
        Cf = fanning_friction_factor(
            "churchill", 1.0, 0.3, rho=1000.0, mu=1.8e-3)
        assert np.isfinite(Cf) and Cf > 0

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            fanning_friction_factor("bogus", 1.0, 0.3)
