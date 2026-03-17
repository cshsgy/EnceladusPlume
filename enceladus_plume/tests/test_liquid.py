"""Tests for the liquid dynamics solver and helpers."""

import numpy as np
import pytest

from enceladus_plume.liquid_dynamics.helpers import (
    vel_now,
    friction,
    additional_term,
    pvpt,
)
from enceladus_plume.liquid_dynamics.solver import liquid_dynamics
from enceladus_plume.config import load_config


class TestVelNow:
    def test_linear_profile(self):
        """Velocity should vary linearly when dwdt/w is constant."""
        zs, vp = vel_now(1.0, 0.0, 100.0, 0.5, 0.01, npts=100)
        assert len(zs) == 100
        assert len(vp) == 100
        # dvdz = -dwdt/w = -0.02 -> slope * delta_z
        expected_slope = -0.01 / 0.5
        actual_slope = (vp[-1] - vp[0]) / (zs[-1] - zs[0])
        assert abs(actual_slope - expected_slope) < 1e-10

    def test_bottom_velocity(self):
        """First element should be v0."""
        zs, vp = vel_now(3.5, 10.0, 100.0, 1.0, 0.0)
        assert abs(vp[0] - 3.5) < 1e-12


class TestFriction:
    def test_zero_velocity_constant(self):
        """Zero velocity gives zero friction (constant model)."""
        zs = np.linspace(0, 100, 50)
        v = np.zeros(50)
        assert abs(friction(zs, v, 1.0, model="constant")) < 1e-15

    def test_positive_constant(self):
        """Positive velocity should give positive friction integral (constant)."""
        zs = np.linspace(0, 100, 200)
        v = np.ones(200) * 0.1
        f = friction(zs, v, 1.0, model="constant")
        assert f > 0

    def test_churchill_runs(self):
        """Churchill friction model should produce a finite result."""
        zs = np.linspace(0, 100, 200)
        v = np.ones(200) * 0.1
        f = friction(zs, v, 0.3, model="churchill",
                     rho=1000.0, mu=1.8e-3)
        assert np.isfinite(f)
        assert f > 0

    def test_laminar_higher_friction_at_low_Re(self):
        """At low Re the laminar model should give larger friction than
        the constant model with a moderate Cf."""
        zs = np.linspace(0, 100, 200)
        v = np.ones(200) * 1e-4  # very slow -> low Re
        w = 0.01
        f_const = friction(zs, v, w, model="constant", Cf_constant=0.004)
        f_lam = friction(zs, v, w, model="laminar",
                         rho=1000.0, mu=1.8e-3)
        assert f_lam > f_const


class TestPvpt:
    def test_zero_dwdt(self):
        """Zero dwdt should give zero result."""
        zs = np.linspace(0, 100, 50)
        assert abs(pvpt(zs, 1.0, 0.0, 0.0)) < 1e-15


class TestLiquidDynamicsSolver:
    def test_sinusoidal_runs(self):
        """Smoke test: sinusoidal width should produce output arrays."""
        cfg = load_config()
        P = cfg.physical.orbital_period
        t_in = np.arange(100, P + 1, 100.0)
        w_in = 0.5 + 0.2 * np.sin(t_in / P * 2 * np.pi)
        L = 20000.0

        cfg.liquid_dynamics.n_periods = 1
        cfg.liquid_dynamics.max_step = 100.0

        w_rec, h_rec, t_rec, v_rec = liquid_dynamics(w_in, t_in, L, cfg)
        assert len(w_rec) > 10
        assert len(h_rec) == len(t_rec)
        assert len(v_rec) == len(t_rec)
        assert t_rec[-1] > 0
