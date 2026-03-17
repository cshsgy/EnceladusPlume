"""Tests for the shared physics functions."""

import math
import numpy as np
import pytest

from enceladus_plume.physics import (
    vapor_pressure,
    evaporation_rate,
    evaporation_rate_simple,
    surface_temperature,
    miki_find_evap,
    find_evap_surface,
    function_to_solve,
)


class TestVaporPressure:
    def test_at_273(self):
        """Pv(273.15) should be a known value from the Clausius-Clapeyron fit."""
        Pv = vapor_pressure(273.15)
        expected = 3.63e12 * math.exp(-6147.0 / 273.15)
        assert abs(Pv - expected) < 1e-3

    def test_monotonic(self):
        """Vapor pressure increases with temperature."""
        T = np.linspace(200, 300, 50)
        Pv = vapor_pressure(T)
        assert np.all(np.diff(Pv) > 0)


class TestEvaporationRate:
    def test_zero_when_equal(self):
        """When gas and wall have same T and P, net evaporation ~ 0."""
        rg = 8.341 / 0.018
        T = 273.15
        P = vapor_pressure(T)
        E = evaporation_rate(P, P, T, T, rg)
        assert abs(E) < 1e-10

    def test_positive_when_wall_warmer(self):
        """Warmer wall should produce positive net evaporation."""
        rg = 8.341 / 0.018
        E = evaporation_rate(vapor_pressure(260.0), vapor_pressure(273.15),
                             260.0, 273.15, rg)
        assert E > 0


class TestSurfaceTemperature:
    def test_positive_root(self):
        """Should return a positive temperature."""
        Ts = surface_temperature(68.0, 273.15, 2000.0, 3.0)
        assert Ts > 0

    def test_reasonable_range(self):
        """Surface temp should be between Te and Tw."""
        Te = 68.0
        Tw = 273.15
        Ts = surface_temperature(Te, Tw, 2000.0, 3.0)
        assert Te <= Ts <= Tw


class TestMikiFindEvap:
    def test_converges(self):
        """Secant method should converge and return reasonable Tw."""
        rg = 8.341 / 0.018
        Tw, Ev = miki_find_evap(270.0, 260.0, 500.0, 0.05, 3.0, 2.84e6, rg)
        assert 200 < Tw < 280
        assert np.isfinite(Ev)


class TestEvaporationRateSimple:
    def test_at_T0(self):
        """At T0 = 273.15 the exponential factor = 1."""
        rg = 8.341 / 0.018
        ec = vapor_pressure(273.15) / math.sqrt(2 * math.pi * rg)
        bv = 2.8e6 / rg
        E = evaporation_rate_simple(ec, bv, 273.15)
        expected = ec / math.sqrt(273.15)
        assert abs(E - expected) / expected < 1e-6


class TestFindEvapSurface:
    def test_returns_finite(self):
        """Should produce a finite evaporation rate."""
        E = find_evap_surface(273.15, 68.0, 5.67e-8, 1000.0, 2.4, 2.8e6)
        assert np.isfinite(E)
