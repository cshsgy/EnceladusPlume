// Liquid-column derivative: the per-RHS-evaluation cost of the liquid solver.
//
// C++ port of enceladus_plume/liquid_dynamics/{helpers.py,solver.py::_derivative}.
// Computes (dv/dt, dh/dt) for the water column; the velocity profile, the wall
// friction integral, and the additional inertial term are all evaluated here so
// no arrays cross the Python/C++ boundary.
//
// Kept numerically identical to the Python reference: same np.linspace grid and
// same np.trapz integration.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "enceladus/friction.hpp"

namespace enceladus {

struct LiquidDerivative {
    double dvdt;
    double dhdt;
};

namespace detail {
// np.trapz(y, x) over the index range [lo, hi).
inline double trapz(const std::vector<double>& y, const std::vector<double>& x,
                    long lo, long hi) {
    double acc = 0.0;
    for (long i = lo; i + 1 < hi; ++i) {
        acc += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) * 0.5;
    }
    return acc;
}
}  // namespace detail

// dv/dt and dh/dt for the liquid column (matches solver.py::_derivative).
inline LiquidDerivative liquid_derivative(
    double v, double h, double L, double g, double w, double dwdt, double dwdt2,
    long npts = 1000, const std::string& friction_model = "constant",
    double Cf_constant = 0.004, double rho = 1000.0, double mu = 1.8e-3,
    double roughness = 0.0, double C_lam = 96.0) {
    const double col_height = h + L;
    if (col_height <= 0.0) return {0.0, 0.0};

    // vel_now: zs = linspace(-L, h, npts); v(z) = v + (-dwdt/w)*(z - zs[0]).
    std::vector<double> zs(npts), vp(npts);
    const double step = (h - (-L)) / static_cast<double>(npts - 1);
    const double dvdz = -dwdt / w;
    for (long i = 0; i < npts; ++i) {
        zs[i] = -L + static_cast<double>(i) * step;
    }
    zs[npts - 1] = h;  // np.linspace pins the endpoint exactly
    const double z0 = zs[0];
    for (long i = 0; i < npts; ++i) {
        vp[i] = v + dvdz * (zs[i] - z0);
    }

    // friction: integral of 2*Cf(z)/w * v|v| along the crack.
    std::vector<double> dfdt(npts);
    const bool constant = (friction_model == "constant");
    for (long i = 0; i < npts; ++i) {
        const double Cf_i = constant
            ? Cf_constant
            : fanning_friction_factor(friction_model, vp[i], w, rho, mu,
                                      roughness, Cf_constant, C_lam);
        dfdt[i] = 2.0 * Cf_i / w * vp[i] * std::fabs(vp[i]);
    }
    const double fric = detail::trapz(dfdt, zs, 0, npts);

    // Advection from the momentum integral: -1/2 (v_h^2 - v0^2),
    // v0 = vp[0] (floor), v_h = vp[npts-1] (water surface).
    double rhs = -0.5 * (vp[npts - 1] * vp[npts - 1] - vp[0] * vp[0]) - g * h - fric;

    // additional_term: double integral of the pvpt term (P linear in z).
    double add_term = 0.0;
    if (npts >= 3) {
        const double integrand_val = dwdt * dwdt / (w * w) - dwdt2 / w;
        std::vector<double> P(npts);
        for (long i = 0; i < npts; ++i) P[i] = integrand_val * (zs[i] - z0);
        add_term = detail::trapz(P, zs, 1, npts);  // np.trapz(P[1:], zs[1:])
    }
    rhs -= add_term;

    return {rhs / col_height, vp[npts - 1]};
}

}  // namespace enceladus
