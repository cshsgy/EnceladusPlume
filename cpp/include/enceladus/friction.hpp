// Darcy-Weisbach friction-factor models for parallel-plate crack geometry.
//
// C++ port of enceladus_plume/enceladus_plume/friction.py. Kept numerically
// identical to the Python reference so the two implementations are
// interchangeable (see cpp/tests and the Python parity test).
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace enceladus {

// Hydraulic diameter for infinite parallel plates with gap w.
inline double hydraulic_diameter(double w) { return 2.0 * w; }

// Bulk Reynolds number based on hydraulic diameter.
inline double reynolds_number(double rho, double v, double D_h, double mu) {
    if (mu <= 0.0 || D_h <= 0.0) return 0.0;
    return rho * std::fabs(v) * D_h / mu;
}

// Convert a constant Fanning coefficient to a Darcy factor.
inline double darcy_constant(double Cf) { return 8.0 * Cf; }

// Laminar Darcy factor for parallel plates: f_D = C_lam / Re.
inline double darcy_laminar(double Re, double C_lam = 96.0) {
    if (Re < 1e-12) return C_lam / 1e-12;
    return C_lam / Re;
}

// Churchill (1977) friction factor -- smooth blend across all regimes.
inline double darcy_churchill(double Re, double roughness = 0.0,
                              double D_h = 1.0, double C_lam = 96.0) {
    if (Re < 1e-12) return C_lam / 1e-12;

    const double eps_over_D = (D_h > 0.0) ? roughness / D_h : 0.0;

    const double lam_term = std::pow(C_lam / (8.0 * Re), 12.0);

    double inner = std::pow(7.0 / Re, 0.9) + 0.27 * eps_over_D;
    if (inner < 1e-30) inner = 1e-30;
    const double A = std::pow(-2.457 * std::log(inner), 16.0);
    const double B = std::pow(37530.0 / Re, 16.0);
    const double turb_term = std::pow(A + B, -1.5);

    return 8.0 * std::pow(lam_term + turb_term, 1.0 / 12.0);
}

// Unified interface: Fanning friction coefficient C_f for the given conditions.
// This is the value that plugs into the formulation 2 * C_f / w * v|v|.
inline double fanning_friction_factor(const std::string& model, double v,
                                      double w, double rho = 1.0,
                                      double mu = 1.0, double roughness = 0.0,
                                      double Cf_constant = 0.004,
                                      double C_lam = 96.0) {
    if (model == "constant") return Cf_constant;

    const double D_h = hydraulic_diameter(w);
    const double Re = reynolds_number(rho, v, D_h, mu);

    double f_D;
    if (model == "laminar") {
        f_D = darcy_laminar(Re, C_lam);
    } else if (model == "churchill") {
        f_D = darcy_churchill(Re, roughness, D_h, C_lam);
    } else {
        throw std::invalid_argument("Unknown friction model: " + model);
    }
    return f_D / 8.0;
}

}  // namespace enceladus
