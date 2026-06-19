// Thermodynamic helpers shared by the gas-column solver.
//
// C++ ports of the subset of enceladus_plume/physics.py needed by the hot
// loops (gas dynamics). Kept numerically identical to the Python reference.
#pragma once

#include <cmath>

namespace enceladus {

// Clausius-Clapeyron vapor pressure of water ice (Pa).
inline double vapor_pressure(double T) {
    return 3.63e12 * std::exp(-6147.0 / T);
}

// Simplified evaporation rate used in the 2023 gas interpolator.
inline double evaporation_rate_simple(double ec, double bv, double Tm) {
    const double T0 = 273.15;
    return ec / std::sqrt(Tm) * std::exp(bv * (1.0 / T0 - 1.0 / Tm));
}

// Bisection for surface temperature, then the corresponding evaporation rate.
// Solves  2*sigma*(Ts^4 - Te^4) + (4*k/(pi*d))*(Ts - Tw) = 0  for Ts,
// then returns  E = -sigma/L * (Ts^4 - Te^4).
inline double find_evap_surface(double Tw, double Te, double sigma, double d,
                                double k, double L) {
    const double PI = 3.141592653589793;
    const double c1 = 2.0 * sigma;
    const double c2 = 4.0 * k / (PI * d);

    double T_l = 1.0;
    double T_r = Tw;
    for (int it = 0; it < 200; ++it) {
        if (T_r - T_l <= 1e-8) break;
        const double T_m = 0.5 * (T_l + T_r);
        const double v_l = c1 * (std::pow(T_l, 4.0) - std::pow(Te, 4.0)) + c2 * (T_l - Tw);
        const double v_m = c1 * (std::pow(T_m, 4.0) - std::pow(Te, 4.0)) + c2 * (T_m - Tw);
        if (v_l * v_m < 0.0) {
            T_r = T_m;
        } else {
            T_l = T_m;
        }
    }

    const double Ts = 0.5 * (T_l + T_r);
    return -sigma / L * (std::pow(Ts, 4.0) - std::pow(Te, 4.0));
}

}  // namespace enceladus
