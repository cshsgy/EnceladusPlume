// Gas-column solver: RK4 density integration + bisection for the inlet ratio r.
//
// C++ port of enceladus_plume/gas_dynamics/interpolator.py
// (_solve_function and solve_r_function). This is the dominant cost in the
// lookup-table build: find_evap_surface bisects at every depth step.
//
// Kept numerically identical to the Python reference (same operation order,
// same np.arange grid, same friction model).
#pragma once

#include <cmath>
#include <vector>

#include "enceladus/friction.hpp"
#include "enceladus/physics.hpp"

namespace enceladus {

struct GasColumnResult {
    double phi_to_zero;
    double rho_to_zero;
    double mach_top;
    double phi_top;
    double rho_top;
    double phi0;
};

struct RResult {
    double r;
    double phi_top;
    double rho_top;
    double phi0;
};

namespace detail {
constexpr double RG = 8.341 / 0.018;
constexpr double SIGMA = 5.67e-8;
}  // namespace detail

// Integrate the gas column for a given (Tb, depth, width, r).
inline GasColumnResult solve_function(
    double Tb, double depth, double width, double r, double kt = 2.4,
    double lv = 2.8e6, double g = 0.113, double Te = 68.0,
    double dz_step = 0.1, const std::string& friction_model = "constant",
    double Cf_constant = 0.002, double mu_vapor = 8.0e-6,
    double roughness = 0.0, double C_lam = 96.0) {
    const double rg = detail::RG;
    const double ec = vapor_pressure(273.15) / std::sqrt(2.0 * M_PI * rg);
    const double bv = lv / rg;

    const double phi = (1.0 - r) * evaporation_rate_simple(ec, bv, Tb);
    const double phi0 = phi;
    const double rho0 = vapor_pressure(Tb) / (rg * Tb) * r;

    // z_f = np.arange(0, depth + dz_step, dz_step); z_f[i] == i * dz_step.
    const long nz = static_cast<long>(std::ceil((depth + dz_step) / dz_step));
    std::vector<double> z_f(nz);
    for (long i = 0; i < nz; ++i) z_f[i] = static_cast<double>(i) * dz_step;

    std::vector<double> f(nz, 0.0);
    double phi_to_zero = depth;
    bool flag = false;
    double f_now = 0.0;

    for (long i = 1; i < nz; ++i) {
        double d_from_top = depth - z_f[i];
        if (d_from_top < 1e-10) d_from_top = 1e-10;
        const double ev_now =
            find_evap_surface(Tb, Te, detail::SIGMA, d_from_top, kt, lv) / width;
        f_now += ev_now * (z_f[i] - z_f[i - 1]);
        f[i] = f_now;
        if ((phi + f_now < 0.0) && !flag) {
            phi_to_zero = z_f[i];
            flag = true;
        }
    }

    auto Cf = [&](double v_mag, double dn) {
        return fanning_friction_factor(friction_model, v_mag, width, dn,
                                       mu_vapor, roughness, Cf_constant, C_lam);
    };

    const double sound = std::sqrt(1.33 * rg * Tb);
    std::vector<double> rho(nz, 0.0);
    std::vector<double> M(nz, 0.0);
    rho[0] = rho0;
    M[0] = phi / rho0 / sound;
    double rho_to_zero = depth;

    for (long i = 1; i < nz; ++i) {
        const double dz = z_f[i] - z_f[i - 1];
        const double dn = rho[i - 1];
        const double v = (phi + f[i]) / dn;
        M[i] = v / sound;
        const double Cd = Cf(v, dn);
        const double k1 = (2.0 * Cd * dn * v * v / width + dn * g) / (v * v - rg * Tb);

        double k2, k3, k4;
        if (i < nz - 1) {
            const double dn2 = rho[i - 1] + dz * 0.5 * k1;
            const double v2 = (phi + 0.5 * (f[i] + f[i + 1])) / dn2;
            const double Cd2 = Cf(v2, dn2);
            k2 = (2.0 * Cd2 * dn2 * v2 * v2 / width + dn2 * g) / (v2 * v2 - rg * Tb);

            const double dn3 = rho[i - 1] + dz * 0.5 * k2;
            const double v3 = (phi + 0.5 * (f[i] + f[i + 1])) / dn3;
            const double Cd3 = Cf(v3, dn3);
            k3 = (2.0 * Cd3 * dn3 * v3 * v3 / width + dn3 * g) / (v3 * v3 - rg * Tb);

            const double dn4 = rho[i - 1] + dz * k3;
            const double v4 = (phi + f[i + 1]) / dn4;
            const double Cd4 = Cf(v4, dn4);
            k4 = (2.0 * Cd4 * dn4 * v4 * v4 / width + dn4 * g) / (v4 * v4 - rg * Tb);
        } else {
            k2 = k1;
            k3 = k2;
            k4 = k3;
        }

        rho[i] = rho[i - 1] + dz / 6.0 * (k1 + k4 + 2.0 * k2 + 2.0 * k3);

        if (rho[i] < 0.0 || (rho[i] - rho[i - 1] > 0.0)) {
            rho_to_zero = z_f[i];
            break;
        }
    }

    GasColumnResult res{phi_to_zero, rho_to_zero, 0.0, 0.0, 0.0, phi0};
    if (rho[nz - 1] > 0.0 && (phi + f[nz - 1]) > 0.0) {
        res.mach_top = M[nz - 1];
        res.phi_top = phi + f[nz - 1];
        res.rho_top = rho[nz - 1];
    }
    return res;
}

// Bisection to find r for given (Tb, depth, width).
inline RResult solve_r_function(double Tb, double depth, double width,
                                double tol = 1e-4) {
    double r_l = 1e-5;
    double r_r = 1.0 - 1e-5;
    double phi_l = 0.0, phi_r = 0.0, rho_l = 0.0, rho_r = 0.0;
    double phi0_l = 0.0, phi0_r = 0.0;

    while (std::fabs(r_l - r_r) > tol) {
        const double r_m = 0.5 * (r_l + r_r);
        const GasColumnResult g = solve_function(Tb, depth, width, r_m);

        if (g.mach_top == 0.0) {
            if (g.rho_to_zero < g.phi_to_zero) {
                r_l = r_m;
                phi_l = g.phi_top; rho_l = g.rho_top; phi0_l = g.phi0;
            } else {
                r_r = r_m;
                phi_r = g.phi_top; rho_r = g.rho_top; phi0_r = g.phi0;
            }
        } else {
            if (g.mach_top > 1.6) {
                r_l = r_m;
                phi_l = g.phi_top; rho_l = g.rho_top; phi0_l = g.phi0;
            } else {
                r_r = r_m;
                phi_r = g.phi_top; rho_r = g.rho_top; phi0_r = g.phi0;
            }
        }
    }

    return RResult{0.5 * (r_l + r_r), 0.5 * (phi_l + phi_r),
                   0.5 * (rho_l + rho_r), 0.5 * (phi0_l + phi0_r)};
}

}  // namespace enceladus
