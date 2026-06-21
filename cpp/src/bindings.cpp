// pybind11 bindings for the Enceladus C++ performance core.
//
// Builds the `_enceladus_core` extension module imported by the Python package
// (enceladus_plume) as an optional accelerator. Each binding mirrors the
// signature/return of its pure-Python counterpart so the two are drop-in
// compatible.
#include <pybind11/pybind11.h>

#include "enceladus/friction.hpp"
#include "enceladus/gas.hpp"
#include "enceladus/liquid.hpp"
#include "enceladus/physics.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_enceladus_core, m) {
    m.doc() = "Enceladus plume C++ performance core (pybind11).";

    // --- friction.hpp ------------------------------------------------------
    m.def("hydraulic_diameter", &enceladus::hydraulic_diameter, py::arg("w"));

    m.def("reynolds_number", &enceladus::reynolds_number, py::arg("rho"),
          py::arg("v"), py::arg("D_h"), py::arg("mu"));

    m.def("darcy_constant", &enceladus::darcy_constant, py::arg("Cf"));

    m.def("darcy_laminar", &enceladus::darcy_laminar, py::arg("Re"),
          py::arg("C_lam") = 96.0);

    m.def("darcy_churchill", &enceladus::darcy_churchill, py::arg("Re"),
          py::arg("roughness") = 0.0, py::arg("D_h") = 1.0,
          py::arg("C_lam") = 96.0);

    m.def("fanning_friction_factor", &enceladus::fanning_friction_factor,
          py::arg("model"), py::arg("v"), py::arg("w"), py::arg("rho") = 1.0,
          py::arg("mu") = 1.0, py::arg("roughness") = 0.0,
          py::arg("Cf_constant") = 0.004, py::arg("C_lam") = 96.0);

    // --- physics.hpp -------------------------------------------------------
    m.def("vapor_pressure", &enceladus::vapor_pressure, py::arg("T"));

    m.def("evaporation_rate_simple", &enceladus::evaporation_rate_simple,
          py::arg("ec"), py::arg("bv"), py::arg("Tm"));

    m.def("find_evap_surface", &enceladus::find_evap_surface, py::arg("Tw"),
          py::arg("Te"), py::arg("sigma"), py::arg("d"), py::arg("k"),
          py::arg("L"));

    // --- liquid.hpp --------------------------------------------------------
    // Returns (dvdt, dhdt) matching solver.py::_derivative.
    m.def(
        "liquid_derivative",
        [](double v, double h, double L, double g, double w, double dwdt,
           double dwdt2, long npts, const std::string& model,
           double Cf_constant, double rho, double mu, double roughness,
           double C_lam) {
            const auto d = enceladus::liquid_derivative(
                v, h, L, g, w, dwdt, dwdt2, npts, model, Cf_constant, rho, mu,
                roughness, C_lam);
            return py::make_tuple(d.dvdt, d.dhdt);
        },
        py::arg("v"), py::arg("h"), py::arg("L"), py::arg("g"), py::arg("w"),
        py::arg("dwdt"), py::arg("dwdt2"), py::arg("npts") = 1000,
        py::arg("model") = "constant", py::arg("Cf_constant") = 0.004,
        py::arg("rho") = 1000.0, py::arg("mu") = 1.8e-3,
        py::arg("roughness") = 0.0, py::arg("C_lam") = 96.0);

    // --- gas.hpp -----------------------------------------------------------
    // Returns (phi_to_zero, rho_to_zero, mach_top, phi_top, rho_top, phi0).
    m.def(
        "solve_function",
        [](double Tb, double depth, double width, double r, double kt,
           double lv, double g, double Te, double dz_step,
           const std::string& friction_model, double Cf_constant,
           double mu_vapor, double roughness, double C_lam) {
            const auto s = enceladus::solve_function(
                Tb, depth, width, r, kt, lv, g, Te, dz_step, friction_model,
                Cf_constant, mu_vapor, roughness, C_lam);
            return py::make_tuple(s.phi_to_zero, s.rho_to_zero, s.mach_top,
                                  s.phi_top, s.rho_top, s.phi0);
        },
        py::arg("Tb"), py::arg("depth"), py::arg("width"), py::arg("r"),
        py::arg("kt") = 2.4, py::arg("lv") = 2.84e6, py::arg("g") = 0.113,
        py::arg("Te") = 68.0, py::arg("dz_step") = 0.1,
        py::arg("friction_model") = "constant", py::arg("Cf_constant") = 0.002,
        py::arg("mu_vapor") = 8.0e-6, py::arg("roughness") = 0.0,
        py::arg("C_lam") = 96.0);

    // Returns (r, phi_top, rho_top, phi0) matching solve_r_function.
    m.def(
        "solve_r_function",
        [](double Tb, double depth, double width, double tol) {
            const auto s = enceladus::solve_r_function(Tb, depth, width, tol);
            return py::make_tuple(s.r, s.phi_top, s.rho_top, s.phi0);
        },
        py::arg("Tb"), py::arg("depth"), py::arg("width"),
        py::arg("tol") = 1e-4);
}
