"""Darcy-Weisbach friction factor models for parallel-plate crack geometry.

The crack is modelled as two infinite parallel plates separated by a gap *w*.
The hydraulic diameter is  D_h = 2 * w.

Three models are provided:

* **constant** -- fixed Fanning friction coefficient (legacy behaviour).
* **laminar**  -- f_D = C_lam / Re_Dh  (C_lam = 96 for parallel plates).
* **churchill** -- Churchill (1977) single-equation model that blends
  laminar, transition, and turbulent regimes smoothly.  In the turbulent
  limit it converges to the Colebrook-White implicit equation.

Convention
----------
The *Darcy* friction factor ``f_D`` relates to the Fanning coefficient by
``C_f = f_D / 8``.  The wall-friction force per unit mass per unit length
used in the momentum equation is

    F_fric = (f_D / D_h) * (v^2 / 2)  =  (f_D / (4 w)) * v^2

which is equivalent to the original formulation  ``2 * C_f / w * v^2``
since  ``2 * (f_D/8) / w = f_D / (4w)``.
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------
# Hydraulic diameter
# -----------------------------------------------------------------------

def hydraulic_diameter(w: float) -> float:
    """Hydraulic diameter for infinite parallel plates with gap *w*."""
    return 2.0 * w


# -----------------------------------------------------------------------
# Reynolds number
# -----------------------------------------------------------------------

def reynolds_number(rho: float, v: float, D_h: float, mu: float) -> float:
    """Bulk Reynolds number based on hydraulic diameter.

    Parameters
    ----------
    rho : fluid density (kg/m^3)
    v   : bulk velocity magnitude (m/s)
    D_h : hydraulic diameter (m)
    mu  : dynamic viscosity (Pa*s)
    """
    if mu <= 0 or D_h <= 0:
        return 0.0
    return rho * abs(v) * D_h / mu


# -----------------------------------------------------------------------
# Darcy friction factor models
# -----------------------------------------------------------------------

def darcy_constant(Cf: float) -> float:
    """Convert a constant Fanning coefficient to a Darcy factor."""
    return 8.0 * Cf


def darcy_laminar(Re: float, C_lam: float = 96.0) -> float:
    """Laminar Darcy factor for parallel plates: f_D = C_lam / Re.

    For a circular pipe C_lam = 64; for infinite parallel plates C_lam = 96.
    """
    if Re < 1e-12:
        return C_lam / 1e-12
    return C_lam / Re


def darcy_churchill(Re: float, roughness: float = 0.0,
                    D_h: float = 1.0, C_lam: float = 96.0) -> float:
    """Churchill (1977) friction factor -- smooth blend across all regimes.

    f_D = 8 * [ (C_lam/(8*Re))^12  +  (A + B)^(-3/2) ]^(1/12)

    where
        A = { -2.457 * ln[ (7/Re)^0.9 + 0.27 * (eps/D_h) ] }^16
        B = (37530 / Re)^16

    The original Churchill formula uses 8/Re in the laminar term (for
    circular pipes, C_lam=64).  Here the laminar term is generalised to
    C_lam/(8*Re) so that for parallel plates (C_lam=96) the correct
    laminar limit is recovered.

    Parameters
    ----------
    Re        : Reynolds number (based on D_h)
    roughness : absolute wall roughness (m), 0 = hydraulically smooth
    D_h       : hydraulic diameter (m)
    C_lam     : laminar-regime constant (96 for parallel plates)
    """
    if Re < 1e-12:
        return C_lam / 1e-12

    eps_over_D = roughness / D_h if D_h > 0 else 0.0

    lam_term = (C_lam / (8.0 * Re)) ** 12

    inner = (7.0 / Re) ** 0.9 + 0.27 * eps_over_D
    if inner < 1e-30:
        inner = 1e-30
    A = (-2.457 * np.log(inner)) ** 16
    B = (37530.0 / Re) ** 16
    turb_term = (A + B) ** (-1.5)

    return 8.0 * (lam_term + turb_term) ** (1.0 / 12.0)


# -----------------------------------------------------------------------
# Unified interface
# -----------------------------------------------------------------------

def fanning_friction_factor(
    model: str,
    v: float,
    w: float,
    rho: float = 1.0,
    mu: float = 1.0,
    roughness: float = 0.0,
    Cf_constant: float = 0.004,
    C_lam: float = 96.0,
) -> float:
    """Return the Fanning friction coefficient C_f for the given conditions.

    This is the value that plugs directly into the original formulation
    ``2 * C_f / w * v |v|``.

    Parameters
    ----------
    model       : one of ``"constant"``, ``"laminar"``, ``"churchill"``
    v           : velocity magnitude (m/s)
    w           : crack width / gap (m)
    rho         : fluid density (kg/m^3)
    mu          : dynamic viscosity (Pa*s)
    roughness   : absolute wall roughness (m)
    Cf_constant : constant Fanning coefficient (used when model="constant")
    C_lam       : laminar-regime constant (96 for parallel plates)
    """
    if model == "constant":
        return Cf_constant

    D_h = hydraulic_diameter(w)
    Re = reynolds_number(rho, v, D_h, mu)

    if model == "laminar":
        f_D = darcy_laminar(Re, C_lam)
    elif model == "churchill":
        f_D = darcy_churchill(Re, roughness, D_h, C_lam)
    else:
        raise ValueError(f"Unknown friction model: {model!r}")

    return f_D / 8.0
