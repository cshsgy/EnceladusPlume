"""Configuration loader -- reads YAML and exposes typed dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclass hierarchy
# ---------------------------------------------------------------------------

@dataclass
class PhysicalParams:
    gravity: float = 0.113
    thermal_diffusivity: float = 1.0e-6
    thermal_conductivity: float = 3.0
    latent_heat: float = 2.84e6
    effective_surface_temp: float = 68.0
    orbital_period: float = 118800.0
    equilibrium_depth: float = 20000.0
    gas_constant_vapor: float = 8.341 / 0.018
    stefan_boltzmann: float = 5.67e-8
    water_temperature: float = 273.15
    liquid_density: float = 1000.0
    liquid_viscosity: float = 1.8e-3
    vapor_viscosity: float = 8.0e-6


@dataclass
class FrictionParams:
    liquid_model: str = "churchill"
    gas_model: str = "churchill"
    liquid_Cf_constant: float = 0.004
    gas_Cf_constant: float = 0.002
    roughness: float = 0.0
    C_lam: float = 96.0


@dataclass
class LiquidDynamicsParams:
    npts_velocity: int = 1000
    ode_method: str = "DOP853"
    rtol: float = 1.0e-8
    atol: float = 1.0e-8
    max_step: float = 50.0
    n_periods: int = 4


@dataclass
class GasDynamicsParams:
    bisection_tol: float = 1.0e-4
    r_min: float = 1.0e-5
    r_max: float = 1.0 - 1.0e-5
    dz: float = 0.1
    constrained_max_change: float = 0.005
    mach_threshold: float = 1.6
    gamma_eff: float = 1.33


@dataclass
class HeatDiffusionParams:
    dx: float = 0.05
    max_x: float = 0.4
    n_vertical_levels: int = 100
    periods_to_run: int = 60


@dataclass
class InterpolationParams:
    n_grid: int = 20
    w_base: float = 0.2


@dataclass
class FullSolverParams:
    wmin: float = 0.01
    wmaxmin: float = 2.0
    depth: float = 20000.0


@dataclass
class ModularSolverParams:
    times_file: str = "Times.txt"
    slips_file: str = "Slip_Time_Functions.txt"


@dataclass
class Config:
    physical: PhysicalParams = field(default_factory=PhysicalParams)
    friction: FrictionParams = field(default_factory=FrictionParams)
    liquid_dynamics: LiquidDynamicsParams = field(default_factory=LiquidDynamicsParams)
    gas_dynamics: GasDynamicsParams = field(default_factory=GasDynamicsParams)
    heat_diffusion: HeatDiffusionParams = field(default_factory=HeatDiffusionParams)
    interpolation: InterpolationParams = field(default_factory=InterpolationParams)
    full_solver: FullSolverParams = field(default_factory=FullSolverParams)
    modular_solver: ModularSolverParams = field(default_factory=ModularSolverParams)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "default.yaml"
)


def _merge_dict(target: dict, source: dict) -> dict:
    """Recursively merge *source* into *target*, returning *target*."""
    for key, val in source.items():
        if isinstance(val, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], val)
        else:
            target[key] = val
    return target


def _dict_to_dataclass(cls, data: dict):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path : str or None
        Path to a YAML file.  If *None*, uses the built-in default.
        If a custom path is given it is merged on top of the defaults,
        so only changed values need to be specified.
    """
    with open(_DEFAULT_CONFIG_PATH, "r") as fh:
        cfg = yaml.safe_load(fh)

    if path is not None:
        with open(path, "r") as fh:
            overrides = yaml.safe_load(fh) or {}
        _merge_dict(cfg, overrides)

    return Config(
        physical=_dict_to_dataclass(PhysicalParams, cfg.get("physical", {})),
        friction=_dict_to_dataclass(FrictionParams, cfg.get("friction", {})),
        liquid_dynamics=_dict_to_dataclass(LiquidDynamicsParams, cfg.get("liquid_dynamics", {})),
        gas_dynamics=_dict_to_dataclass(GasDynamicsParams, cfg.get("gas_dynamics", {})),
        heat_diffusion=_dict_to_dataclass(HeatDiffusionParams, cfg.get("heat_diffusion", {})),
        interpolation=_dict_to_dataclass(InterpolationParams, cfg.get("interpolation", {})),
        full_solver=_dict_to_dataclass(FullSolverParams, cfg.get("full_solver", {})),
        modular_solver=_dict_to_dataclass(ModularSolverParams, cfg.get("modular_solver", {})),
    )
