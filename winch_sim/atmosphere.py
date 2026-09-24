"""International Standard Atmosphere (ISA) with a temperature offset.

Pressure follows the ISA pressure-height relation; the temperature is the ISA value
plus `Env.isa_dT` (a hot day has lower density at the same height).  Valid through
the troposphere and the lower stratosphere (isothermal above 11 km).
"""

import jax.numpy as jnp

from .params import Env

R_AIR = 287.05287  # specific gas constant of dry air [J/(kg K)]
G0 = 9.80665  # standard gravity used by the ISA definition [m/s^2]
T0 = 288.15  # sea-level temperature [K]
P0 = 101325.0  # sea-level pressure [Pa]
LAPSE = 0.0065  # troposphere lapse rate [K/m]
H_TROPO = 11000.0  # tropopause [m]
RHO0 = P0 / (R_AIR * T0)  # 1.225 kg/m^3, the reference for indicated airspeed

_EXP = G0 / (R_AIR * LAPSE)
_T11 = T0 - LAPSE * H_TROPO
_P11 = P0 * (_T11 / T0) ** _EXP


def isa(env: Env, z):
    """(temperature [K], pressure [Pa], density [kg/m^3]) at height z above the field."""
    h = env.field_elevation + z
    h_t = jnp.minimum(h, H_TROPO)
    T_std = T0 - LAPSE * h_t
    p_tropo = P0 * (T_std / T0) ** _EXP
    p_strat = _P11 * jnp.exp(-G0 * (h - H_TROPO) / (R_AIR * _T11))
    p = jnp.where(h <= H_TROPO, p_tropo, p_strat)
    T = T_std + env.isa_dT
    return T, p, p / (R_AIR * T)


def density(env: Env, z):
    return isa(env, z)[2]


def indicated_airspeed(env: Env, z, V_true):
    """Indicated (= equivalent) airspeed: same dynamic pressure at sea-level density.

    Instrument/position errors and compressibility are neglected (fine below ~300 km/h).
    """
    return V_true * jnp.sqrt(density(env, z) / RHO0)
