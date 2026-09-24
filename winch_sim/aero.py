"""Glider aerodynamics and wind."""

import jax.numpy as jnp

from .params import Env, Glider
from .smooth import perp, safe_norm, soft_clip, softplus


def wind(env: Env, z):
    """Wind velocity vector (earth frame). A headwind blows towards -x."""
    zc = softplus(z, 0.5) + 0.1  # keep the power law finite at and below ground
    W = env.wind_ref * (zc / 10.0) ** env.wind_exp
    return jnp.stack([-W, jnp.zeros_like(W)], axis=-1)


def air_data(g: Glider, env: Env, pos, vel, theta):
    """Airspeed vector, airspeed, angle of attack and flight path angle."""
    va = vel - wind(env, pos[1])
    V = safe_norm(va, 1e-6)
    c, s = jnp.cos(theta), jnp.sin(theta)
    u = va[0] * c + va[1] * s  # along body x
    w = -va[0] * s + va[1] * c  # along body z (up)
    alpha = jnp.arctan2(-w, u + 1e-3)
    gamma = jnp.arctan2(va[1], va[0] + 1e-3)
    return va, V, alpha, gamma


def lift_coefficient(g: Glider, alpha, de, q_hat):
    CL_lin = g.CL0 + g.CLa * alpha + g.CLde * de + g.CLq * q_hat
    return soft_clip(CL_lin, g.CLmin, g.CLmax, 0.05)


def drag_coefficient(g: Glider, alpha, CL):
    alpha_stall = (g.CLmax - g.CL0) / g.CLa
    alpha_neg = (g.CLmin - g.CL0) / g.CLa
    beyond = softplus(alpha - alpha_stall, 0.02) + softplus(alpha_neg - alpha, 0.02)
    # post-stall flat-plate drag grows like ~1.3 sin^2(excess alpha)
    return g.CD0 + g.k_ind * CL**2 + 1.3 * jnp.sin(jnp.minimum(beyond, 1.5)) ** 2


def aero_forces(g: Glider, env: Env, pos, vel, theta, q, de):
    """Aerodynamic force (earth frame), pitching moment about the CG, and lift."""
    va, V, alpha, _ = air_data(g, env, pos, vel, theta)
    qbar = 0.5 * env.rho * V**2
    q_hat = q * g.chord / (2.0 * jnp.sqrt(V**2 + 1.0))
    CL = lift_coefficient(g, alpha, de, q_hat)
    CD = drag_coefficient(g, alpha, CL)
    alpha_m = soft_clip(alpha, -0.4, 0.4, 0.05)  # moment curve beyond stall: flat
    Cm = g.Cm0 + g.Cma * alpha_m + g.Cmq * q_hat + g.Cmde * de
    e_v = va / V
    F = qbar * g.S * (CL * perp(e_v) - CD * e_v)
    M = qbar * g.S * g.chord * Cm
    return F, M, qbar * g.S * CL
