"""Winch models.

Both models share one equation for the reel-in speed at the drum,

    M_eff * dv_reel/dt = throttle(t, beta) * F_avail(v_reel) - T_winch - b * v_reel,

and differ only in F_avail:
  * tension-controlled winch: F_avail = F_max (P_max -> inf), i.e. the rope is pulled
    with a prescribed tension profile;
  * engine winch: power-limited curve F_avail ~ min(F_max, P_max / v_reel).
"""

import jax.numpy as jnp

from .params import Glider, Winch
from .smooth import smoothstep

G = 9.81


def available_pull(w: Winch, v_reel):
    return w.P_max / jnp.sqrt(v_reel**2 + (w.P_max / w.F_max) ** 2)


def throttle(w: Winch, t, beta):
    """Driver's throttle: ramp up after t_start, fade out as the glider gets high."""
    ramp = smoothstep((t - w.t_start) / w.t_ramp)
    fade = 1.0 - (1.0 - w.fade_floor) * smoothstep(
        (beta - w.fade_beta0) / (w.fade_beta1 - w.fade_beta0)
    )
    return w.throttle * ramp * fade


def reel_acceleration(w: Winch, t, beta, v_reel, T_winch, attached):
    F = throttle(w, t, beta) * available_pull(w, v_reel)
    a_launch = (F - T_winch - w.b_fric * v_reel) / w.M_eff
    # After release the driver brakes the drum and lets the rope fall.
    a_brake = -v_reel / 2.0
    return attached * a_launch + (1.0 - attached) * a_brake


def _defaults(**kw) -> dict:
    d = {
        "b_fric": 20.0,
        "t_start": 1.0,
        "t_ramp": 3.0,
        "fade_beta0": jnp.deg2rad(50.0),
        "fade_beta1": jnp.deg2rad(70.0),
        "fade_floor": 0.4,
        "height": 1.0,
    }
    d.update(kw)
    return d


def tension_winch(glider: Glider, tension_per_weight: float = 1.1, **kw) -> Winch:
    """Ideal tension-controlled winch: pulls with tension_per_weight * glider weight."""
    return Winch(
        F_max=tension_per_weight * glider.mass * G,
        P_max=1e12,
        throttle=1.0,
        M_eff=60.0,
        **_defaults(**kw),
    )


def engine_winch(
    glider: Glider,
    P_max: float = 200e3,
    F_max: float = 12e3,
    pull_per_weight: float = 1.1,
    v_design: float = 15.0,
    **kw,
) -> Winch:
    """Power-limited engine winch.

    The driver sets the throttle so that the pull at the design reel speed equals
    pull_per_weight * glider weight (heavier glider -> more throttle).  At lower
    reel speeds the engine pulls harder, at higher speeds less.
    """
    w = Winch(
        F_max=F_max,
        P_max=P_max,
        throttle=1.0,
        M_eff=250.0,
        **_defaults(**kw),
    )
    thr = pull_per_weight * glider.mass * G / available_pull(w, v_design)
    return Winch(
        F_max=F_max,
        P_max=P_max,
        throttle=jnp.minimum(thr, 1.0),
        M_eff=250.0,
        **_defaults(**kw),
    )
