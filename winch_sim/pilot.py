"""Pilot model: attitude reference + PID on the elevator with a first-order stick lag."""

import jax
import jax.numpy as jnp

from .params import Pilot
from .smooth import smoothstep, soft_clip


def attitude_reference(p: Pilot, theta_rest, height, V_ias, beta, attached):
    """Pitch attitude the pilot aims for, as a smooth function of the flight state.

    ground roll -> gradual rotation with height -> climb attitude (trimmed with the
    indicated-airspeed error) -> lower the nose as the glider gets high over the winch;
    after release: glide attitude.
    """
    s_rot = smoothstep((height - p.h_rot0) / (p.h_rot1 - p.h_rot0))
    s_top = smoothstep((beta - p.top_beta0) / (p.top_beta1 - p.top_beta0))
    # Too fast -> raise the nose (active as soon as airborne), too slow -> lower it.
    speed_trim = p.K_V * soft_clip(V_ias - p.V_target, -4.0, 2.0, 1.0)
    speed_trim = speed_trim * airborne_factor(height)
    theta_ground = theta_rest + p.theta_liftoff
    theta_climb = theta_ground + s_rot * (p.theta_climb - theta_ground) + speed_trim
    theta_launch = (1.0 - s_top) * theta_climb + s_top * p.theta_top
    return attached * theta_launch + (1.0 - attached) * p.theta_glide


def elevator_rates(p: Pilot, theta_ref, theta, q, de, e_int, airborne):
    """Time derivatives of the elevator deflection and the attitude-error integral.

    Negative elevator (trailing edge up) pitches nose-up.  The integrator only runs
    once airborne (no wind-up while the tail is on the ground).
    """
    err = theta_ref - theta
    cmd = -(p.Kp * err + p.Ki * e_int) + p.Kd * q
    de_cmd = p.de_max * jnp.tanh(cmd / p.de_max)
    de_dot = (de_cmd - de) / p.tau
    e_dot = airborne * err - (1.0 - airborne) * e_int
    return de_dot, e_dot


def airborne_factor(height):
    return jax.nn.sigmoid((height - 0.5) / 0.1)
