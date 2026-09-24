"""Parameter containers.

All parameter sets are `equinox.Module`s holding only float (or float-array) leaves,
so they are JAX pytrees: they can be passed through `jit`, stacked and `vmap`-ed over
(e.g. a batch of gliders) and differentiated with `jax.grad`.

Conventions (see docs/physics.md):
  * earth frame: x horizontal, pointing from the glider's start towards the winch; z up
  * body frame: x forward along the fuselage, z up (towards the canopy)
  * pitch angle theta is positive nose-up, the pitching moment positive nose-up
  * SI units throughout (m, s, kg, N, rad)
"""

import dataclasses
import math

import equinox as eqx
from jax import Array


def unit(u: str, default=dataclasses.MISSING, display: str | None = None):
    """Dataclass field carrying its SI unit as metadata ('-' = dimensionless).

    `display` is the unit used when reporting the parameter (e.g. "mm" for a rope
    diameter stored in m); see DISPLAY_UNITS.
    """
    meta = {"unit": u, "display": display or DEFAULT_DISPLAY.get(u, u)}
    return eqx.field(default=default, metadata=meta)


# display unit -> size of one display unit in SI
DISPLAY_UNITS = {
    "mm": 1e-3,
    "kN": 1e3,
    "kW": 1e3,
    "km/h": 1 / 3.6,
    "deg": math.pi / 180,
}
DEFAULT_DISPLAY = {"rad": "deg", "W": "kW"}


class Glider(eqx.Module):
    """Longitudinal rigid-body and aerodynamic model of a glider."""

    mass: float | Array = unit("kg")  # launch mass
    S: float | Array = unit("m^2")  # wing area
    chord: float | Array = unit("m")  # mean aerodynamic chord
    I_yy: float | Array = unit("kg m^2")  # pitch moment of inertia about the CG

    # Lift: C_L = CL0 + CLa*alpha + CLde*de + CLq*q_hat, softly limited to [CLmin, CLmax]
    CL0: float | Array = unit("-")
    CLa: float | Array = unit("1/rad")
    CLde: float | Array = unit("1/rad")
    CLq: float | Array = unit("-")
    CLmax: float | Array = unit("-")
    CLmin: float | Array = unit("-")

    # Drag polar: C_D = CD0 + k_ind*C_L^2 (+ post-stall drag)
    CD0: float | Array = unit("-")
    k_ind: float | Array = unit("-")

    # Pitching moment about the CG: C_m = Cm0 + Cma*alpha + Cmq*q_hat + Cmde*de
    Cm0: float | Array = unit("-")
    Cma: float | Array = unit("1/rad")
    Cmq: float | Array = unit("-")
    Cmde: float | Array = unit("1/rad")

    # Tow hook (winch / CG hook) position relative to the CG, body frame [m]
    hook_x: float | Array = unit("m")
    hook_z: float | Array = unit("m")

    # Ground contact points (nose skid, main wheel, tail wheel/skid), body frame
    contact_x: Array = unit("m")  # x of nose skid, main wheel, tail
    contact_z: Array = unit("m")  # z of nose skid, main wheel, tail
    contact_k: Array = unit("N/m")  # spring stiffness
    contact_c: Array = unit("N s/m")  # damping
    contact_mu: Array = unit("-")  # friction coefficient

    # Operating limits
    V_W: float | Array = unit("m/s", display="km/h")  # max winch-launch speed (IAS)
    weak_link: float | Array = unit("N", display="kN")  # weak link breaking load


class Rope(eqx.Module):
    """Winch cable: lumped-mass elastic rope with weight and aerodynamic drag."""

    diameter: float | Array = unit("m", display="mm")  # rope diameter
    mu: float | Array = unit("kg/m")  # mass per unit length
    EA: float | Array = unit("N", display="kN")  # axial stiffness
    breaking_load: float | Array = unit("N", display="kN")  # minimum breaking load
    zeta: float | Array = unit("-")  # damping ratio of one segment's axial mode
    CDn: float | Array = unit("-")  # normal (cross-flow) drag coefficient
    Cf: float | Array = unit("-")  # tangential skin-friction coefficient
    ground_mu: float | Array = unit("-")  # rope-on-grass friction coefficient
    ground_sink: float | Array = unit(
        "m"
    )  # static penetration used for the ground spring
    # Rope end at the glider: drogue parachute + strop + weak link assembly
    end_mass: float | Array = unit("kg")
    end_CdA: float | Array = unit(
        "m^2"
    )  # drag area of the (trailing, closed) parachute


class Winch(eqx.Module):
    """Winch model: available rope pull F(v) acting on an effective drum mass.

    F_avail(v) = P_max / sqrt(v^2 + (P_max/F_max)^2) is a smooth version of
    min(F_max, P_max/v).  With P_max -> infinity the winch delivers a prescribed
    tension F_max (an ideal tension-controlled winch); with finite P_max it follows a
    power-limited engine curve.  The pull is scaled by the driver's throttle schedule.
    """

    F_max: float | Array = unit(
        "N", display="kN"
    )  # max pull at low speed (torque limit / drum radius)
    P_max: float | Array = unit("W")  # max power at the drum
    throttle: float | Array = unit("-")  # driver's throttle setting (0..1)
    M_eff: float | Array = unit("kg")  # effective mass of drum + drivetrain at the rope
    b_fric: float | Array = unit("N s/m")  # viscous drum loss
    t_start: float | Array = unit("s")  # time the driver starts to pull
    t_ramp: float | Array = unit("s")  # duration of the throttle ramp-up
    # Near the top, the driver reduces power as the glider elevation (seen from the
    # winch) goes from fade_beta0 to fade_beta1, down to fade_floor * full power.
    fade_beta0: float | Array = unit("rad")
    fade_beta1: float | Array = unit("rad")
    fade_floor: float | Array = unit("-")
    height: float | Array = unit("m")  # height of the drum / rope exit


class Pilot(eqx.Module):
    """Pilot model: pitch-attitude PID with a height/elevation dependent reference."""

    theta_climb: float | Array = unit("rad")  # full-climb pitch attitude
    theta_liftoff: float | Array = unit(
        "rad"
    )  # back-stick on the ground roll: attitude above rest
    h_rot0: float | Array = unit("m")  # height at which rotation into the climb starts
    h_rot1: float | Array = unit("m")  # height at which full climb attitude is reached
    V_target: float | Array = unit("m/s", display="km/h")  # target climb speed (IAS)
    K_V: float | Array = unit("rad s/m")  # attitude trim per airspeed error
    theta_top: float | Array = unit(
        "rad"
    )  # attitude the pilot lowers the nose to at the top
    top_beta0: float | Array = unit(
        "rad"
    )  # glider elevation where lowering the nose starts
    top_beta1: float | Array = unit("rad")  # ... and where it is complete
    theta_glide: float | Array = unit("rad")  # attitude after release
    Kp: float | Array = unit("-")  # elevator per attitude error
    Ki: float | Array = unit("1/s")  # integral gain on the attitude error
    Kd: float | Array = unit("s")  # elevator per pitch rate
    tau: float | Array = unit("s")  # pilot/stick lag
    de_max: float | Array = unit("rad")  # elevator deflection limit
    release_angle: float | Array = unit(
        "rad"
    )  # pilot releases at this local cable angle
    back_release_angle: float | Array = unit(
        "rad"
    )  # hook back-release (cable vs. body x)


class Env(eqx.Module):
    field_elevation: float | Array = unit("m", 0.0)  # airfield height above MSL
    isa_dT: float | Array = unit("K", 0.0)  # temperature offset from ISA
    g: float | Array = unit("m/s^2", 9.81)  # gravitational acceleration
    wind_ref: float | Array = unit("m/s", 0.0)  # headwind at 10 m height
    wind_exp: float | Array = unit("-", 1.0 / 7.0)  # power-law shear exponent


class Launch(eqx.Module):
    """Everything needed to simulate one launch."""

    glider: Glider
    rope: Rope
    winch: Winch
    pilot: Pilot
    env: Env
    rope_length: float | Array = unit("m")  # length of rope laid out on the runway
    slack: float | Array = unit("-")  # relative slack in the laid-out rope
