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

import equinox as eqx
from jax import Array


class Glider(eqx.Module):
    """Longitudinal rigid-body and aerodynamic model of a glider."""

    mass: float | Array  # launch mass [kg]
    S: float | Array  # wing area [m^2]
    chord: float | Array  # mean aerodynamic chord [m]
    I_yy: float | Array  # pitch moment of inertia about the CG [kg m^2]

    # Lift: C_L = CL0 + CLa*alpha + CLde*de + CLq*q_hat, softly limited to [CLmin, CLmax]
    CL0: float | Array
    CLa: float | Array
    CLde: float | Array
    CLq: float | Array
    CLmax: float | Array
    CLmin: float | Array

    # Drag polar: C_D = CD0 + k_ind*C_L^2 (+ post-stall drag)
    CD0: float | Array
    k_ind: float | Array

    # Pitching moment about the CG: C_m = Cm0 + Cma*alpha + Cmq*q_hat + Cmde*de
    Cm0: float | Array
    Cma: float | Array
    Cmq: float | Array
    Cmde: float | Array

    # Tow hook (winch / CG hook) position relative to the CG, body frame [m]
    hook_x: float | Array
    hook_z: float | Array

    # Ground contact points (nose skid, main wheel, tail wheel/skid), body frame
    contact_x: Array  # shape (3,)
    contact_z: Array  # shape (3,)
    contact_k: Array  # spring stiffness [N/m]
    contact_c: Array  # damping [N s/m]
    contact_mu: Array  # friction coefficient

    # Operating limits
    V_W: float | Array  # max winch-launch airspeed [m/s]
    weak_link: float | Array  # weak link breaking load [N]


class Rope(eqx.Module):
    """Winch cable: lumped-mass elastic rope with weight and aerodynamic drag."""

    diameter: float | Array  # [m]
    mu: float | Array  # mass per unit length [kg/m]
    EA: float | Array  # axial stiffness [N]
    zeta: float | Array  # internal damping ratio of a single segment's axial mode
    CDn: float | Array  # normal (cross-flow) drag coefficient
    Cf: float | Array  # tangential skin-friction coefficient
    ground_mu: float | Array  # rope-on-grass friction coefficient
    ground_sink: float | Array  # static penetration used for the ground spring [m]
    # Rope end at the glider: drogue parachute + strop + weak link assembly
    end_mass: float | Array  # [kg]
    end_CdA: float | Array  # drag area of the (trailing, closed) parachute [m^2]


class Winch(eqx.Module):
    """Winch model: available rope pull F(v) acting on an effective drum mass.

    F_avail(v) = P_max / sqrt(v^2 + (P_max/F_max)^2) is a smooth version of
    min(F_max, P_max/v).  With P_max -> infinity the winch delivers a prescribed
    tension F_max (an ideal tension-controlled winch); with finite P_max it follows a
    power-limited engine curve.  The pull is scaled by the driver's throttle schedule.
    """

    F_max: float | Array  # max pull at low speed (torque limit / drum radius) [N]
    P_max: float | Array  # max power at the drum [W]
    throttle: float | Array  # driver's throttle setting (0..1)
    M_eff: float | Array  # effective mass of drum + drivetrain at the rope [kg]
    b_fric: float | Array  # viscous drum loss [N s/m]
    t_start: float | Array  # time the driver starts to pull [s]
    t_ramp: float | Array  # duration of the throttle ramp-up [s]
    # Near the top, the driver reduces power as the glider elevation (seen from the
    # winch) goes from fade_beta0 to fade_beta1, down to fade_floor * full power.
    fade_beta0: float | Array
    fade_beta1: float | Array
    fade_floor: float | Array
    height: float | Array  # height of the drum / rope exit [m]


class Pilot(eqx.Module):
    """Pilot model: pitch-attitude PID with a height/elevation dependent reference."""

    theta_climb: float | Array  # full-climb pitch attitude [rad]
    theta_liftoff: float | Array  # back-stick on the ground roll: attitude above rest
    h_rot0: float | Array  # height at which rotation into the climb starts [m]
    h_rot1: float | Array  # height at which full climb attitude is reached [m]
    V_target: float | Array  # target climb airspeed [m/s]
    K_V: float | Array  # attitude trim per airspeed error [rad/(m/s)]
    theta_top: float | Array  # attitude the pilot lowers the nose to at the top [rad]
    top_beta0: float | Array  # glider elevation where lowering the nose starts [rad]
    top_beta1: float | Array  # ... and where it is complete [rad]
    theta_glide: float | Array  # attitude after release [rad]
    Kp: float | Array  # elevator per attitude error [rad/rad]
    Ki: float | Array  # [rad/(rad s)]
    Kd: float | Array  # elevator per pitch rate [rad/(rad/s)]
    tau: float | Array  # pilot/stick lag [s]
    de_max: float | Array  # elevator deflection limit [rad]
    release_angle: float | Array  # pilot releases at this local cable angle [rad]
    back_release_angle: float | Array  # hook back-release (cable vs. body x) [rad]


class Env(eqx.Module):
    rho: float | Array = 1.225  # air density [kg/m^3]
    g: float | Array = 9.81  # [m/s^2]
    wind_ref: float | Array = 0.0  # headwind at 10 m height [m/s]
    wind_exp: float | Array = 1.0 / 7.0  # power-law shear exponent


class Launch(eqx.Module):
    """Everything needed to simulate one launch."""

    glider: Glider
    rope: Rope
    winch: Winch
    pilot: Pilot
    env: Env
    rope_length: float | Array  # length of rope laid out on the runway [m]
    slack: float | Array  # relative slack in the laid-out rope
