"""Glider catalogue.

The numbers are *approximate* values assembled from typical flight-manual / type
certificate data (span, area, typical launch mass, best L/D, placarded winch speed) and
generic estimates for everything a flight manual does not give (lift slope, pitch
derivatives, inertia, hook and wheel geometry).  They are good enough to show the
differences between classes of gliders, not to certify anything.
"""

import math

import jax.numpy as jnp

from .params import Glider, Pilot


def make_glider(
    *,
    mass: float,
    span: float,
    area: float,
    LD_max: float,
    CLmax: float,
    V_W_kmh: float,
    weak_link_daN: float,
    radius_of_gyration: float,
    oswald: float = 0.85,
    CL0: float = 0.35,
    static_margin: float = 0.15,
    tail_arm: float = 4.5,
    tail_volume: float = 0.5,
    hook_x: float = 0.25,
    hook_z: float = -0.45,
    wheel_x: float = 0.2,
    wheel_z: float = -0.8,
    tail_x: float = -5.0,
    tail_z: float = -0.4,
) -> Glider:
    """Build a Glider from handbook-style data plus generic aerodynamic estimates."""
    AR = span**2 / area
    chord = area / span
    k_ind = 1.0 / (math.pi * oswald * AR)
    # (L/D)max = 1 / (2 sqrt(CD0 k))  ->  CD0 = 1 / (4 k (L/D)^2)
    CD0 = 1.0 / (4.0 * k_ind * LD_max**2)
    # Helmbold lift slope of the wing, +8% for the tail's contribution
    CLa_wing = 2 * math.pi * AR / (2 + math.sqrt(AR**2 + 4))
    CLa = 1.08 * CLa_wing
    CLa_tail = 4.0  # per rad, low aspect ratio tailplane incl. downwash
    lt_c = tail_arm / chord
    Cma = -CLa * static_margin
    Cmq = -2.0 * CLa_tail * tail_volume * lt_c * 0.9
    Cmde = -CLa_tail * tail_volume * 0.5  # elevator effectiveness tau ~ 0.5
    CLde = CLa_tail * tail_volume / lt_c * 0.5
    CLq = 2.0 * CLa_tail * tail_volume * 0.9
    # Trim at best-glide C_L with neutral elevator
    CL_bg = math.sqrt(CD0 / k_ind)
    alpha_trim = (CL_bg - CL0) / CLa
    Cm0 = -Cma * alpha_trim
    return Glider(
        mass=mass,
        S=area,
        chord=chord,
        I_yy=mass * radius_of_gyration**2,
        CL0=CL0,
        CLa=CLa,
        CLde=CLde,
        CLq=CLq,
        CLmax=CLmax,
        CLmin=-0.6,
        CD0=CD0,
        k_ind=k_ind,
        Cm0=Cm0,
        Cma=Cma,
        Cmq=Cmq,
        Cmde=Cmde,
        hook_x=hook_x,
        hook_z=hook_z,
        # nose skid, main wheel, tail wheel/skid
        contact_x=jnp.array([1.8, wheel_x, tail_x]),
        contact_z=jnp.array([-0.55, wheel_z, tail_z]),
        contact_k=jnp.array([4e4, 25.0 * mass * 9.81, 4e4]),
        contact_c=jnp.array([2e3, 20.0 * mass, 2e3]),
        contact_mu=jnp.array([0.4, 0.05, 0.3]),
        V_W=V_W_kmh / 3.6,
        weak_link=weak_link_daN * 10.0,
    )


def rest_attitude(g: Glider):
    """Pitch attitude when resting on main wheel and tail (tail-down)."""
    dz = g.contact_z[2] - g.contact_z[1]
    dx = g.contact_x[1] - g.contact_x[2]
    return jnp.arctan2(dz, dx)


def stall_speed(g: Glider, rho: float = 1.225, gravity: float = 9.81):
    return jnp.sqrt(2 * g.mass * gravity / (rho * g.S * g.CLmax))


def make_pilot(
    *,
    V_target_kmh: float,
    theta_climb_deg: float = 40.0,
    **overrides: float,
) -> Pilot:
    d = math.radians
    kwargs: dict[str, float] = {
        "theta_climb": d(theta_climb_deg),
        "theta_liftoff": d(3.0),
        "h_rot0": 2.0,
        "h_rot1": 30.0,
        "V_target": V_target_kmh / 3.6,
        "K_V": d(2.5),  # 2.5 deg nose-up per m/s too fast
        "theta_top": d(5.0),
        "top_beta0": d(55.0),
        "top_beta1": d(70.0),
        "theta_glide": d(0.0),
        "Kp": 1.2,
        "Ki": 0.8,
        "Kd": 0.6,
        "tau": 0.25,
        "de_max": d(25.0),
        "release_angle": d(72.0),
        "back_release_angle": d(110.0),
    }
    kwargs.update(overrides)
    return Pilot(**kwargs)


# name -> (glider, pilot)
CATALOGUE: dict[str, tuple[Glider, Pilot]] = {
    # Vintage single seater, light and slow
    "Ka 8": (
        make_glider(
            mass=290.0,
            span=15.0,
            area=14.15,
            LD_max=27.0,
            CLmax=1.3,
            V_W_kmh=100.0,
            weak_link_daN=500.0,
            radius_of_gyration=1.1,
        ),
        make_pilot(V_target_kmh=85.0, theta_climb_deg=35.0),
    ),
    # Club-class 15 m single seater
    "LS4": (
        make_glider(
            mass=360.0,
            span=15.0,
            area=10.5,
            LD_max=40.0,
            CLmax=1.4,
            V_W_kmh=130.0,
            weak_link_daN=600.0,
            radius_of_gyration=1.15,
        ),
        make_pilot(V_target_kmh=105.0),
    ),
    # Two-seat trainer
    "ASK 21": (
        make_glider(
            mass=470.0,
            span=17.0,
            area=17.95,
            LD_max=34.0,
            CLmax=1.35,
            V_W_kmh=150.0,
            weak_link_daN=1000.0,
            radius_of_gyration=1.35,
            hook_x=0.35,
        ),
        make_pilot(V_target_kmh=105.0),
    ),
    # 18 m flapped racer, dry
    "ASG 29": (
        make_glider(
            mass=420.0,
            span=18.0,
            area=10.5,
            LD_max=52.0,
            CLmax=1.45,
            V_W_kmh=150.0,
            weak_link_daN=850.0,
            radius_of_gyration=1.2,
        ),
        make_pilot(V_target_kmh=115.0),
    ),
    # 20 m two-seater, heavy
    "DG-1000": (
        make_glider(
            mass=620.0,
            span=20.0,
            area=17.53,
            LD_max=46.0,
            CLmax=1.4,
            V_W_kmh=150.0,
            weak_link_daN=1000.0,
            radius_of_gyration=1.4,
            hook_x=0.35,
        ),
        make_pilot(V_target_kmh=115.0),
    ),
}
