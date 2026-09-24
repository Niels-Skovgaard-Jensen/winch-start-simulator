"""State, vector field and derived quantities of the coupled glider-rope-winch system."""

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from . import aero, cable, pilot, winch
from .atmosphere import density, indicated_airspeed
from .gliders import rest_attitude
from .ground import contact_force
from .params import Launch
from .smooth import cross2, perp, rotate


class State(eqx.Module):
    pos: Array  # (2,) glider CG position, earth frame
    vel: Array  # (2,) glider CG velocity, earth frame
    theta: Array  # pitch attitude
    q: Array  # pitch rate
    node_p: Array  # (N-1, 2) free rope node positions
    node_v: Array  # (N-1, 2) free rope node velocities
    L0: Array  # unstretched paid-out rope length
    v_reel: Array  # reel-in speed at the drum
    de: Array  # elevator deflection
    e_int: Array  # pilot attitude-error integral


class Args(eqx.Module):
    launch: Launch
    attached: Array  # 1.0 while on the cable, 0.0 after release
    theta_rest: Array
    z_rest: Array  # CG height at rest on the ground
    winch_pos: Array  # (2,)


def make_args(launch: Launch, attached=1.0) -> Args:
    g = launch.glider
    th = rest_attitude(g)
    c = rotate(th, jnp.stack([g.contact_x, g.contact_z], -1))
    z_rest = -c[1:, 1].min()
    winch_x = g.hook_x + launch.rope_length
    return Args(
        launch=launch,
        attached=jnp.asarray(attached, float),
        theta_rest=th,
        z_rest=z_rest,
        winch_pos=jnp.stack([jnp.asarray(winch_x, float), launch.winch.height]),
    )


def initial_state(args: Args, n_segments: int) -> State:
    """Glider at rest, tail down; rope laid out straight on the runway with slack."""
    L = args.launch
    th = args.theta_rest
    pos = jnp.array([0.0, args.z_rest])
    hook = pos + rotate(th, jnp.stack([L.glider.hook_x, L.glider.hook_z]))
    s = jnp.linspace(0.0, 1.0, n_segments + 1)[1:-1]
    x = args.winch_pos[0] + s * (hook[0] - args.winch_pos[0])
    z = jnp.full_like(x, -L.rope.ground_sink)
    dist = jnp.hypot(args.winch_pos[0] - hook[0], args.winch_pos[1] - hook[1])
    zero = jnp.asarray(0.0)
    return State(
        pos=pos,
        vel=jnp.zeros(2),
        theta=jnp.asarray(th, float),
        q=zero,
        node_p=jnp.stack([x, z], -1),
        node_v=jnp.zeros((n_segments - 1, 2)),
        L0=dist * (1.0 + L.slack),
        v_reel=zero,
        de=zero,
        e_int=zero,
    )


def _hook_kinematics(args: Args, y: State):
    g = args.launch.glider
    r_hook = rotate(y.theta, jnp.stack([g.hook_x, g.hook_z]))
    return y.pos + r_hook, y.vel + y.q * perp(r_hook), r_hook


def winch_elevation(args: Args, y: State):
    d = y.pos - args.winch_pos
    return jnp.arctan2(d[1], -d[0])


def evaluate(t, y: State, args: Args):
    """Compute state derivatives and all intermediate quantities."""
    L = args.launch
    g, env = L.glider, L.env
    attached = args.attached

    hook_pos, hook_vel, r_hook = _hook_kinematics(args, y)
    rf = cable.rope_forces(
        L.rope,
        env,
        args.winch_pos,
        hook_pos,
        hook_vel,
        y.node_p,
        y.node_v,
        y.L0,
        y.v_reel,
        attached,
    )

    F_aero, M_aero, lift = aero.aero_forces(g, env, y.pos, y.vel, y.theta, y.q, y.de)

    r_c = rotate(y.theta, jnp.stack([g.contact_x, g.contact_z], -1))
    v_c = y.vel + y.q * perp(r_c)
    F_c = contact_force(
        -(y.pos[1] + r_c[:, 1]), v_c, g.contact_k, g.contact_c, g.contact_mu
    )

    m_tot = g.mass + rf.end_mass
    weight = jnp.array([0.0, -1.0]) * (g.mass * env.g)  # end weight is in hook_force
    F_tot = F_aero + F_c.sum(0) + rf.hook_force + weight
    M_tot = M_aero + cross2(r_c, F_c).sum() + cross2(r_hook, rf.hook_force)

    # Pilot
    height = y.pos[1] - args.z_rest
    _, V_tas, alpha, gamma = aero.air_data(g, env, y.pos, y.vel, y.theta)
    V_ias = indicated_airspeed(env, y.pos[1], V_tas)  # what the pilot flies
    beta = winch_elevation(args, y)
    theta_ref = pilot.attitude_reference(
        L.pilot, args.theta_rest, height, V_ias, beta, attached
    )
    de_dot, e_dot = pilot.elevator_rates(
        L.pilot,
        theta_ref,
        y.theta,
        y.q,
        y.de,
        y.e_int,
        pilot.airborne_factor(height),
    )

    v_reel_dot = winch.reel_acceleration(
        L.winch, t, beta, y.v_reel, rf.tension[0], attached
    )

    dy = State(
        pos=y.vel,
        vel=F_tot / m_tot,
        theta=y.q,
        q=M_tot / g.I_yy,
        node_p=y.node_v,
        node_v=rf.node_force / rf.node_mass,
        L0=-y.v_reel,
        v_reel=v_reel_dot,
        de=de_dot,
        e_int=e_dot,
    )

    # Local cable direction at the hook (towards the next rope node)
    d_hook = y.node_p[-1] - hook_pos
    cable_angle = jnp.arctan2(-d_hook[1], d_hook[0])  # below horizontal
    cable_body = jnp.arctan2(
        -(d_hook[1] * jnp.cos(y.theta) - d_hook[0] * jnp.sin(y.theta)),
        d_hook[0] * jnp.cos(y.theta) + d_hook[1] * jnp.sin(y.theta),
    )  # angle of the cable below the glider's longitudinal axis
    F_nongrav = F_tot - weight
    n_z = jnp.dot(F_nongrav, perp(jnp.stack([jnp.cos(y.theta), jnp.sin(y.theta)])))
    info = {
        "t": t,
        "x": y.pos[0],
        "z": y.pos[1],
        "height": height,
        "V_tas": V_tas,
        "V_ias": V_ias,
        "rho": density(env, y.pos[1]),
        "alpha": alpha,
        "gamma": gamma,
        "theta": y.theta,
        "theta_ref": theta_ref,
        "de": y.de,
        "T_hook": rf.tension[-1],
        "T_winch": rf.tension[0],
        "T_max_rope": rf.tension.max(),
        "beta": beta,
        "cable_angle": cable_angle,
        "cable_body": cable_body,
        "n_z": n_z / (m_tot * env.g),  # accelerometer reading, body normal axis
        "n_wing": lift / (g.mass * env.g),  # wing load factor L/W
        "v_reel": y.v_reel,
        "L0": y.L0,
        "winch_pull": winch.throttle(L.winch, t, beta)
        * winch.available_pull(L.winch, y.v_reel),
        "winch_power": rf.tension[0] * y.v_reel,
        "ground_load": F_c[:, 1].sum(),
    }
    return dy, info


def vector_field(t, y: State, args: Args) -> State:
    return evaluate(t, y, args)[0]


def diagnostics(t, y: State, args: Args) -> dict:
    return evaluate(t, y, args)[1]
