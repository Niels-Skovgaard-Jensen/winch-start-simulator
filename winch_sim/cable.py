"""Lumped-mass winch rope with elasticity, weight, aerodynamic drag and ground contact.

The paid-out rope of unstretched length L0 is split into N segments of equal rest
length l0 = L0/N.  Node 0 is the rope exit at the winch (fixed), nodes 1..N-1 are free
point masses mu*l0, node N is the tow hook on the glider.  Reeling in (L0' = -v_reel)
shrinks every segment's rest length and node mass uniformly, which keeps the number of
states fixed (needed for JAX / diffrax).
"""

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from .aero import wind
from .atmosphere import density
from .ground import contact_force
from .params import Env, Rope
from .smooth import safe_norm, softplus


def rope_from_datasheet(
    *,
    diameter_mm: float,
    mass_kg_per_100m: float,
    breaking_load_kN: float,
    elongation_at_break_pct: float | None = None,
    EA_N: float | None = None,
    CDn: float = 1.2,
    Cf: float = 0.01,
    ground_mu: float = 0.4,
    zeta: float = 0.2,
    end_mass_kg: float = 8.0,
    end_CdA_m2: float = 0.1,
    ground_sink: float = 0.02,
) -> Rope:
    """Rope from manufacturer datasheet values.

    Axial stiffness either given directly (EA_N) or from the elongation at break,
    assuming a linear load-elongation curve: EA = breaking load / strain at break.
    (Synthetic ropes are stiffer at working loads than this secant value suggests;
    give EA_N from the datasheet's load-elongation curve at ~20-30 % of the breaking
    load when you have it.)
    """
    if EA_N is None:
        if elongation_at_break_pct is None:
            raise ValueError("give either EA_N or elongation_at_break_pct")
        EA_N = breaking_load_kN * 1e3 / (elongation_at_break_pct / 100.0)
    return Rope(
        diameter=diameter_mm * 1e-3,
        mu=mass_kg_per_100m / 100.0,
        EA=EA_N,
        breaking_load=breaking_load_kN * 1e3,
        zeta=zeta,
        CDn=CDn,
        Cf=Cf,
        ground_mu=ground_mu,
        ground_sink=ground_sink,
        end_mass=end_mass_kg,
        end_CdA=end_CdA_m2,
    )


# Generic presets (approximate, typical values; not a specific product).
STEEL = rope_from_datasheet(
    diameter_mm=4.5,
    mass_kg_per_100m=8.0,
    breaking_load_kN=17.0,
    EA_N=1.0e6,
    Cf=0.02,
    ground_mu=0.5,
)
DYNEEMA = rope_from_datasheet(
    diameter_mm=5.0,
    mass_kg_per_100m=1.6,
    breaking_load_kN=27.0,
    EA_N=6.0e5,
)

ROPES = {"steel": STEEL, "dyneema": DYNEEMA}


def load_rope(path) -> Rope:
    """Read a rope from a TOML file.

    Either datasheet keys (see `rope_from_datasheet`, e.g. diameter_mm,
    mass_kg_per_100m, breaking_load_kN, elongation_at_break_pct / EA_N, ...) or,
    with `base = "steel"|"dyneema"`, SI overrides of `Rope` fields on a preset
    (diameter, mu, EA, breaking_load, zeta, CDn, Cf, ground_mu, ...).
    """
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    data.pop("name", None)
    if "base" in data:
        return override_rope(ROPES[data.pop("base")], data)
    return rope_from_datasheet(**data)


def override_rope(rope: Rope, values: dict) -> Rope:
    """Replace `Rope` fields (SI units) by name."""
    unknown = set(values) - set(Rope.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown rope fields: {sorted(unknown)}")
    return dataclasses.replace(rope, **{k: float(v) for k, v in values.items()})


class RopeForces(NamedTuple):
    node_force: Array  # (N-1, 2) total force on each free node (incl. weight, ground)
    node_mass: Array  # scalar, mass of each free node
    hook_force: Array  # (2,) force the rope assembly exerts on the glider's hook
    end_mass: Array  # scalar, rope mass lumped at the hook (half segment + strop)
    tension: Array  # (N,) segment tensions, [0] at the winch, [-1] at the hook
    direction: Array  # (N, 2) unit vectors from node i-1 to node i


def rope_forces(
    rope: Rope,
    env: Env,
    winch_pos,
    hook_pos,
    hook_vel,
    node_p,
    node_v,
    L0,
    v_reel,
    attached,
) -> RopeForces:
    """All forces in the rope.  `attached` (1 or 0) says if the glider still hooked on."""
    n_seg = node_p.shape[0] + 1
    P = jnp.concatenate([winch_pos[None], node_p, hook_pos[None]], axis=0)
    V = jnp.concatenate([jnp.zeros((1, 2)), node_v, hook_vel[None]], axis=0)

    l0 = L0 / n_seg
    l0_dot = -v_reel / n_seg
    d = P[1:] - P[:-1]
    ell = safe_norm(d)
    e = d / ell[:, None]
    ell_dot = jnp.sum((V[1:] - V[:-1]) * e, axis=-1)

    # Axial force: elastic + internal damping; a rope cannot push (softplus).
    # c = 2 zeta sqrt(k_seg m_node) = 2 zeta sqrt(EA mu), independent of l0.
    c_int = 2.0 * rope.zeta * jnp.sqrt(rope.EA * rope.mu)
    strain = ell / l0 - 1.0
    T_raw = rope.EA * strain + c_int * (ell_dot - ell * l0_dot / l0)
    T = softplus(T_raw, 2.0)
    # After release, the last segment has a free end: no tension.
    T = T.at[-1].multiply(attached)

    # Aerodynamic drag per segment (cross-flow principle), split to both ends.
    mid = 0.5 * (P[1:] + P[:-1])
    v_rel = 0.5 * (V[1:] + V[:-1]) - wind(env, mid[:, 1])
    v_t = jnp.sum(v_rel * e, axis=-1, keepdims=True) * e
    v_n = v_rel - v_t
    qd = 0.5 * density(env, mid[:, 1])[:, None] * rope.diameter * ell[:, None]
    F_drag = -qd * (
        rope.CDn * safe_norm(v_n)[:, None] * v_n
        + rope.Cf * jnp.pi * safe_norm(v_t)[:, None] * v_t
    )
    half_drag = 0.5 * F_drag
    F_drag_nodes = half_drag[:-1] + half_drag[1:]  # free nodes 1..N-1

    m_node = rope.mu * l0
    F_tension = T[1:, None] * e[1:] - T[:-1, None] * e[:-1]
    F_weight = jnp.array([0.0, -1.0]) * (m_node * env.g)
    k_g = m_node * env.g / rope.ground_sink
    c_g = 2.0 * m_node * jnp.sqrt(env.g / rope.ground_sink)
    F_ground = contact_force(-node_p[:, 1], node_v, k_g, c_g, rope.ground_mu)
    node_force = F_tension + F_drag_nodes + F_weight + F_ground

    # Hook: top segment tension, half the top segment's drag, parachute drag, and the
    # weight of the rope end assembly.
    v_rel_hook = hook_vel - wind(env, hook_pos[1])
    F_chute = (
        -0.5
        * density(env, hook_pos[1])
        * rope.end_CdA
        * safe_norm(v_rel_hook)
        * v_rel_hook
    )
    end_mass = rope.end_mass + 0.5 * m_node
    hook_force = attached * (
        -T[-1] * e[-1]
        + half_drag[-1]
        + F_chute
        + jnp.array([0.0, -1.0]) * (end_mass * env.g)
    )
    return RopeForces(node_force, m_node, hook_force, attached * end_mass, T, e)


def elastic_catenary(
    span: float, height_diff: float, H: float, w: float, EA: float, n: int = 200
):
    """Static elastic catenary between (0, 0) and (span, height_diff).

    H is the horizontal tension component, w the weight per unit (unstretched) length.
    Returns (x, z, unstretched length).  Used to validate the lumped-mass rope.
    Irvine (1981) parametrisation by unstretched arc length s from the left support;
    V0 is the vertical tension component at s = 0 (negative when the rope sags).
    """

    def shape(V0, L):
        s = np.linspace(0.0, L, n)
        Vs = V0 + w * s  # vertical tension component along the rope (V0 < 0 = sag)
        x = H * s / EA + (H / w) * (np.arcsinh(Vs / H) - np.arcsinh(V0 / H))
        z = (w * s**2 / 2 + V0 * s) / EA + (H / w) * (
            np.sqrt(1 + (Vs / H) ** 2) - np.sqrt(1 + (V0 / H) ** 2)
        )
        return x, z

    # Solve the 2x2 system end(V0, L) = (span, height_diff) with Newton.
    L = float(np.hypot(span, height_diff))
    V0 = -w * L / 2 + H * height_diff / max(span, 1e-9)
    for _ in range(100):
        x, z = shape(V0, L)
        r = np.array([x[-1] - span, z[-1] - height_diff])
        if np.max(np.abs(r)) < 1e-10:
            break
        J = np.zeros((2, 2))
        for j, (dV, dL) in enumerate([(1e-4 * H, 0.0), (0.0, 1e-6 * L)]):
            xp, zp = shape(V0 + dV, L + dL)
            J[:, j] = (np.array([xp[-1] - span, zp[-1] - height_diff]) - r) / (dV + dL)
        step = np.linalg.solve(J, -r)
        V0 += step[0]
        L += step[1]
    x, z = shape(V0, L)
    return x, z, L
