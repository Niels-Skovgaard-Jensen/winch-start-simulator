import dataclasses

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import winch_sim  # noqa: F401
from winch_sim import aero, cable
from winch_sim.atmosphere import RHO0, density, indicated_airspeed, isa
from winch_sim.cable import DYNEEMA, STEEL
from winch_sim.dynamics import State, diagnostics, make_args, vector_field
from winch_sim.gliders import CATALOGUE
from winch_sim.params import Env, Launch
from winch_sim.simulate import EVENTS, solve_batch, solve_launch, summarize, unstack
from winch_sim.winch import engine_winch, tension_winch

ENV = Env()


def _launch(name="ASK 21", rope=DYNEEMA, winch="tension", **kw):
    g, p = CATALOGUE[name]
    w = tension_winch(g) if winch == "tension" else engine_winch(g)
    return Launch(
        glider=g, rope=rope, winch=w, pilot=p, env=ENV, rope_length=1200.0, slack=0.002
    )


# --- rope element ---------------------------------------------------------------


def _two_node_rope(rope, end, end_vel=(0.0, 0.0), L0=100.0):
    """Rope of 2 segments from (0, 100) to `end` (relative), middle node halfway."""
    base = jnp.array([0.0, 100.0])
    end = jnp.asarray(end) + base
    return cable.rope_forces(
        rope,
        Env(g=0.0),
        base,
        end,
        jnp.asarray(end_vel),
        ((base + end) / 2)[None],
        (jnp.asarray(end_vel) / 2)[None],
        jnp.asarray(L0),
        jnp.asarray(0.0),
        1.0,
    )


def test_rope_tension_is_EA_strain():
    rf = _two_node_rope(STEEL, (101.0, 50.0), L0=100.0)
    ell = np.hypot(101.0, 50.0)
    expected = STEEL.EA * (ell / 100.0 - 1.0)
    np.testing.assert_allclose(rf.tension, expected, rtol=1e-6)


def test_slack_rope_has_no_tension():
    rf = _two_node_rope(STEEL, (90.0, 0.0), L0=100.0)
    assert np.all(np.asarray(rf.tension) < 1e-6)


def test_rope_crossflow_drag():
    """A rope moving broadside feels 1/2 rho CDn d l v_n^2 of drag.

    Winch end fixed, middle node and hook moving up at v: segment 1 moves at v/2 on
    average, segment 2 at v; the middle node gets half of each segment's drag.
    """
    v = 20.0
    rf = cable.rope_forces(
        DYNEEMA,
        Env(g=0.0),
        jnp.array([0.0, 100.0]),  # well above the ground
        jnp.array([100.0, 100.0]),
        jnp.array([0.0, v]),
        jnp.array([[50.0, 100.0]]),
        jnp.array([[0.0, v]]),
        jnp.asarray(110.0),  # slack: no tension
        jnp.asarray(0.0),
        1.0,
    )
    q = 0.5 * float(density(ENV, 100.0)) * DYNEEMA.diameter * 50.0 * DYNEEMA.CDn
    expected = -0.5 * q * ((v / 2) ** 2 + v**2)
    np.testing.assert_allclose(rf.node_force[0, 1], expected, rtol=1e-4)


def test_hanging_rope_matches_elastic_catenary():
    """Lumped rope between two fixed points settles onto the elastic catenary."""
    n_seg = 20
    span, dz, H = 300.0, 80.0, 1500.0
    x_c, z_c, L_c = cable.elastic_catenary(
        span, dz, H, STEEL.mu * ENV.g, STEEL.EA, n=n_seg + 1
    )
    # Hang it 500 m up so the ground plays no role.
    lift = jnp.array([0.0, 500.0])
    end = jnp.array([span, dz]) + lift

    def f(t, y, args):
        p, v = y
        rf = cable.rope_forces(STEEL, ENV, lift, end, jnp.zeros(2), p, v, L_c, 0.0, 1.0)
        # extra viscous damping to settle quickly
        return v, rf.node_force / rf.node_mass - 2.0 * v

    s = np.linspace(0, 1, n_seg + 1)[1:-1]
    p0 = jnp.stack([s * span, s * dz], -1) + lift
    sol = dfx.diffeqsolve(
        dfx.ODETerm(f),
        dfx.Tsit5(),
        0.0,
        60.0,
        1e-3,
        (p0, jnp.zeros_like(p0)),
        stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7),
        max_steps=200_000,
    )
    assert sol.ys is not None
    p_end = np.asarray(sol.ys[0][-1]) - np.asarray(lift)
    sag = np.min(z_c - x_c * dz / span)
    assert sag < -5.0  # the test case really sags
    np.testing.assert_allclose(p_end[:, 1], z_c[1:-1], atol=0.01 * abs(sag))
    np.testing.assert_allclose(p_end[:, 0], x_c[1:-1], atol=0.01 * abs(sag))


# --- glider ---------------------------------------------------------------------


def test_steady_glide_ratio_matches_polar():
    """Released glider in free flight settles into a glide with -dx/dz = CL/CD."""
    L = _launch("LS4")
    args = make_args(L, attached=0.0)
    g = L.glider
    n = 4
    V0 = 25.0
    y0 = State(
        pos=jnp.array([0.0, 3000.0]),
        vel=jnp.array([V0, -1.0]),
        theta=jnp.asarray(L.pilot.theta_glide),
        q=jnp.asarray(0.0),
        node_p=jnp.stack([jnp.linspace(1000, 1100, n - 1), jnp.full(n - 1, -0.02)], -1),
        node_v=jnp.zeros((n - 1, 2)),
        L0=jnp.asarray(300.0),
        v_reel=jnp.asarray(0.0),
        de=jnp.asarray(0.0),
        e_int=jnp.asarray(0.0),
    )
    sol = dfx.diffeqsolve(
        dfx.ODETerm(vector_field),
        dfx.Tsit5(),
        0.0,
        400.0,
        1e-3,
        y0,
        args=args,
        stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7),
        max_steps=100_000,
    )
    y = jax.tree.map(lambda a: a[-1], sol.ys)
    glide = -y.vel[0] / y.vel[1]
    d = diagnostics(400.0, y, args)
    q_hat = y.q * g.chord / (2 * jnp.sqrt(d["V_tas"] ** 2 + 1))
    CL = aero.lift_coefficient(g, d["alpha"], y.de, q_hat)
    CD = aero.drag_coefficient(g, d["alpha"], CL)
    assert abs(float(y.q)) < 1e-3  # steady
    np.testing.assert_allclose(glide, CL / CD, rtol=0.01)
    assert 20 < float(glide) <= 40.5  # within the polar, below (L/D)max


# --- complete launches ------------------------------------------------------------


@pytest.fixture(scope="module")
def ask21_dyneema():
    L = _launch()
    return L, summarize(L, solve_launch(L, n_segments=12))


def test_launch_is_plausible(ask21_dyneema):
    _, s = ask21_dyneema
    assert s["event"] == "release"
    assert 350 < s["release_height"] < 600  # ~40% of a 1200 m rope
    assert 30 < s["release_time"] < 70
    assert 1.3 < s["max_n"] < 4.0
    assert s["max_T_hook"] < CATALOGUE["ASK 21"][0].weak_link


def test_rope_discretisation_converges(ask21_dyneema):
    L, s12 = ask21_dyneema
    s24 = summarize(L, solve_launch(L, n_segments=24))
    assert (
        abs(s24["release_height"] - s12["release_height"])
        < 0.02 * s12["release_height"]
    )


def test_heavy_steel_rope_costs_height(ask21_dyneema):
    _L, s_dyn = ask21_dyneema
    s_steel = summarize(_launch(rope=STEEL), solve_launch(_launch(rope=STEEL), 12))
    assert s_steel["release_height"] < s_dyn["release_height"]


def test_engine_winch_launch_releases():
    L = _launch(winch="engine")
    s = summarize(L, solve_launch(L, n_segments=12))
    assert s["event"] == "release"
    assert s["release_height"] > 300


def test_weak_link_breaks_on_excessive_pull():
    g, p = CATALOGUE["Ka 8"]
    L = Launch(
        glider=g,
        rope=DYNEEMA,
        winch=tension_winch(g, tension_per_weight=2.2),
        pilot=p,
        env=ENV,
        rope_length=1200.0,
        slack=0.002,
    )
    s = summarize(L, solve_launch(L, n_segments=12))
    assert s["event"] == "weak_link"


def test_vmap_batch_matches_single_runs(ask21_dyneema):
    L_ask, s_ask = ask21_dyneema
    L_ka8 = _launch("Ka 8")
    sol = solve_batch([L_ask, L_ka8], n_segments=12)
    s0 = summarize(L_ask, unstack(sol, 0))
    s1 = summarize(L_ka8, unstack(sol, 1))
    np.testing.assert_allclose(s0["release_height"], s_ask["release_height"], rtol=1e-6)
    assert abs(s0["release_height"] - s1["release_height"]) > 1.0
    assert EVENTS[int(unstack(sol, 1).event)] == "release"


# --- atmosphere -------------------------------------------------------------------


def test_isa_standard_values():
    T, p, rho = isa(Env(), 0.0)
    assert float(rho) == pytest.approx(1.225, abs=1e-4) and float(RHO0) == float(rho)
    T, p, rho = isa(Env(), 11000.0)  # tropopause
    assert float(T) == pytest.approx(216.65) and float(p) == pytest.approx(
        22632, rel=1e-3
    )
    assert float(rho) == pytest.approx(0.3639, rel=1e-3)
    # same density from field elevation or height above the field
    assert float(density(Env(field_elevation=1500.0), 500.0)) == pytest.approx(
        float(density(Env(), 2000.0))
    )
    # hot day -> thinner air
    assert float(density(Env(isa_dT=20.0), 0.0)) < 1.225


def test_ias_equals_tas_at_sea_level_and_is_lower_aloft():
    assert float(indicated_airspeed(Env(), 0.0, 30.0)) == pytest.approx(30.0, rel=1e-6)
    ias = float(indicated_airspeed(Env(), 3000.0, 30.0))
    assert ias == pytest.approx(30.0 * np.sqrt(0.9091 / 1.225), rel=2e-3)


def test_high_airfield_costs_height():
    L = _launch()
    hi = dataclasses.replace(L, env=Env(field_elevation=2000.0, isa_dT=15.0))
    s0 = summarize(L, solve_launch(L, n_segments=12))
    s1 = summarize(hi, solve_launch(hi, n_segments=12))
    assert s1["event"] == "release"
    assert s1["release_height"] < s0["release_height"]
    assert s1["ground_roll"] > s0["ground_roll"]  # higher TAS needed to lift off


def test_rope_too_heavy_to_drag_ends_with_no_liftoff():
    g, p = CATALOGUE["Ka 8"]
    heavy = dataclasses.replace(STEEL, mu=1.0)  # 1 kg/m: 1.2 t of rope on the grass
    L = Launch(
        glider=g,
        rope=heavy,
        winch=tension_winch(g),
        pilot=p,
        env=ENV,
        rope_length=1200.0,
        slack=0.002,
    )
    s = summarize(L, solve_launch(L, n_segments=12))
    assert s["event"] == "no_liftoff"


def test_segments_per_km():
    from winch_sim.simulate import n_segments_for

    assert n_segments_for(1200.0, 10) == 12
    assert n_segments_for(50000.0, 10) == 500
    assert n_segments_for(200.0, 10) == 4  # minimum
