"""Solve a winch launch with diffrax.

Stage 1 integrates from standstill until a release event (pilot release at the top,
hook back-release, weak-link failure, or the rope reeled in).  Stage 2 continues the
glider (and the falling rope) in free flight for a few seconds.
"""

from typing import Literal, NamedTuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array

from .dynamics import State, diagnostics, initial_state, make_args, vector_field
from .params import Launch

EVENTS = ("release", "back_release", "weak_link", "rope_in")
T_MAX = 1.0e5  # [s] give up if the launch has not ended by then (~28 h)
MAX_STEPS = 5_000_000


def _info(t, y, args, name):
    return diagnostics(t, y, args)[name]


def _ev_release(t, y, args, **kw):
    return _info(t, y, args, "cable_angle") - args.launch.pilot.release_angle


def _ev_back_release(t, y, args, **kw):
    return _info(t, y, args, "cable_body") - args.launch.pilot.back_release_angle


def _ev_weak_link(t, y, args, **kw):
    return _info(t, y, args, "T_hook") - args.launch.glider.weak_link


def _ev_rope_in(t, y, args, **kw):
    return 30.0 - y.L0


def launch_event() -> dfx.Event:
    """End-of-launch events (see EVENTS), located exactly by root finding."""
    return dfx.Event(
        cond_fn=[_ev_release, _ev_back_release, _ev_weak_link, _ev_rope_in],
        root_finder=optx.Newton(rtol=1e-8, atol=1e-8),
        direction=True,
    )


def step_controller(rtol: float, atol: float) -> dfx.PIDController:
    return dfx.PIDController(rtol=rtol, atol=atol, pcoeff=0.3, icoeff=0.4)


def _solver(name: str):
    if name == "tsit5":
        return dfx.Tsit5()
    if name == "kvaerno5":
        return dfx.Kvaerno5()
    raise ValueError(name)


class LaunchSolution(NamedTuple):
    ts: Array  # stage-1 save times
    ys: State
    valid: Array  # usable save points (all, unless the solve failed)
    ts_post: Array  # stage-2 (free flight) save times
    ys_post: State
    t_release: Array
    event: Array  # index into EVENTS, -1 if t_max was reached
    num_steps: Array


@eqx.filter_jit
def solve_launch(
    launch: Launch,
    n_segments: int = 16,
    t_max: float = T_MAX,
    n_save: int = 2000,
    t_post: float = 8.0,
    n_post: int = 200,
    solver: Literal["tsit5", "kvaerno5"] = "tsit5",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = MAX_STEPS,
) -> LaunchSolution:
    """Simulate a launch.

    Pass 1 runs to the end-of-launch event (or t_max) saving only the final state,
    so t_max can be very large at no cost.  Pass 2 repeats the (deterministic) solve
    from 0 to the release time and saves n_save evenly spaced points for plotting.
    Stage 2 continues in free flight for t_post seconds (n_post points).
    """
    args = make_args(launch, attached=1.0)
    y0 = initial_state(args, n_segments)
    term = dfx.ODETerm(vector_field)
    controller = step_controller(rtol, atol)
    common = {
        "stepsize_controller": controller,
        "max_steps": max_steps,
        "throw": False,
        "dt0": 1e-3,
    }
    sol = dfx.diffeqsolve(
        term,
        _solver(solver),
        t0=0.0,
        t1=t_max,
        y0=y0,
        args=args,
        saveat=dfx.SaveAt(t1=True),
        event=launch_event(),
        **common,
    )
    assert sol.ts is not None and sol.ys is not None and sol.event_mask is not None
    t_rel = sol.ts[0]
    y_rel = jax.tree.map(lambda a: a[0], sol.ys)
    mask = jnp.stack(sol.event_mask)
    ev = jnp.where(mask.any(), jnp.argmax(mask), -1)

    ts = jnp.linspace(0.0, t_rel, n_save)
    sol_path = dfx.diffeqsolve(
        term,
        _solver(solver),
        t0=0.0,
        t1=t_rel,
        y0=y0,
        args=args,
        saveat=dfx.SaveAt(ts=ts),
        **common,
    )
    assert sol_path.ys is not None

    args_post = make_args(launch, attached=0.0)
    ts_post = t_rel + jnp.linspace(0.0, t_post, n_post)
    sol2 = dfx.diffeqsolve(
        term,
        _solver(solver),
        t0=t_rel,
        t1=t_rel + t_post,
        y0=y_rel,
        args=args_post,
        saveat=dfx.SaveAt(ts=ts_post),
        **common,
    )
    assert sol2.ys is not None
    return LaunchSolution(
        ts=ts,
        ys=sol_path.ys,
        valid=jnp.isfinite(ts),
        ts_post=ts_post,
        ys_post=sol2.ys,
        t_release=t_rel,
        event=ev,
        num_steps=sol.stats["num_steps"],
    )


def solution_diagnostics(launch: Launch, sol: LaunchSolution):
    """Derived time series (airspeed, tension, angles, ...) for both stages."""
    a1 = make_args(launch, 1.0)
    a2 = make_args(launch, 0.0)
    d1 = jax.vmap(diagnostics, in_axes=(0, 0, None))(sol.ts, sol.ys, a1)
    d2 = jax.vmap(diagnostics, in_axes=(0, 0, None))(sol.ts_post, sol.ys_post, a2)
    return d1, d2


def summarize(launch: Launch, sol: LaunchSolution) -> dict[str, float | str]:
    d1, _ = solution_diagnostics(launch, sol)
    valid = np.asarray(sol.valid)
    d = {k: np.asarray(v)[valid] for k, v in d1.items()}
    ev = int(sol.event)
    y_end = jax.tree.map(lambda a: np.asarray(a)[0], sol.ys_post)
    z_rest = float(make_args(launch).z_rest)
    airborne = d["height"] > 5.0
    return {
        "event": EVENTS[ev] if ev >= 0 else "t_max",
        "release_height": float(y_end.pos[1] - z_rest),
        "release_time": float(sol.t_release),
        "ground_roll": float(d["x"][np.argmax(d["height"] > 0.5)]),
        "max_V_kmh": float(d["V"].max() * 3.6),
        "V_W_kmh": float(launch.glider.V_W) * 3.6,
        "min_V_air_kmh": float(d["V"][airborne].min() * 3.6) if airborne.any() else 0.0,
        "max_T_hook": float(d["T_hook"].max()),
        "max_T_winch": float(d["T_winch"].max()),
        "rope_safety_factor": float(launch.rope.breaking_load)
        / float(d["T_max_rope"].max()),
        "max_n": float(d["n_wing"][airborne].max()) if airborne.any() else 0.0,
        "max_power_kW": float(d["winch_power"].max() / 1e3),
        "rope_used": float(d["L0"][0] - d["L0"][-1]),
        "num_steps": int(sol.num_steps),
    }


def time_series(launch: Launch, sol: LaunchSolution) -> dict[str, np.ndarray]:
    """Diagnostics of launch + free flight joined into one set of numpy arrays.

    Adds `released` (0/1) and the rope node positions `rope_x`, `rope_z` (time, node),
    including the winch and hook ends.
    """
    d1, d2 = solution_diagnostics(launch, sol)
    valid = np.asarray(sol.valid)
    ok2 = np.isfinite(np.asarray(sol.ts_post))
    out = {
        k: np.concatenate([np.asarray(d1[k])[valid], np.asarray(d2[k])[ok2]])
        for k in d1
    }
    out["released"] = np.concatenate([np.zeros(valid.sum()), np.ones(ok2.sum())])
    nodes = np.concatenate(
        [np.asarray(sol.ys.node_p)[valid], np.asarray(sol.ys_post.node_p)[ok2]]
    )
    args = make_args(launch)
    n_t = nodes.shape[0]
    winch_xz = np.broadcast_to(np.asarray(args.winch_pos), (n_t, 1, 2))
    g = launch.glider
    th = out["theta"]
    hook = np.stack(
        [
            out["x"] + np.cos(th) * float(g.hook_x) - np.sin(th) * float(g.hook_z),
            out["z"] + np.sin(th) * float(g.hook_x) + np.cos(th) * float(g.hook_z),
        ],
        -1,
    )[:, None]
    # After release the hook is no longer the rope end: drop it from the rope shape.
    hook = np.where(out["released"][:, None, None] > 0, nodes[:, -1:], hook)
    rope = np.concatenate([winch_xz, nodes, hook], axis=1)
    out["rope_x"], out["rope_z"] = rope[..., 0], rope[..., 1]
    return out


def stack(launches: list[Launch]) -> Launch:
    """Stack launches (same rope discretisation) into one batched Launch pytree."""
    return jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *launches)


def unstack(tree, i: int):
    return jax.tree.map(lambda a: a[i], tree)


def solve_batch(launches: list[Launch], **kwargs) -> LaunchSolution:
    """Solve many launches at once with jax.vmap (one compiled program).

    Returns a batched LaunchSolution; use `unstack(sol, i)` to get launch i.
    """
    batched = stack(launches)
    return jax.vmap(lambda L: solve_launch(L, **kwargs))(batched)
