"""Sensitivity of the release height to every model parameter.

`jax.grad` is taken straight through the diffrax solve (reverse mode with
recursive checkpointing), including the release event: diffrax locates the event
time with a root finder and differentiates it implicitly.  One backward pass gives
dh/dp for all ~80 parameters at once.

Sensitivities are local (a linearisation around the given launch) and every
parameter is varied on its own with all others held fixed; e.g. changing the glider
mass does not change a winch pull that was preset as "1.1 x weight".
"""

import dataclasses
from typing import NamedTuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .dynamics import initial_state, make_args, vector_field
from .params import DISPLAY_UNITS, Launch
from .simulate import (
    MAX_STEPS,
    T_MAX,
    launch_event,
    stack,
    step_controller,
    unstack,
)

CONTACT_POINTS = ("nose", "wheel", "tail")


def _release(
    launch: Launch,
    n_segments: int,
    t_max: float,
    rtol: float,
    atol: float,
    max_steps: int,
):
    """(height above rest position at the end of the launch, launch ended?)"""
    args = make_args(launch, attached=1.0)
    y0 = initial_state(args, n_segments)
    sol = dfx.diffeqsolve(
        dfx.ODETerm(vector_field),
        dfx.Tsit5(),
        t0=0.0,
        t1=t_max,
        dt0=1e-3,
        y0=y0,
        args=args,
        saveat=dfx.SaveAt(t1=True),
        stepsize_controller=step_controller(rtol, atol),
        event=launch_event(),
        max_steps=max_steps,
        adjoint=dfx.RecursiveCheckpointAdjoint(),
    )
    assert sol.ys is not None and sol.event_mask is not None
    ended = jnp.stack(sol.event_mask).any()
    return sol.ys.pos[-1, 1] - args.z_rest, ended


@eqx.filter_jit
def release_height(
    launch: Launch,
    n_segments: int = 12,
    t_max: float = T_MAX,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = MAX_STEPS,
):
    """Height above the rest position at the end of the launch (differentiable).

    Same solve as stage 1 of `simulate.solve_launch`; if no end-of-launch event
    occurs before t_max, this is the height at t_max.
    """
    return _release(launch, n_segments, t_max, rtol, atol, max_steps)[0]


def as_float_arrays(launch: Launch) -> Launch:
    """All leaves as float arrays, so every parameter is differentiable."""
    return jax.tree.map(lambda x: jnp.asarray(x, dtype=float), launch)


@eqx.filter_jit
def height_and_gradient(
    launch: Launch, n_segments: int = 12, tol: float = 1e-8, t_max: float = T_MAX
):
    """((release height, launch ended?), d(height)/d(parameter) as a Launch pytree).

    The gradient is that of the discretised solve, so tiny sensitivities are only
    as accurate as the solver tolerance (rtol = atol = tol) allows.
    """
    f = eqx.filter_value_and_grad(
        lambda L: _release(L, n_segments, t_max, tol, tol, MAX_STEPS), has_aux=True
    )
    return f(as_float_arrays(launch))


class Sensitivity(NamedTuple):
    name: str  # e.g. "rope.mu"
    value: float  # parameter value, in `unit`
    unit: str
    dh_dp: float  # [m per unit]
    dh_10pct: float  # height change for a +10 % change of the parameter [m]
    elasticity: float  # (p/h) dh/dp: % height per % parameter


def _walk(params, grads, prefix=""):
    """Yield (name, unit, value, gradient) for every scalar parameter, converted to
    the parameter's display unit (value in display units, gradient per display unit).
    """
    for f in dataclasses.fields(params):
        p, g = getattr(params, f.name), getattr(grads, f.name)
        name = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(p):
            yield from _walk(p, g, name + ".")
            continue
        unit = f.metadata.get("display", f.metadata.get("unit", "?"))
        scale = DISPLAY_UNITS.get(unit, 1.0)
        p, g = np.asarray(p, float) / scale, np.asarray(g, float) * scale
        if p.ndim == 0:
            yield name, unit, float(p), float(g)
        else:
            labels = CONTACT_POINTS if p.shape == (3,) else range(p.size)
            for lab, pi, gi in zip(labels, p.ravel(), g.ravel(), strict=True):
                yield f"{name}[{lab}]", unit, float(pi), float(gi)


def _rows(launch: Launch, h: float, grads) -> list[Sensitivity]:
    rows = [
        Sensitivity(
            name=name,
            value=value,
            unit=unit,
            dh_dp=g,
            dh_10pct=0.1 * value * g,
            elasticity=value * g / h,
        )
        for name, unit, value, g in _walk(launch, grads)
    ]
    rows.sort(key=lambda r: (-abs(r.elasticity), -abs(r.dh_dp)))
    return rows


def _check_ended(ended, t_max: float) -> None:
    if not bool(ended):
        raise RuntimeError(
            f"the launch did not end (no release/weak-link/rope-in event) within "
            f"t_max = {t_max:g} s; sensitivities of the height at t_max would be "
            f"meaningless. Increase t_max."
        )


def sensitivities(
    launch: Launch, n_segments: int = 12, tol: float = 1e-8, t_max: float = T_MAX
):
    """Release height [m] and Sensitivity rows, sorted by |elasticity| (largest first)."""
    launch = as_float_arrays(launch)
    (h, ended), grads = height_and_gradient(launch, n_segments, tol, t_max)
    _check_ended(ended, t_max)
    return float(h), _rows(launch, float(h), grads)


def batch_sensitivities(
    launches: list[Launch],
    n_segments: int = 12,
    tol: float = 1e-8,
    t_max: float = T_MAX,
):
    """`sensitivities` for many launches at once (one vmap-ed, compiled program)."""
    launches = [as_float_arrays(L) for L in launches]
    (h, ended), grads = jax.vmap(
        lambda L: height_and_gradient(L, n_segments, tol, t_max)
    )(stack(launches))
    out = []
    for i, L in enumerate(launches):
        _check_ended(ended[i], t_max)
        out.append((float(h[i]), _rows(L, float(h[i]), unstack(grads, i))))
    return out


def sensitivity_unit(unit: str) -> str:
    if unit == "-":
        return "m"
    return f"m/({unit})" if (" " in unit or "/" in unit) else f"m/{unit}"


def format_table(h: float, rows: list[Sensitivity], top: int | None = None) -> str:
    lines = [
        f"release height h = {h:.1f} m",
        (
            f"{'parameter':28s} {'value':>12s} {'unit':>8s} {'dh/dp':>11s} "
            f"{'[unit]':<14s} {'dh(+10%)':>9s} {'elast.':>7s}"
        ),
    ]
    for r in rows[:top]:
        lines.append(
            f"{r.name:28s} {r.value:12.4g} {r.unit:>8s} {r.dh_dp:11.4g} "
            f"{sensitivity_unit(r.unit):<14s} {r.dh_10pct:9.2f} {r.elasticity:7.3f}"
        )
    return "\n".join(lines)
