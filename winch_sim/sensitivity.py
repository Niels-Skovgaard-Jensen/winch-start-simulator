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
from .simulate import launch_event, step_controller

CONTACT_POINTS = ("nose", "wheel", "tail")


@eqx.filter_jit
def release_height(
    launch: Launch,
    n_segments: int = 12,
    t_max: float = 200.0,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = 20_000,
):
    """Height above the rest position at the end of the launch (differentiable)."""
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
    assert sol.ys is not None
    return sol.ys.pos[-1, 1] - args.z_rest


def as_float_arrays(launch: Launch) -> Launch:
    """All leaves as float arrays, so every parameter is differentiable."""
    return jax.tree.map(lambda x: jnp.asarray(x, dtype=float), launch)


@eqx.filter_jit
def height_and_gradient(launch: Launch, n_segments: int = 12, tol: float = 1e-8):
    """Release height and d(height)/d(parameter) as a Launch-shaped pytree.

    The gradient is that of the discretised solve, so tiny sensitivities are only
    as accurate as the solver tolerance (rtol = atol = tol) allows.
    """
    f = eqx.filter_value_and_grad(
        lambda L: release_height(L, n_segments, rtol=tol, atol=tol, max_steps=200_000)
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


def sensitivities(launch: Launch, n_segments: int = 12, tol: float = 1e-8):
    """Release height [m] and Sensitivity rows, sorted by |elasticity| (largest first)."""
    launch = as_float_arrays(launch)
    h, grads = height_and_gradient(launch, n_segments, tol)
    h = float(h)
    rows = []
    for name, unit, value, g in _walk(launch, grads):
        rows.append(
            Sensitivity(
                name=name,
                value=value,
                unit=unit,
                dh_dp=g,
                dh_10pct=0.1 * value * g,
                elasticity=value * g / h,
            )
        )
    rows.sort(key=lambda r: (-abs(r.elasticity), -abs(r.dh_dp)))
    return h, rows


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
