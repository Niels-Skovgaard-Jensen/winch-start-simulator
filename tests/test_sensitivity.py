import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import winch_sim  # noqa: F401
from winch_sim.cable import STEEL, load_rope, override_rope, rope_from_datasheet
from winch_sim.gliders import CATALOGUE
from winch_sim.params import Env, Launch
from winch_sim.sensitivity import (
    as_float_arrays,
    height_and_gradient,
    release_height,
    sensitivities,
)
from winch_sim.winch import tension_winch

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def launch():
    g, p = CATALOGUE["ASK 21"]
    return as_float_arrays(
        Launch(
            glider=g,
            rope=STEEL,
            winch=tension_winch(g),
            pilot=p,
            env=Env(),
            rope_length=1200.0,
            slack=0.002,
        )
    )


@pytest.mark.parametrize(
    "path", ["rope_length", "rope.mu", "rope.CDn", "glider.mass", "winch.F_max"]
)
def test_gradient_matches_finite_differences(launch, path):
    _, grads = height_and_gradient(launch)

    def get(tree):
        for part in path.split("."):
            tree = getattr(tree, part)
        return float(tree)

    def with_value(v):
        parts = path.split(".")
        if len(parts) == 1:
            return dataclasses.replace(launch, **{parts[0]: jnp.asarray(v)})
        sub = dataclasses.replace(
            getattr(launch, parts[0]), **{parts[1]: jnp.asarray(v)}
        )
        return dataclasses.replace(launch, **{parts[0]: sub})

    v = get(launch)
    dv = 1e-3 * abs(v)
    kw = {"rtol": 1e-9, "atol": 1e-9, "max_steps": 200_000}
    fd = (
        release_height(with_value(v + dv), **kw)
        - release_height(with_value(v - dv), **kw)
    ) / (2 * dv)
    np.testing.assert_allclose(get(grads), float(fd), rtol=5e-3)


def test_sensitivity_table_has_units_and_physics(launch):
    h, rows = sensitivities(launch)
    by_name = {r.name: r for r in rows}
    assert all(r.unit != "?" for r in rows)
    assert len(rows) == len(by_name) > 70
    # longer rope and stronger winch -> higher; heavier glider, draggier rope -> lower
    assert by_name["rope_length"].dh_dp > 0
    assert by_name["winch.F_max"].dh_dp > 0
    assert by_name["glider.mass"].dh_dp < 0
    assert by_name["rope.CDn"].dh_dp < 0
    assert by_name["rope.mu"].dh_dp < 0
    assert by_name["pilot.theta_climb"].unit == "deg"
    np.testing.assert_allclose(
        by_name["rope_length"].elasticity,
        by_name["rope_length"].dh_dp * 1200.0 / h,
    )


def test_rope_from_datasheet():
    r = rope_from_datasheet(
        diameter_mm=6.0,
        mass_kg_per_100m=2.3,
        breaking_load_kN=38.0,
        elongation_at_break_pct=4.0,
    )
    assert r.diameter == pytest.approx(0.006)
    assert r.mu == pytest.approx(0.023)
    assert r.EA == pytest.approx(38e3 / 0.04)
    with pytest.raises(ValueError):
        rope_from_datasheet(diameter_mm=5, mass_kg_per_100m=1.6, breaking_load_kN=27)


def test_rope_files_and_overrides():
    r = load_rope(ROOT / "ropes" / "example_dyneema_6mm.toml")
    assert r.breaking_load == pytest.approx(38e3)
    s = load_rope(ROOT / "ropes" / "example_steel_override.toml")
    assert s.mu == pytest.approx(0.095) and s.Cf == STEEL.Cf
    assert override_rope(STEEL, {"EA": "2e6"}).EA == pytest.approx(2e6)
    with pytest.raises(ValueError):
        override_rope(STEEL, {"stiffness": 1.0})
