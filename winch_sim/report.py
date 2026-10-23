"""Shared reporting helpers (CLI and GUI)."""

import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from .sensitivity import sensitivity_unit

# (summary key, header, format spec, scale)
COLUMNS = [
    ("event", "end", "", 1.0),
    ("release_height", "h_rel m", ".0f", 1.0),
    ("release_time", "t_rel s", ".1f", 1.0),
    ("ground_roll", "roll m", ".0f", 1.0),
    ("max_V_ias_kmh", "Vmax IAS", ".0f", 1.0),
    ("V_W_kmh", "V_W IAS", ".0f", 1.0),
    ("min_V_ias_kmh", "Vmin IAS", ".0f", 1.0),
    ("max_V_tas_kmh", "Vmax TAS", ".0f", 1.0),
    ("max_T_hook", "Tmax kN", ".2f", 1e-3),
    ("max_n", "n max", ".2f", 1.0),
    ("max_power_kW", "P max kW", ".0f", 1.0),
    ("rope_safety_factor", "rope SF", ".1f", 1.0),
]


def resample_series(
    series: dict, dt: float | None = None, n: int | None = None
) -> dict[str, np.ndarray]:
    """Series linearly interpolated onto evenly spaced times from t[0] to t[-1].

    Give either the time step `dt` (the last point is the last whole step) or the
    number of points `n` (including both ends).  The release time is saved twice
    (end of the launch, start of the free flight), so the jump there has zero
    width and is not blended; a point exactly at release gets the launch value.
    """
    if (dt is None) == (n is None):
        raise ValueError("give either dt or n")
    t = np.asarray(series["t"], float)
    if dt is not None:
        if not dt > 0:
            raise ValueError(f"time step must be positive, got {dt}")
        grid = t[0] + dt * np.arange(int(np.floor((t[-1] - t[0]) / dt + 1e-9)) + 1)
    else:
        if n is None or n < 2:
            raise ValueError(f"need at least 2 points, got {n}")
        grid = np.linspace(t[0], t[-1], n)
    # One matrix for all columns: interp1d treats 1-D and 2-D values differently
    # at a repeated time, and each row must stay on one side of the release.
    keys = [k for k, v in series.items() if np.ndim(v) in (1, 2) and len(v) == t.size]
    cols = [np.asarray(series[k], float).reshape(t.size, -1) for k in keys]
    res = interp1d(t, np.hstack(cols), axis=0, assume_sorted=True)(grid)
    parts = np.split(res, np.cumsum([c.shape[1] for c in cols])[:-1], axis=1)
    out = {
        k: p.reshape((grid.size,) + np.shape(series[k])[1:])
        for k, p in zip(keys, parts, strict=True)
    }
    out["t"] = grid
    return out


def write_series_csv(
    fn: Path | str,
    series: dict,
    rope: bool = False,
    dt: float | None = None,
    n: int | None = None,
) -> None:
    """Time series (SI units) as CSV, one row per save point.

    With `rope`, the rope node positions follow as columns rope_x_0 ... rope_x_N,
    rope_z_0 ... rope_z_N (node 0 at the winch, node N at the hook, or the free rope
    end after release).  That adds two columns per node, often doubling the file.
    With `dt` or `n`, the series is first resampled (see resample_series).
    """
    if dt is not None or n is not None:
        series = resample_series(series, dt=dt, n=n)
    keys = [k for k, v in series.items() if np.ndim(v) == 1]
    cols = [series[k] for k in keys]
    if rope:
        for k in ("rope_x", "rope_z"):
            nodes = np.asarray(series[k])
            keys += [f"{k}_{j}" for j in range(nodes.shape[1])]
            cols += list(nodes.T)
    np.savetxt(
        fn,
        np.column_stack(cols),
        delimiter=",",
        header=",".join(keys),
        comments="",
    )


def write_sensitivity_csv(fn: Path | str, rows) -> None:
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "parameter",
                "value",
                "unit",
                "dh_dp",
                "dh_dp_unit",
                "dh_for_plus_10pct_m",
                "elasticity",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.name,
                    f"{r.value:.6g}",
                    r.unit,
                    f"{r.dh_dp:.6g}",
                    sensitivity_unit(r.unit),
                    f"{r.dh_10pct:.6g}",
                    f"{r.elasticity:.6g}",
                ]
            )
