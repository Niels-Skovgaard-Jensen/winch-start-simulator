"""Shared reporting helpers (CLI and GUI)."""

import csv
from pathlib import Path

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
