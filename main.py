"""Simulate winch launches for a set of gliders and compare them.

Examples:
    uv run main.py                                   # all gliders, Dyneema, tension winch
    uv run main.py --gliders "ASK 21" LS4 --rope steel --winch engine
    uv run main.py --rope-length 1000 --wind 5 --out results
    uv run main.py --rope-file ropes/example_dyneema_6mm.toml
    uv run main.py --rope steel --rope-param mu=0.09 --rope-param EA=1.2e6
    uv run main.py --gliders "ASK 13" --sensitivity  # d(release height)/d(parameter)
"""

import argparse
import csv
from pathlib import Path

import winch_sim  # noqa: F401  (enables float64)
from winch_sim import plots
from winch_sim.cable import ROPES, load_rope, override_rope
from winch_sim.gliders import CATALOGUE
from winch_sim.params import Env, Launch
from winch_sim.sensitivity import format_table, sensitivities, sensitivity_unit
from winch_sim.simulate import solve_batch, summarize, time_series, unstack
from winch_sim.winch import engine_winch, tension_winch

# (summary key, header, format spec, scale)
COLUMNS = [
    ("event", "end", "", 1.0),
    ("release_height", "h_rel m", ".0f", 1.0),
    ("release_time", "t_rel s", ".1f", 1.0),
    ("ground_roll", "roll m", ".0f", 1.0),
    ("max_V_kmh", "Vmax km/h", ".0f", 1.0),
    ("V_W_kmh", "V_W km/h", ".0f", 1.0),
    ("min_V_air_kmh", "Vmin km/h", ".0f", 1.0),
    ("max_T_hook", "Tmax kN", ".2f", 1e-3),
    ("max_n", "n max", ".2f", 1.0),
    ("max_power_kW", "P max kW", ".0f", 1.0),
    ("rope_safety_factor", "rope SF", ".1f", 1.0),
]
W = 11


def _cell(value, spec: str, scale: float) -> str:
    if isinstance(value, str):
        return f"{value:>{W}s}"
    return f"{value * scale:>{W}{spec}}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--gliders", nargs="+", default=list(CATALOGUE), choices=list(CATALOGUE)
    )
    ap.add_argument("--rope", choices=list(ROPES), default="dyneema")
    ap.add_argument("--winch", choices=["tension", "engine"], default="tension")
    ap.add_argument(
        "--pull", type=float, default=1.1, help="winch pull / glider weight"
    )
    ap.add_argument("--rope-length", type=float, default=1200.0)
    ap.add_argument("--wind", type=float, default=0.0, help="headwind at 10 m [m/s]")
    ap.add_argument("--segments", type=int, default=12, help="rope segments")
    ap.add_argument(
        "--rope-file", type=Path, help="rope from a TOML file (see ropes/*.toml)"
    )
    ap.add_argument(
        "--rope-param",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override a Rope field in SI units, e.g. mu=0.09 (repeatable)",
    )
    ap.add_argument(
        "--sensitivity",
        action="store_true",
        help="print d(release height)/d(parameter) for every parameter",
    )
    ap.add_argument("--top", type=int, default=25, help="rows in sensitivity table")
    ap.add_argument(
        "--only",
        nargs="+",
        metavar="PREFIX",
        help="restrict the sensitivity table to e.g. rope, glider, winch.F_max",
    )
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()

    rope = load_rope(a.rope_file) if a.rope_file else ROPES[a.rope]
    rope_name = a.rope_file.stem if a.rope_file else a.rope
    if a.rope_param:
        rope = override_rope(rope, dict(kv.split("=", 1) for kv in a.rope_param))
        rope_name += "_mod"

    launches = []
    for name in a.gliders:
        glider, pilot = CATALOGUE[name]
        if a.winch == "tension":
            w = tension_winch(glider, tension_per_weight=a.pull)
        else:
            w = engine_winch(glider, pull_per_weight=a.pull)
        launches.append(
            Launch(
                glider=glider,
                rope=rope,
                winch=w,
                pilot=pilot,
                env=Env(wind_ref=a.wind),
                rope_length=a.rope_length,
                slack=0.002,
            )
        )

    print(
        f"rope: {rope_name} {a.rope_length:.0f} m (d = {rope.diameter * 1e3:.1f} mm, "
        f"{rope.mu * 100:.1f} kg/100 m, EA = {rope.EA / 1e3:.0f} kN, "
        f"breaking load {rope.breaking_load / 1e3:.1f} kN), "
        f"winch: {a.winch} ({a.pull} x weight), headwind {a.wind} m/s\n"
    )
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{rope_name}_{a.winch}"

    if a.sensitivity:
        for name, L in zip(a.gliders, launches, strict=True):
            h, rows = sensitivities(L, n_segments=a.segments)
            shown = rows
            if a.only:
                shown = [r for r in rows if r.name.startswith(tuple(a.only))]
            print(f"=== {name} ===")
            print(format_table(h, shown, a.top))
            fn = a.out / f"sensitivity_{name.replace(' ', '_')}_{tag}.csv"
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
                            r.value,
                            r.unit,
                            r.dh_dp,
                            sensitivity_unit(r.unit),
                            r.dh_10pct,
                            r.elasticity,
                        ]
                    )
            print(f"(all {len(rows)} parameters in {fn})\n")
        return

    sol = solve_batch(launches, n_segments=a.segments)

    print(f"{'glider':10s}" + "".join(f"{h:>{W}s}" for _, h, _, _ in COLUMNS))
    runs = {}
    for i, (name, L) in enumerate(zip(a.gliders, launches)):
        s_i = unstack(sol, i)
        summ = summarize(L, s_i)
        print(f"{name:10s}" + "".join(_cell(summ[k], f, sc) for k, _, f, sc in COLUMNS))
        runs[name] = time_series(L, s_i)

    fig = plots.compare(
        runs, f"Winch launch comparison — {rope_name} rope, {a.winch} winch"
    )
    fig.savefig(a.out / f"compare_{tag}.png", dpi=130)
    for name, s in runs.items():
        fn = a.out / f"detail_{name.replace(' ', '_')}_{tag}.png"
        plots.launch_detail(f"{name} — {rope_name} rope, {a.winch} winch", s).savefig(
            fn, dpi=130
        )
    print(f"\nplots written to {a.out}/")


if __name__ == "__main__":
    main()
