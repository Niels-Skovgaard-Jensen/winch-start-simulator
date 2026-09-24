"""Simulate winch launches for a set of gliders and compare them.

Examples:
    uv run main.py                                   # all gliders, Dyneema, tension winch
    uv run main.py --gliders "ASK 21" LS4 --rope steel --winch engine
    uv run main.py --rope-length 1000 --wind 5 --out results
    uv run main.py --rope-file ropes/example_dyneema_6mm.toml
    uv run main.py --rope steel --rope-param mu=0.09 --rope-param EA=1.2e6
    uv run main.py --gliders "ASK 13" --only rope    # sensitivity table: rope only
    uv run main.py --no-sensitivity                  # skip the sensitivities
"""

import argparse
import csv
from pathlib import Path

import winch_sim  # noqa: F401  (enables float64)
from winch_sim import plots
from winch_sim.cable import ROPES, load_rope, override_rope
from winch_sim.gliders import CATALOGUE
from winch_sim.params import Env, Launch
from winch_sim.sensitivity import (
    batch_sensitivities,
    format_table,
    sensitivity_unit,
)
from winch_sim.simulate import (
    EVENTS,
    LAUNCH_END_EVENTS,
    T_MAX,
    n_segments_for,
    solve_batch,
    summarize,
    time_series,
    unstack,
)
from winch_sim.winch import engine_winch, tension_winch

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
    ap.add_argument(
        "--segments-per-km",
        type=float,
        default=10.0,
        help="rope resolution: segments per km of laid-out rope (min. 4 segments)",
    )
    ap.add_argument(
        "--field-elevation", type=float, default=0.0, help="airfield height AMSL [m]"
    )
    ap.add_argument(
        "--isa-dt", type=float, default=0.0, help="temperature offset from ISA [K]"
    )
    ap.add_argument(
        "--t-max",
        type=float,
        default=T_MAX,
        help=f"give up if a launch has not ended after this many seconds ({T_MAX:g})",
    )
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
        "--no-sensitivity",
        action="store_true",
        help="skip d(release height)/d(parameter) (computed by default)",
    )
    ap.add_argument("--top", type=int, default=15, help="rows in sensitivity table")
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
                env=Env(
                    wind_ref=a.wind,
                    field_elevation=a.field_elevation,
                    isa_dT=a.isa_dt,
                ),
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
    n_seg = n_segments_for(a.rope_length, a.segments_per_km)
    print(
        f"field {a.field_elevation:.0f} m AMSL, ISA{a.isa_dt:+.0f} K, "
        f"rope model: {n_seg} segments ({a.segments_per_km:g}/km, "
        f"{a.rope_length / n_seg:.0f} m each); speeds in km/h\n"
    )
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{rope_name}_{a.winch}"

    sol = solve_batch(launches, n_segments=n_seg, t_max=a.t_max)

    print(f"{'glider':10s}" + "".join(f"{h:>{W}s}" for _, h, _, _ in COLUMNS))
    runs = {}
    summaries = {}
    for i, (name, L) in enumerate(zip(a.gliders, launches)):
        s_i = unstack(sol, i)
        summ = summaries[name] = summarize(L, s_i)
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

    if a.no_sensitivity:
        return
    print("\nSensitivity of the release height (sorted by |elasticity|)")
    ok = [summaries[n]["event"] in EVENTS[:LAUNCH_END_EVENTS] for n in a.gliders]
    for name, good in zip(a.gliders, ok, strict=True):
        if not good:
            print(f"(skipping {name}: launch ended with {summaries[name]['event']})")
    names = [n for n, good in zip(a.gliders, ok, strict=True) if good]
    launches = [L for L, good in zip(launches, ok, strict=True) if good]
    if not launches:
        return
    results = batch_sensitivities(launches, n_segments=n_seg, t_max=a.t_max)
    for (name, L), (h, rows) in zip(zip(names, launches), results, strict=True):
        h_sim = summaries[name]["release_height"]
        if abs(h - h_sim) > 0.01 * abs(h_sim) + 0.5:
            print(
                f"WARNING: sensitivity solve h = {h:.1f} m vs simulation {h_sim:.1f} m"
            )
        shown = rows
        if a.only:
            shown = [r for r in rows if r.name.startswith(tuple(a.only))]
        print(f"\n=== {name} ===")
        print(format_table(h, shown, a.top))
        fn = a.out / f"sensitivity_{name.replace(' ', '_')}_{tag}.csv"
        write_sensitivity_csv(fn, rows)
        print(f"(all {len(rows)} parameters in {fn})")


def write_sensitivity_csv(fn: Path, rows) -> None:
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


if __name__ == "__main__":
    main()
