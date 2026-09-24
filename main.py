"""Simulate winch launches for a set of gliders and compare them.

Examples:
    uv run main.py                                   # all gliders, Dyneema, tension winch
    uv run main.py --gliders "ASK 21" LS4 --rope steel --winch engine
    uv run main.py --rope-length 1000 --wind 5 --out results
"""

import argparse
from pathlib import Path

import winch_sim  # noqa: F401  (enables float64)
from winch_sim import plots
from winch_sim.cable import ROPES
from winch_sim.gliders import CATALOGUE
from winch_sim.params import Env, Launch
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
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()

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
                rope=ROPES[a.rope],
                winch=w,
                pilot=pilot,
                env=Env(wind_ref=a.wind),
                rope_length=a.rope_length,
                slack=0.002,
            )
        )

    sol = solve_batch(launches, n_segments=a.segments)

    print(
        f"rope: {a.rope} {a.rope_length:.0f} m, winch: {a.winch} ({a.pull} x weight), headwind {a.wind} m/s\n"
    )
    print(f"{'glider':10s}" + "".join(f"{h:>{W}s}" for _, h, _, _ in COLUMNS))
    runs = {}
    for i, (name, L) in enumerate(zip(a.gliders, launches)):
        s_i = unstack(sol, i)
        summ = summarize(L, s_i)
        print(f"{name:10s}" + "".join(_cell(summ[k], f, sc) for k, _, f, sc in COLUMNS))
        runs[name] = time_series(L, s_i)

    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.rope}_{a.winch}"
    fig = plots.compare(
        runs, f"Winch launch comparison — {a.rope} rope, {a.winch} winch"
    )
    fig.savefig(a.out / f"compare_{tag}.png", dpi=130)
    for name, s in runs.items():
        fn = a.out / f"detail_{name.replace(' ', '_')}_{tag}.png"
        plots.launch_detail(f"{name} — {a.rope} rope, {a.winch} winch", s).savefig(
            fn, dpi=130
        )
    print(f"\nplots written to {a.out}/")


if __name__ == "__main__":
    main()
