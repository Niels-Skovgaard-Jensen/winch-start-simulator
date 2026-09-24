"""How the importance of the model's components changes with rope length.

For a range of rope lengths, compute the sensitivity of the release height to
every parameter (jax.grad through the solve), group the parameters by component
(rope length, winch, glider, pilot, rope, air & wind), and plot each component's
share of the total |elasticity| as a 100 % stacked area.

    uv run scripts/elasticity_vs_rope_length.py            # compute + plot
    uv run scripts/elasticity_vs_rope_length.py --plot-only

Results (CSV and PNG) go to results/.  Already computed rope lengths are reused,
so an interrupted run can be resumed.
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import winch_sim  # noqa: F401  (float64)
from winch_sim import plots
from winch_sim.cable import ROPES
from winch_sim.gliders import CATALOGUE
from winch_sim.params import Env, Launch
from winch_sim.sensitivity import sensitivities
from winch_sim.simulate import (
    EVENTS,
    LAUNCH_END_EVENTS,
    n_segments_for,
    solve_launch,
    summarize,
)
from winch_sim.winch import engine_winch, tension_winch

LENGTHS = [300, 500, 800, 1200, 2000, 3000, 5000, 8000, 12000, 20000]

# Component groups, in stacking order (bottom to top) = categorical slot order.
GROUPS = [
    "rope length",
    "winch pull",
    "glider",
    "pilot (climb)",
    "end of launch",
    "rope",
]

# The top of the launch: the driver's throttle-down angles, the pilot's
# nose-lowering and the release/back-release angles.  Their local elasticities
# are large and change sign with rope length (they set *when* the launch ends),
# so they get their own component instead of blurring "winch" and "pilot".
END_OF_LAUNCH = {
    "winch.fade_beta0",
    "winch.fade_beta1",
    "winch.fade_floor",
    "pilot.top_beta0",
    "pilot.top_beta1",
    "pilot.theta_top",
    "pilot.release_angle",
    "pilot.back_release_angle",
}

# Air & wind (headwind, field elevation, ISA offset) default to zero, so their
# elasticity is zero by definition; they are not shown (see the note in the plot).
AIR = "env."

# Left out of the shares: gravity is not a design choice (its elasticity simply
# mirrors the weight), and for this winch model the throttle and F_max scale the
# pull identically, so counting both would count the winch pull twice.
EXCLUDE = {"env.g", "winch.throttle"}


def group_of(name: str) -> str | None:
    if name in EXCLUDE or name.startswith(AIR):
        return None
    if name == "rope_length":
        return "rope length"
    if name in END_OF_LAUNCH:
        return "end of launch"
    if name.startswith("rope.") or name == "slack":
        return "rope"
    if name.startswith("winch."):
        return "winch pull"
    if name.startswith("pilot."):
        return "pilot (climb)"
    if name.startswith("glider."):
        return "glider"
    raise ValueError(f"no group for {name}")


def make_launch(glider: str, rope: str, winch: str, length: float) -> Launch:
    g, p = CATALOGUE[glider]
    w = tension_winch(g) if winch == "tension" else engine_winch(g)
    return Launch(
        glider=g,
        rope=ROPES[rope],
        winch=w,
        pilot=p,
        env=Env(),
        rope_length=float(length),
        slack=0.002,
    )


def compute(args, csv_path: Path) -> None:
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            done = {float(r["rope_length_m"]) for r in csv.DictReader(f)}
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(
                [
                    "rope_length_m",
                    "n_segments",
                    "event",
                    "release_height_m",
                    "parameter",
                    "unit",
                    "value",
                    "dh_dp",
                    "elasticity",
                ]
            )
        for length in args.lengths:
            if float(length) in done:
                print(f"{length:>6} m: already computed")
                continue
            L = make_launch(args.glider, args.rope, args.winch, length)
            n = n_segments_for(length, args.segments_per_km)
            t0 = time.perf_counter()
            s = summarize(L, solve_launch(L, n_segments=n))
            if s["event"] not in EVENTS[:LAUNCH_END_EVENTS]:
                print(f"{length:>6} m: launch failed ({s['event']}), skipped")
                w.writerow([length, n, s["event"], "", "", "", "", "", ""])
                f.flush()
                continue
            h, rows = sensitivities(L, n_segments=n)
            for r in rows:
                w.writerow(
                    [
                        length,
                        n,
                        s["event"],
                        f"{h:.3f}",
                        r.name,
                        r.unit,
                        f"{r.value:.6g}",
                        f"{r.dh_dp:.6g}",
                        f"{r.elasticity:.6g}",
                    ]
                )
            f.flush()
            print(
                f"{length:>6} m: {n:3d} segments, h = {h:7.1f} m, "
                f"{s['event']}, {time.perf_counter() - t0:5.1f} s",
                flush=True,
            )


def load(csv_path: Path):
    """-> rope lengths, release heights, {group: summed |elasticity|},
    rope-length elasticity, (name, |elasticity|) of the strongest other parameter."""
    data: dict[float, dict] = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if not r["parameter"]:
                continue
            d = data.setdefault(
                float(r["rope_length_m"]),
                {"h": float(r["release_height_m"]), "params": {}},
            )
            d["params"][r["parameter"]] = float(r["elasticity"])
    lengths = np.array(sorted(data))
    heights = np.array([data[x]["h"] for x in lengths])
    summed = {g: np.zeros(len(lengths)) for g in GROUPS}
    strongest = []
    for i, x in enumerate(lengths):
        others = []
        for name, e in data[x]["params"].items():
            g = group_of(name)
            if g is not None:
                summed[g][i] += abs(e)
                if name != "rope_length":
                    others.append((abs(e), name))
        strongest.append(max(others))
    e_rope = summed["rope length"].copy()
    return lengths, heights, summed, e_rope, strongest


def plot(csv_path: Path, png_path: Path, title: str) -> None:
    lengths, heights, summed, e_rope, strongest = load(csv_path)
    total = sum(summed.values())
    share = {g: summed[g] / total for g in GROUPS}
    km = lengths / 1000.0
    colors = dict(zip(GROUPS, plots.SERIES, strict=False))

    fig, (ax_s, ax_e, ax_h) = plt.subplots(
        3,
        1,
        figsize=(10, 11.5),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 2, 1.3]},
    )

    # 1) shares: 100 % stacked area, 2 px surface gaps between the bands
    ax_s.stackplot(
        km,
        *[share[g] * 100 for g in GROUPS],
        colors=[colors[g] for g in GROUPS],
        labels=GROUPS,
        edgecolor=plots.SURFACE,
        linewidth=2,
        alpha=0.9,
    )
    ax_s.set(
        ylim=(0, 100),
        ylabel="share of total |elasticity| [%]",
        title="Which part of the model decides the release height?",
    )
    # direct labels inside the bands, at the x where each band is widest
    bottom = np.zeros(len(km))
    for g in GROUPS:
        top = bottom + share[g] * 100
        width = (top - bottom)[1:-1]  # interior points: labels never clip
        i = int(np.argmax(width)) + 1
        if (top - bottom)[i] > 9:
            ax_s.text(
                km[i],
                (top[i] + bottom[i]) / 2,
                g,
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        bottom = top
    # (a) rope length stops being the largest component (last crossing, so a
    #     single bump does not count); (b) its own elasticity drops below 0.5
    others = np.max([share[g] for g in GROUPS[1:]], axis=0)
    x_largest = crossing(km, share["rope length"] - others, last=True)
    x_half = crossing(km, e_rope - 0.5, last=True)
    marks = [
        (x_largest, "rope length no longer\nthe largest component"),
        (x_half, "rope-length elasticity < 0.5:\n+10 % rope gives < +5 % height"),
    ]
    for x, text in marks:
        if x is None:
            continue
        for ax in (ax_s, ax_e, ax_h):
            ax.axvline(x, color=plots.INK_2, lw=1, ls="--", zorder=5)
        ax_s.annotate(
            f"{text}\n≈ {x:.1f} km",
            (x, 99),
            xytext=(5, -2),
            textcoords="offset points",
            va="top",
            fontsize=8.5,
            color=plots.INK,
            bbox={"fc": plots.SURFACE, "ec": "none", "pad": 2},
        )
    ax_s.legend(
        loc="lower left",
        ncols=3,
        frameon=True,
        framealpha=0.95,
        facecolor=plots.SURFACE,
        edgecolor="none",
        reverse=True,
    )

    # 2) absolute summed |elasticity| per component (same scale for all lines)
    for g in GROUPS:
        ax_e.plot(
            km,
            summed[g],
            color=colors[g],
            lw=2,
            marker="o",
            ms=4,
            mec=plots.SURFACE,
            mew=1,
            label=g,
        )
    ax_e.axhline(0.5, color=plots.INK_2, lw=0.8, ls=":")
    ymax = max(float(v.max()) for v in summed.values()) * 1.08
    ax_e.set_ylim(0, ymax)
    # end labels, nudged apart so they never overlap
    placed: list[float] = []
    for y, g in sorted((float(summed[g][-1]), g) for g in GROUPS):
        y_lab = max(y, placed[-1] + 0.06 * ymax) if placed else y
        placed.append(y_lab)
        ax_e.annotate(
            g,
            (km[-1], y),
            xytext=(km[-1] * 1.12, y_lab),
            textcoords="data",
            va="center",
            fontsize=8.5,
            color=plots.INK,
            arrowprops={"arrowstyle": "-", "color": plots.INK_2, "lw": 0.6},
        )
    ax_e.set(
        ylabel="summed |elasticity|",
        title="Absolute size of each component (rope length is a single parameter; "
        "dotted: 0.5)",
    )

    # 3) context: the release height itself
    ax_h.plot(
        km, heights, color=plots.INK_2, lw=2, marker="o", ms=4, mec=plots.SURFACE, mew=1
    )
    ax_h.set(
        ylabel="release height [m]", xlabel="rope length [km]", title="Release height"
    )
    ax_h.set_ylim(bottom=0)
    ax_h.set_xscale("log")
    ticks = [x for x in (0.3, 0.5, 1, 2, 3, 5, 10, 20) if km[0] <= x <= km[-1]]
    ax_h.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax_h.minorticks_off()
    ax_e.set_xlim(km[0], km[-1] * 1.6)  # room for the end labels

    fig.suptitle(title, color=plots.INK, fontsize=12, fontweight="bold")
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.965))  # room for the note
    fig.text(
        0.01,
        0.002,
        "Share = summed |elasticity| of a component's parameters / total. "
        "Not shown: gravity (mirrors the weight), the winch throttle (same "
        "effect as the pull F_max), and air & wind (headwind, field elevation, "
        "ISA offset), whose defaults are zero and so have zero elasticity.",
        fontsize=7.5,
        color=plots.INK_2,
        wrap=True,
    )
    fig.savefig(png_path, dpi=130)
    print(f"plot written to {png_path}")

    # table view of the plotted numbers
    print(
        f"\n{'length':>8} {'h':>7} {'e_rope':>7} "
        + " ".join(f"{g:>14}" for g in GROUPS)
        + "   strongest other parameter"
    )
    for i, x in enumerate(lengths):
        e_o, name_o = strongest[i]
        print(
            f"{x:>6.0f} m {heights[i]:>6.0f} m {e_rope[i]:>7.2f} "
            + " ".join(f"{share[g][i] * 100:>13.1f}%" for g in GROUPS)
            + f"   {name_o} ({e_o:.2f})"
        )
    for x, text in marks:
        where = "not reached" if x is None else f"{x:.1f} km"
        print(f"{text.replace(chr(10), ' ')}: {where}")


def crossing(x, y, last: bool = False):
    """x where y goes from positive to <= 0 (log-linear interpolation); the first
    such crossing, or the last one if y ends negative and `last` is set."""
    found = []
    for i in range(len(y) - 1):
        if y[i] > 0 >= y[i + 1]:
            f = y[i] / (y[i] - y[i + 1])
            lx = np.log(x[i]) + f * (np.log(x[i + 1]) - np.log(x[i]))
            found.append(float(np.exp(lx)))
    if not found:
        return None
    return found[-1] if last else found[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--glider", default="ASK 21", choices=list(CATALOGUE))
    ap.add_argument("--rope", default="dyneema", choices=list(ROPES))
    ap.add_argument("--winch", default="tension", choices=["tension", "engine"])
    ap.add_argument("--lengths", type=float, nargs="+", default=LENGTHS)
    ap.add_argument("--segments-per-km", type=float, default=10.0)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.glider.replace(' ', '_')}_{a.rope}_{a.winch}"
    csv_path = a.out / f"elasticity_vs_rope_length_{tag}.csv"
    if not a.plot_only:
        compute(a, csv_path)
    plot(
        csv_path,
        a.out / f"elasticity_vs_rope_length_{tag}.png",
        f"{a.glider}, {a.rope} rope, {a.winch} winch",
    )


if __name__ == "__main__":
    main()
