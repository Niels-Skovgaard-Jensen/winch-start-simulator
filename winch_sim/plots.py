"""Matplotlib figures for one or several launches."""

from collections.abc import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .measured import ALIGN_MODES, line_up

# Categorical slots in fixed order (validated reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "font.size": 9,
    }
)

Series = Mapping[str, np.ndarray]
# measured tension: {name: (time [s], tension [N])}, see measured.traces
Traces = Mapping[str, tuple[np.ndarray, np.ndarray]]
GAP_S = 2.0  # don't draw a measured line across a dropout longer than this


def gapped(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs where samples are far apart, so lines don't bridge dropouts."""
    breaks = np.flatnonzero(np.diff(t) > GAP_S) + 1
    return np.insert(t, breaks, np.nan), np.insert(y.astype(float), breaks, np.nan)


def _measured(
    ax, measured: Traces | None, first_colour: int = 0, t_sim: tuple | None = None
):
    """Measured tension (kN) as thin lines, continuing the colour cycle.

    With the simulated time range `t_sim` the view stays on it (plus the first
    seconds after release); pan or zoom out for the rest of the measured trace.
    """
    if not measured:
        return
    for k, (name, (t, T)) in enumerate(measured.items()):
        c = SERIES[(first_colour + k) % len(SERIES)]
        ax.plot(
            *gapped(t, T / 1e3), color=c, lw=0.9, alpha=0.9, label=f"measured: {name}"
        )
    if t_sim is not None:
        ax.set_xlim(t_sim[0] - 5.0, t_sim[1] + 15.0)
    ax.legend(loc="upper right", fontsize=7)


def _release_marker(ax, s: Series, x, y, color):
    i = int(np.argmax(s["released"] > 0)) if s["released"].any() else len(x) - 1
    ax.plot(x[i], y[i], "o", ms=5, color=color, mec=SURFACE, mew=1.5, zorder=5)


def compare(
    runs: Mapping[str, Series], title: str = "", measured: Traces | None = None
) -> Figure:
    """Trajectory and key time histories for several launches (one colour each).

    `measured` tension traces are drawn on the hook-tension panel.
    """
    fig, axs = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    ax_traj, ax_h, ax_V, ax_T, ax_n, ax_att = axs.ravel()
    for k, (name, s) in enumerate(runs.items()):
        c = SERIES[k % len(SERIES)]
        t = s["t"]
        ax_traj.plot(s["x"], s["height"], color=c, label=name)
        _release_marker(ax_traj, s, s["x"], s["height"], c)
        ax_h.plot(t, s["height"], color=c, label=name)
        _release_marker(ax_h, s, t, s["height"], c)
        ax_V.plot(t, s["V_ias"] * 3.6, color=c, label=name)
        att = s["released"] == 0
        ax_T.plot(t[att], s["T_hook"][att] / 1e3, color=c, label=name)
        ax_n.plot(t, s["n_wing"], color=c, label=name)
        ax_att.plot(t, np.degrees(s["theta"]), color=c, label=name)

    ax_traj.set(
        title="Trajectory", xlabel="distance along runway [m]", ylabel="height [m]"
    )
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_h.set(title="Height", xlabel="time [s]", ylabel="m")
    ax_V.set(title="Indicated airspeed (IAS)", xlabel="time [s]", ylabel="km/h")
    t_sim = (
        (min(s["t"][0] for s in runs.values()), max(s["t"][-1] for s in runs.values()))
        if runs
        else None
    )
    _measured(ax_T, measured, len(runs), t_sim)
    ax_T.set(title="Cable tension at the hook", xlabel="time [s]", ylabel="kN")
    ax_n.set(title="Wing load factor  L / W", xlabel="time [s]", ylabel="–")
    ax_att.set(title="Pitch attitude", xlabel="time [s]", ylabel="deg")
    if runs:
        ax_traj.legend(loc="upper left")
    if title:
        fig.suptitle(title, color=INK, fontsize=12, fontweight="bold")
    return fig


def launch_detail(
    name: str, s: Series, n_snapshots: int = 8, measured: Traces | None = None
) -> Figure:
    """One launch in detail: rope shapes, forces, angles, winch.

    `measured` tension traces are drawn on the rope-tension panel.
    """
    fig, axs = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    ax_rope, ax_T, ax_V, ax_ang, ax_w, ax_ctl = axs.ravel()
    t = s["t"]
    att = s["released"] == 0

    idx = np.linspace(0, int(att.sum()) - 1, n_snapshots).astype(int)
    shades = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, n_snapshots))
    for j, i in enumerate(idx):
        ax_rope.plot(s["rope_x"][i], s["rope_z"][i], color=shades[j], lw=1.2)
        ax_rope.plot(s["x"][i], s["z"][i], "o", ms=4, color=shades[j])
        ax_rope.annotate(
            f"{t[i]:.0f} s",
            (s["x"][i], s["z"][i]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color=INK_2,
        )
    ax_rope.plot(s["x"], s["z"], color=INK_2, lw=0.8, ls="--")
    ax_rope.set(
        title="Rope shape (sag) during the launch", xlabel="x [m]", ylabel="z [m]"
    )
    ax_rope.set_aspect("equal", adjustable="datalim")

    ax_T.plot(t[att], s["T_hook"][att] / 1e3, color=SERIES[0], label="at hook")
    ax_T.plot(t[att], s["T_winch"][att] / 1e3, color=SERIES[1], label="at winch")
    ax_T.plot(t[att], s["winch_pull"][att] / 1e3, color=SERIES[2], label="winch pull")
    ax_T.set(title="Rope tension", xlabel="time [s]", ylabel="kN")
    ax_T.legend()
    _measured(ax_T, measured, 3, (t[0], t[-1]))

    ax_V.plot(t, s["V_ias"] * 3.6, color=SERIES[0], label="indicated airspeed")
    ax_V.plot(t, s["V_tas"] * 3.6, color=SERIES[0], ls="--", label="true airspeed")
    ax_V.plot(t, s["v_reel"] * 3.6, color=SERIES[1], label="reel-in speed")
    ax_V.set(title="Speeds", xlabel="time [s]", ylabel="km/h")
    ax_V.legend()

    ax_ang.plot(t, np.degrees(s["theta"]), color=SERIES[0], label="pitch θ")
    moving = s["V_tas"] > 5.0  # alpha is meaningless while (nearly) standing still
    alpha = np.where(moving, np.degrees(s["alpha"]), np.nan)
    ax_ang.plot(t, alpha, color=SERIES[1], label="angle of attack α")
    ax_ang.plot(
        t[att],
        np.degrees(s["cable_angle"][att]),
        color=SERIES[2],
        label="cable angle at hook",
    )
    ax_ang.plot(t, np.degrees(s["beta"]), color=SERIES[3], label="elevation from winch")
    ax_ang.set(title="Angles", xlabel="time [s]", ylabel="deg")
    ax_ang.legend()

    ax_w.plot(t[att], s["winch_power"][att] / 1e3, color=SERIES[0])
    ax_w.set(
        title="Power delivered into the rope at the winch",
        xlabel="time [s]",
        ylabel="kW",
    )

    ax_ctl.plot(t, np.degrees(s["de"]), color=SERIES[0])
    ax_ctl.set(
        title="Pilot elevator input δe (− = nose up)", xlabel="time [s]", ylabel="deg"
    )
    fig.suptitle(name, color=INK, fontsize=12, fontweight="bold")
    return fig


def measured_launches(launches, mode: str = "release", zero: bool = True) -> Figure:
    """Measured launches lined up at release (or the start of the pull) at t = 0,
    with the detected start of the pull (o) and release (v)."""
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    for k, m in enumerate(launches):
        c = SERIES[k % len(SERIES)]
        t, T = line_up(m, mode), m.tension(zero) / 1e3
        label = m.name if m.start_recorded else f"{m.name} (start not recorded)"
        ax.plot(*gapped(t, T), color=c, lw=1.0, label=label)
        for tk, mk in ((m.t_pull, "o"), (m.t_release, "v")):
            i = int(np.searchsorted(m.t, tk))
            filled = mk == "v" or m.start_recorded
            ax.plot(
                t[i],
                T[i],
                mk,
                ms=7,
                color=c,
                mec=c,
                mfc=c if filled else SURFACE,
                mew=1.5,
                zorder=5,
            )
    ax.set(
        title="Measured rope tension, lined up "
        f"{ALIGN_MODES[mode]} (t = 0); o start of the pull, v release",
        xlabel="time [s]",
        ylabel="kN" if zero else "kN (sensor offset included)",
    )
    ax.legend(loc="upper left")
    return fig
