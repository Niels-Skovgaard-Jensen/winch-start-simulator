"""Matplotlib figures for one or several launches."""

from collections.abc import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

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


def _release_marker(ax, s: Series, x, y, color):
    i = int(np.argmax(s["released"] > 0)) if s["released"].any() else len(x) - 1
    ax.plot(x[i], y[i], "o", ms=5, color=color, mec=SURFACE, mew=1.5, zorder=5)


def compare(runs: Mapping[str, Series], title: str = "") -> Figure:
    """Trajectory and key time histories for several launches (one colour each)."""
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
    ax_T.set(title="Cable tension at the hook", xlabel="time [s]", ylabel="kN")
    ax_n.set(title="Wing load factor  L / W", xlabel="time [s]", ylabel="–")
    ax_att.set(title="Pitch attitude", xlabel="time [s]", ylabel="deg")
    ax_traj.legend(loc="upper left")
    if title:
        fig.suptitle(title, color=INK, fontsize=12, fontweight="bold")
    return fig


def launch_detail(name: str, s: Series, n_snapshots: int = 8) -> Figure:
    """One launch in detail: rope shapes, forces, angles, winch."""
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
