"""Qt desktop app: edit every model setting, run launches, visualise and compare them.

Start with `uv run winch-gui` (or `uv run python -m winch_sim.gui`).
"""

import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

import winch_sim  # noqa: F401  (float64)

from .. import plots
from ..cable import ROPES, load_rope
from ..gliders import CATALOGUE
from ..params import Env, Glider, Launch, Pilot, Rope, Winch
from ..report import COLUMNS, write_sensitivity_csv
from ..sensitivity import sensitivities, sensitivity_unit
from ..simulate import (
    EVENTS,
    LAUNCH_END_EVENTS,
    T_MAX,
    n_segments_for,
    solve_launch,
    summarize,
    time_series,
)
from ..winch import engine_winch, tension_winch
from .forms import FloatEdit, ModuleForm, scrolled

WINCH_PRESETS = ("tension", "engine")


@dataclasses.dataclass
class Numerics:
    segments_per_km: float = 10.0
    t_max: float = T_MAX
    rtol: float = 1e-6
    atol: float = 1e-6
    t_post: float = 8.0
    sensitivity: bool = True
    sensitivity_tol: float = 1e-8


@dataclasses.dataclass
class RunResult:
    name: str
    launch: Launch
    numerics: Numerics
    n_segments: int
    summary: dict
    series: dict
    sens: tuple | None  # (height, rows)
    note: str
    elapsed: float


def default_launch(glider="ASK 21", rope="dyneema") -> Launch:
    g, p = CATALOGUE[glider]
    return Launch(
        glider=g,
        rope=ROPES[rope],
        winch=tension_winch(g),
        pilot=p,
        env=Env(),
        rope_length=1200.0,
        slack=0.002,
    )


def run_launch(name: str, launch: Launch, num: Numerics) -> RunResult:
    """Simulate one launch (+ sensitivities).  Pure function, runs in a worker."""
    t0 = time.perf_counter()
    n = n_segments_for(launch.rope_length, num.segments_per_km)
    sol = solve_launch(
        launch,
        n_segments=n,
        t_max=num.t_max,
        t_post=num.t_post,
        rtol=num.rtol,
        atol=num.atol,
    )
    summary = summarize(launch, sol)
    series = time_series(launch, sol)
    sens, note = None, ""
    if num.sensitivity:
        if summary["event"] in EVENTS[:LAUNCH_END_EVENTS]:
            sens = sensitivities(
                launch, n_segments=n, tol=num.sensitivity_tol, t_max=num.t_max
            )
        else:
            note = f"no sensitivities: launch ended with {summary['event']}"
    return RunResult(
        name, launch, num, n, summary, series, sens, note, time.perf_counter() - t0
    )


class Worker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, name, launch, numerics):
        super().__init__()
        self.args = (name, launch, numerics)

    @QtCore.Slot()
    def run(self):
        try:
            self.finished.emit(run_launch(*self.args))
        except Exception:  # noqa: BLE001  (report anything to the user)
            self.failed.emit(traceback.format_exc())


class FigurePane(QtWidgets.QWidget):
    """Holds one matplotlib figure with the standard navigation toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.canvas = None
        self.toolbar = None
        self.placeholder = QtWidgets.QLabel("Run a simulation to see results here.")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lay.addWidget(self.placeholder)

    def set_figure(self, fig: Figure | None):
        for w in (self.canvas, self.toolbar):
            if w is not None:
                self.lay.removeWidget(w)
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.canvas = self.toolbar = None
        self.placeholder.setVisible(fig is None)
        if fig is None:
            return
        plt.close(fig)  # detach from pyplot; the canvas keeps the figure alive
        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.lay.addWidget(self.toolbar)
        self.lay.addWidget(self.canvas, 1)
        self.canvas.draw_idle()


class NumericsForm(QtWidgets.QWidget):
    def __init__(self, num: Numerics, parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        self.edits: dict[str, FloatEdit] = {}
        tips = {
            "segments_per_km": "rope resolution (min. 4 segments in total)",
            "t_max": "give up if the launch has not ended by then [s]",
            "rtol": "solver relative tolerance",
            "atol": "solver absolute tolerance",
            "t_post": "free flight simulated after release [s]",
            "sensitivity_tol": "solver tolerance for the gradient solve",
        }
        for f in dataclasses.fields(num):
            if f.type is bool or f.name == "sensitivity":
                continue
            e = FloatEdit()
            e.setToolTip(tips.get(f.name, ""))
            self.edits[f.name] = e
            form.addRow(f.name, e)
        self.sens = QtWidgets.QCheckBox("compute sensitivities d(h)/d(parameter)")
        form.addRow(self.sens)
        self.set(num)

    def set(self, num: Numerics):
        for k, e in self.edits.items():
            e.set_value(getattr(num, k))
        self.sens.setChecked(num.sensitivity)

    def get(self) -> Numerics:
        bad = [k for k, e in self.edits.items() if not e.is_valid()]
        if bad:
            raise ValueError(f"numerics: invalid value for {', '.join(bad)}")
        return Numerics(
            **{k: e.value() for k, e in self.edits.items()},
            sensitivity=self.sens.isChecked(),
        )


# event -> (headline, marker, colour); colours: good / warning / critical / neutral
RELEASE_STYLE = {
    "release": ("PILOT RELEASE", "v", "#1a7f37"),
    "back_release": ("BACK-RELEASE", "^", "#b45309"),
    "weak_link": ("WEAK LINK BREAK", "X", "#c62828"),
    "rope_in": ("ROPE REELED IN", "s", "#52514e"),
    "no_liftoff": ("NO LIFT-OFF", "s", "#52514e"),
    "t_max": ("TIME LIMIT REACHED", "s", "#52514e"),
}


def release_info(run: RunResult) -> tuple[int, str, str, str, str]:
    """(index of the release sample, headline, detail, marker, colour)."""
    s = run.series
    released = np.flatnonzero(s["released"] > 0)
    i = int(released[0]) - 1 if released.size else len(s["t"]) - 1
    i = max(i, 0)
    event = str(run.summary["event"])
    head, marker, color = RELEASE_STYLE.get(event, (event.upper(), "s", "#52514e"))
    L = run.launch
    detail = {
        "release": f"cable angle at hook {np.degrees(s['cable_angle'][i]):.0f}° "
        f"(pilot releases at {np.degrees(float(L.pilot.release_angle)):.0f}°)",
        "back_release": f"cable {np.degrees(s['cable_body'][i]):.0f}° below the "
        f"fuselage axis (hook trips at "
        f"{np.degrees(float(L.pilot.back_release_angle)):.0f}°)",
        "weak_link": f"hook tension {s['T_hook'][i] / 1e3:.2f} kN reached the weak "
        f"link rating {float(L.glider.weak_link) / 1e3:.2f} kN",
        "rope_in": "less than 30 m of rope left out",
        "no_liftoff": "still on the ground after 120 s",
        "t_max": "the launch did not end within t_max",
    }.get(event, "")
    detail = f"t = {s['t'][i]:.1f} s, h = {s['height'][i]:.0f} m: {detail}"
    return i, head, detail, marker, color


class RopeView(QtWidgets.QWidget):
    """Rope shape and glider at a chosen time, with a slider and playback."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.run: RunResult | None = None
        lay = QtWidgets.QVBoxLayout(self)
        self.fig = Figure(figsize=(9, 5), layout="constrained")
        self.ax = self.fig.add_subplot()
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)
        ctl = QtWidgets.QHBoxLayout()
        self.play = QtWidgets.QPushButton("▶ Play")
        self.play.setCheckable(True)
        self.play.toggled.connect(self._toggle)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._draw)
        ctl.addWidget(self.play)
        ctl.addWidget(self.slider, 1)
        lay.addLayout(ctl)
        self.readout = QtWidgets.QLabel()
        self.readout.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        lay.addWidget(self.readout)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._step)

    def set_run(self, run: RunResult | None):
        self.run = run
        self.ax.clear()
        if run is None:
            self.canvas.draw_idle()
            return
        s = run.series
        n = len(s["t"])
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        ax = self.ax
        ax.plot(s["x"], s["z"], color=plots.INK_2, lw=0.8, ls="--", label="glider path")
        ax.plot(s["rope_x"][0, :1], s["rope_z"][0, :1], "s", color=plots.INK, ms=7)
        (self.rope_line,) = ax.plot([], [], color=plots.SERIES[0], lw=1.6, label="rope")
        (self.glider_pt,) = ax.plot(
            [], [], "o", color=plots.SERIES[1], ms=8, mec=plots.SURFACE, label="glider"
        )
        xs = np.concatenate([s["x"], s["rope_x"].ravel()])
        zs = np.concatenate([s["z"], s["rope_z"].ravel()])
        pad = 0.05 * (xs.max() - xs.min() + 1)
        # invisible bounding points so the view covers the whole launch
        ax.plot(
            [xs.min() - pad, xs.max() + pad],
            [min(zs.min(), 0) - pad, zs.max() + pad],
            alpha=0,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xlabel="x [m]", ylabel="z [m]", title=run.name)
        # end-of-launch indicator: marker where it happened + banner, shown once the
        # animation reaches that moment
        self.i_rel, head, detail, marker, color = release_info(run)
        (self.release_pt,) = ax.plot(
            [s["x"][self.i_rel]],
            [s["z"][self.i_rel]],
            marker,
            color=color,
            ms=12,
            mec=plots.SURFACE,
            mew=1.5,
            zorder=6,
            label=head.lower(),
        )
        self.banner = ax.text(
            0.98,
            0.96,
            f"{head}\n{detail}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=color,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.4", "fc": plots.SURFACE, "ec": color},
            zorder=7,
        )
        self.release_head = head
        ax.legend(loc="upper left")
        self._draw(0)

    def _toggle(self, on: bool):
        self.play.setText("⏸ Pause" if on else "▶ Play")
        if on and self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
        (self.timer.start if on else self.timer.stop)()

    def _step(self):
        step = max(1, (self.slider.maximum() + 1) // 400)
        v = self.slider.value() + step
        if v > self.slider.maximum():
            self.play.setChecked(False)
            v = self.slider.maximum()
        self.slider.setValue(v)

    def _draw(self, i: int):
        if self.run is None:
            return
        s = self.run.series
        self.rope_line.set_data(s["rope_x"][i], s["rope_z"][i])
        self.glider_pt.set_data([s["x"][i]], [s["z"][i]])
        ended = i >= self.i_rel
        self.release_pt.set_visible(ended)
        self.banner.set_visible(ended)
        phase = f"after {self.release_head.lower()}" if ended else "on the cable"
        self.readout.setText(
            f"t = {s['t'][i]:.1f} s  ({phase})   height {s['height'][i]:.0f} m   "
            f"IAS {s['V_ias'][i] * 3.6:.0f} km/h   TAS {s['V_tas'][i] * 3.6:.0f} km/h"
            f"   T hook {s['T_hook'][i] / 1e3:.2f} kN   T winch "
            f"{s['T_winch'][i] / 1e3:.2f} kN   θ {np.degrees(s['theta'][i]):.0f}°   "
            f"α {np.degrees(s['alpha'][i]):.1f}°   n {s['n_wing'][i]:.2f}   "
            f"ρ {s['rho'][i]:.3f} kg/m³"
        )
        self.canvas.draw_idle()


def num_item(x: float, fmt: str) -> QtWidgets.QTableWidgetItem:
    """Table item that shows `fmt` but sorts numerically."""
    it = QtWidgets.QTableWidgetItem()
    it.setData(QtCore.Qt.ItemDataRole.DisplayRole, float(f"{x:{fmt}}"))
    it.setTextAlignment(
        QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
    )
    return it


class SensitivityView(QtWidgets.QWidget):
    HEAD = (
        "rank",
        "parameter",
        "value",
        "unit",
        "dh/dp",
        "per",
        "dh(+10%) m",
        "elasticity",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.run: RunResult | None = None
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.info = QtWidgets.QLabel()
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("filter, e.g. rope  or  glider.C")
        self.filter.textChanged.connect(self._fill)
        export = QtWidgets.QPushButton("Export CSV…")
        export.clicked.connect(self._export)
        top.addWidget(self.info, 1)
        top.addWidget(self.filter)
        top.addWidget(export)
        lay.addLayout(top)
        self.table = QtWidgets.QTableWidget(0, len(self.HEAD))
        self.table.setHorizontalHeaderLabels(list(self.HEAD))
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        lay.addWidget(self.table, 1)
        lay.addWidget(
            QtWidgets.QLabel(
                "rank: by |elasticity|.  dh(+10%): change of release height for a +10 % change of the "
                "parameter alone.  elasticity: % height per % parameter.  Local "
                "(linearised) values from jax.grad through the ODE solve."
            )
        )

    def set_run(self, run: RunResult | None):
        self.run = run
        self._fill()

    def _fill(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        run = self.run
        if run is None or run.sens is None:
            self.info.setText(run.note if run and run.note else "no sensitivities")
            return
        h, rows = run.sens
        self.info.setText(f"{run.name}: release height h = {h:.1f} m")
        key = self.filter.text().strip()
        rows = [(k + 1, r) for k, r in enumerate(rows) if not key or key in r.name]
        self.table.setRowCount(len(rows))
        for i, (rank, r) in enumerate(rows):
            items = [
                num_item(rank, "d"),
                QtWidgets.QTableWidgetItem(r.name),
                num_item(r.value, ".6g"),
                QtWidgets.QTableWidgetItem(r.unit),
                num_item(r.dh_dp, ".4g"),
                QtWidgets.QTableWidgetItem(sensitivity_unit(r.unit)),
                num_item(r.dh_10pct, ".2f"),
                num_item(r.elasticity, ".4f"),
            ]
            for j, it in enumerate(items):
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)
        # default order: rank = sorted by |elasticity|; click a header to re-sort
        self.table.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)
        self.table.resizeColumnsToContents()

    def _export(self):
        if self.run is None or self.run.sens is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export sensitivities", "sensitivity.csv", "CSV (*.csv)"
        )
        if fn:
            write_sensitivity_csv(fn, self.run.sens[1])


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Winch launch simulator")
        self.resize(1560, 980)
        self.runs: list[RunResult] = []
        self._thread: QtCore.QThread | None = None
        self._worker: Worker | None = None
        self._run_counter = 0

        L = default_launch()
        self.glider_form = ModuleForm(L.glider)
        self.pilot_form = ModuleForm(L.pilot)
        self.rope_form = ModuleForm(L.rope)
        self.winch_form = ModuleForm(L.winch)
        self.env_form = ModuleForm(L.env)
        self.launch_form = ModuleForm(L)  # rope_length, slack (nested modules skipped)
        self.num_form = NumericsForm(Numerics())

        split = QtWidgets.QSplitter()
        split.addWidget(self._settings_panel())
        split.addWidget(self._results_panel())
        split.setStretchFactor(1, 1)
        split.setSizes([560, 1000])
        self.setCentralWidget(split)
        self._menus()
        self.statusBar().showMessage(
            "Ready.  The first run compiles the model (~10 s)."
        )

    # ---- layout -----------------------------------------------------------------

    def _settings_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(540)
        lay = QtWidgets.QVBoxLayout(panel)

        presets = QtWidgets.QGroupBox("Presets (fill the settings below; edit freely)")
        grid = QtWidgets.QGridLayout(presets)
        self.glider_combo = QtWidgets.QComboBox()
        self.glider_combo.addItems(list(CATALOGUE))
        self.glider_combo.setCurrentText("ASK 21")
        self.glider_combo.currentTextChanged.connect(self._apply_glider)
        grid.addWidget(QtWidgets.QLabel("Glider + pilot"), 0, 0)
        grid.addWidget(self.glider_combo, 0, 1, 1, 2)

        self.rope_combo = QtWidgets.QComboBox()
        self.rope_combo.addItems(list(ROPES))
        self.rope_combo.setCurrentText("dyneema")
        self.rope_combo.currentTextChanged.connect(self._apply_rope)
        load_rope_btn = QtWidgets.QPushButton("Load TOML…")
        load_rope_btn.clicked.connect(self._load_rope)
        grid.addWidget(QtWidgets.QLabel("Rope"), 1, 0)
        grid.addWidget(self.rope_combo, 1, 1)
        grid.addWidget(load_rope_btn, 1, 2)

        self.winch_combo = QtWidgets.QComboBox()
        self.winch_combo.addItems(WINCH_PRESETS)
        self.pull_edit = FloatEdit()
        self.pull_edit.set_value(1.1)
        self.pull_edit.setToolTip(
            "winch pull / glider weight (tension winch), or "
            "pull at 15 m/s reel speed / weight (engine winch)"
        )
        apply_winch = QtWidgets.QPushButton("Apply")
        apply_winch.setToolTip(
            "recompute the winch settings for the current glider mass"
        )
        apply_winch.clicked.connect(self._apply_winch)
        self.winch_combo.currentTextChanged.connect(self._apply_winch)
        grid.addWidget(QtWidgets.QLabel("Winch"), 2, 0)
        wrow = QtWidgets.QHBoxLayout()
        wrow.addWidget(self.winch_combo)
        wrow.addWidget(QtWidgets.QLabel("pull/weight"))
        wrow.addWidget(self.pull_edit)
        grid.addLayout(wrow, 2, 1)
        grid.addWidget(apply_winch, 2, 2)
        lay.addWidget(presets)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(scrolled(self._launch_env_tab()), "Launch && air")
        tabs.addTab(scrolled(self.glider_form), "Glider")
        tabs.addTab(scrolled(self.pilot_form), "Pilot")
        tabs.addTab(scrolled(self.rope_form), "Rope")
        tabs.addTab(scrolled(self.winch_form), "Winch")
        tabs.addTab(scrolled(self.num_form), "Numerics")
        lay.addWidget(tabs, 1)

        run_box = QtWidgets.QHBoxLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("run label (optional)")
        self.run_btn = QtWidgets.QPushButton("Run simulation")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self._start_run)
        self.busy = QtWidgets.QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumWidth(120)
        self.busy.hide()
        run_box.addWidget(self.name_edit, 1)
        run_box.addWidget(self.busy)
        run_box.addWidget(self.run_btn)
        lay.addLayout(run_box)
        return panel

    def _launch_env_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("<b>Launch layout</b>"))
        lay.addWidget(self.launch_form)
        lay.addWidget(QtWidgets.QLabel("<b>Atmosphere and wind (ISA)</b>"))
        lay.addWidget(self.env_form)
        lay.addStretch(1)
        return w

    def _results_panel(self) -> QtWidgets.QWidget:
        self.results = QtWidgets.QTabWidget()

        runs = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(runs)
        heads = (
            ["compare", "run", "segments"] + [h for _, h, _, _ in COLUMNS] + ["time s"]
        )
        self.runs_table = QtWidgets.QTableWidget(0, len(heads))
        self.runs_table.setHorizontalHeaderLabels(heads)
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.runs_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.runs_table.itemChanged.connect(self._compare_changed)
        rl.addWidget(self.runs_table, 1)
        btns = QtWidgets.QHBoxLayout()
        for text, slot in [
            ("Load settings of selected run", self._load_run_settings),
            ("Export time series of selected run…", self._export_series),
            ("Remove selected", self._remove_selected),
            ("Clear all", self._clear_runs),
        ]:
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            btns.addWidget(b)
        btns.addStretch(1)
        rl.addLayout(btns)
        self.results.addTab(runs, "Runs")

        compare = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(compare)
        cbtns = QtWidgets.QHBoxLayout()
        self.compare_info = QtWidgets.QLabel()
        clear = QtWidgets.QPushButton("Clear comparison")
        clear.setToolTip(
            "untick all runs (runs are kept; new runs are added to the comparison)"
        )
        clear.clicked.connect(lambda: self._set_all_compared(False))
        select_all = QtWidgets.QPushButton("Compare all runs")
        select_all.clicked.connect(lambda: self._set_all_compared(True))
        cbtns.addWidget(self.compare_info, 1)
        cbtns.addWidget(select_all)
        cbtns.addWidget(clear)
        cl.addLayout(cbtns)
        self.compare_pane = FigurePane()
        cl.addWidget(self.compare_pane, 1)
        self.results.addTab(compare, "Comparison")

        detail = QtWidgets.QWidget()
        dl = QtWidgets.QVBoxLayout(detail)
        self.detail_combo = QtWidgets.QComboBox()
        self.detail_combo.currentIndexChanged.connect(self._show_detail)
        dl.addWidget(self.detail_combo)
        self.detail_pane = FigurePane()
        dl.addWidget(self.detail_pane, 1)
        self.results.addTab(detail, "Detail")

        rope = QtWidgets.QWidget()
        rpl = QtWidgets.QVBoxLayout(rope)
        self.rope_combo_run = QtWidgets.QComboBox()
        self.rope_combo_run.currentIndexChanged.connect(
            lambda i: self.rope_view.set_run(self._run_at(i))
        )
        rpl.addWidget(self.rope_combo_run)
        self.rope_view = RopeView()
        rpl.addWidget(self.rope_view, 1)
        self.results.addTab(rope, "Rope && glider")

        sens = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(sens)
        self.sens_combo = QtWidgets.QComboBox()
        self.sens_combo.currentIndexChanged.connect(
            lambda i: self.sens_view.set_run(self._run_at(i))
        )
        sl.addWidget(self.sens_combo)
        self.sens_view = SensitivityView()
        sl.addWidget(self.sens_view, 1)
        self.results.addTab(sens, "Sensitivity")
        return self.results

    def _menus(self):
        m = self.menuBar().addMenu("&File")
        for text, slot, key in [
            (
                "Save settings…",
                self._save_settings,
                QtGui.QKeySequence.StandardKey.Save,
            ),
            (
                "Load settings…",
                self._load_settings,
                QtGui.QKeySequence.StandardKey.Open,
            ),
            ("Load rope (TOML)…", self._load_rope, None),
            ("Quit", self.close, QtGui.QKeySequence.StandardKey.Quit),
        ]:
            act = m.addAction(text)
            act.triggered.connect(slot)
            if key is not None:
                act.setShortcut(key)
        r = self.menuBar().addMenu("&Run")
        act = r.addAction("Run simulation")
        act.setShortcut("Ctrl+R")
        act.triggered.connect(self._start_run)

    # ---- presets ----------------------------------------------------------------

    def _apply_glider(self, name: str):
        g, p = CATALOGUE[name]
        self.glider_form.set_instance(g)
        self.pilot_form.set_instance(p)
        self._apply_winch()  # the winch preset scales with the glider weight

    def _apply_rope(self, name: str):
        self.rope_form.set_instance(ROPES[name])

    def _load_rope(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load rope", "ropes", "TOML (*.toml)"
        )
        if not fn:
            return
        try:
            self.rope_form.set_instance(load_rope(fn))
            self.statusBar().showMessage(f"rope loaded from {fn}")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Rope", str(e))

    def _apply_winch(self, *_):
        try:
            glider = self.glider_form.instance()
            pull = self.pull_edit.value()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Winch preset", str(e))
            return
        if self.winch_combo.currentText() == "tension":
            w = tension_winch(glider, tension_per_weight=pull)
        else:
            w = engine_winch(glider, pull_per_weight=pull)
        self.winch_form.set_instance(w)

    # ---- settings <-> launch ------------------------------------------------------

    def current_launch(self) -> Launch:
        return Launch(
            glider=self.glider_form.instance(),
            rope=self.rope_form.instance(),
            winch=self.winch_form.instance(),
            pilot=self.pilot_form.instance(),
            env=self.env_form.instance(),
            **self.launch_form.values_si(),
        )

    def set_launch(self, L: Launch):
        self.glider_form.set_instance(L.glider)
        self.pilot_form.set_instance(L.pilot)
        self.rope_form.set_instance(L.rope)
        self.winch_form.set_instance(L.winch)
        self.env_form.set_instance(L.env)
        self.launch_form.set_instance(L)

    def settings_dict(self) -> dict:
        L = self.current_launch()

        def plain(m):
            return {
                f.name: np.asarray(getattr(m, f.name)).tolist()
                for f in dataclasses.fields(m)
                if not dataclasses.is_dataclass(getattr(m, f.name))
            }

        return {
            "units": "SI",
            "glider": plain(L.glider),
            "pilot": plain(L.pilot),
            "rope": plain(L.rope),
            "winch": plain(L.winch),
            "env": plain(L.env),
            "launch": plain(L),
            "numerics": dataclasses.asdict(self.num_form.get()),
        }

    def apply_settings_dict(self, d: dict):
        classes = {"glider": Glider, "pilot": Pilot, "rope": Rope, "winch": Winch}
        parts = {
            k: cls(
                **{
                    f: np.asarray(v, float) if isinstance(v, list) else v
                    for f, v in d[k].items()
                }
            )
            for k, cls in classes.items()
        }
        L = Launch(env=Env(**d["env"]), **parts, **d["launch"])
        self.set_launch(L)
        self.num_form.set(Numerics(**d["numerics"]))

    def _save_settings(self):
        try:
            d = self.settings_dict()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Save settings", str(e))
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save settings", "launch.json", "JSON (*.json)"
        )
        if fn:
            Path(fn).write_text(json.dumps(d, indent=2))
            self.statusBar().showMessage(f"settings saved to {fn}")

    def _load_settings(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load settings", "", "JSON (*.json)"
        )
        if not fn:
            return
        try:
            self.apply_settings_dict(json.loads(Path(fn).read_text()))
            self.statusBar().showMessage(f"settings loaded from {fn}")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Load settings", str(e))

    # ---- running ----------------------------------------------------------------

    def _run_name(self) -> str:
        self._run_counter += 1
        label = self.name_edit.text().strip()
        base = label or (
            f"{self.glider_combo.currentText()} · {self.rope_combo.currentText()} · "
            f"{self.winch_combo.currentText()}"
        )
        return f"#{self._run_counter} {base}"

    def _start_run(self):
        if self._thread is not None:
            return
        try:
            launch = self.current_launch()
            num = self.num_form.get()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid settings", str(e))
            return
        name = self._run_name()
        self.run_btn.setEnabled(False)
        self.busy.show()
        msg = "simulating" + (" + sensitivities" if num.sensitivity else "")
        self.statusBar().showMessage(f"{name}: {msg}…")
        self._thread = QtCore.QThread(self)
        self._worker = Worker(name, launch, num)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._run_done)
        self._worker.failed.connect(self._run_failed)
        self._thread.start()

    def _finish_thread(self):
        assert self._thread is not None
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        self.run_btn.setEnabled(True)
        self.busy.hide()

    def _run_done(self, run: RunResult):
        self._finish_thread()
        self.add_run(run)
        s = run.summary
        self.statusBar().showMessage(
            f"{run.name}: {s['event']} at {s['release_height']:.0f} m after "
            f"{s['release_time']:.1f} s ({run.elapsed:.1f} s wall time). {run.note}"
        )

    def _run_failed(self, tb: str):
        self._finish_thread()
        self.statusBar().showMessage("simulation failed")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Simulation failed")
        box.setText(tb.strip().splitlines()[-1])
        box.setDetailedText(tb)
        box.exec()

    # ---- results ----------------------------------------------------------------

    def add_run(self, run: RunResult):
        self.runs.append(run)
        t = self.runs_table
        t.blockSignals(True)
        i = t.rowCount()
        t.insertRow(i)
        chk = QtWidgets.QTableWidgetItem()
        chk.setFlags(chk.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        chk.setCheckState(QtCore.Qt.CheckState.Checked)
        t.setItem(i, 0, chk)
        t.setItem(i, 1, QtWidgets.QTableWidgetItem(run.name))
        t.setItem(i, 2, num_item(run.n_segments, "d"))
        for j, (key, _, spec, scale) in enumerate(COLUMNS):
            v = run.summary[key]
            item = (
                QtWidgets.QTableWidgetItem(v)
                if isinstance(v, str)
                else num_item(v * scale, spec or "g")
            )
            t.setItem(i, 3 + j, item)
        t.setItem(i, 3 + len(COLUMNS), num_item(run.elapsed, ".1f"))
        t.blockSignals(False)
        t.resizeColumnsToContents()
        for combo in (self.detail_combo, self.rope_combo_run, self.sens_combo):
            combo.addItem(run.name)
            combo.setCurrentIndex(combo.count() - 1)
        self._refresh_compare()

    def _run_at(self, i: int) -> RunResult | None:
        return self.runs[i] if 0 <= i < len(self.runs) else None

    def _selected_row(self) -> int | None:
        rows = {ix.row() for ix in self.runs_table.selectedIndexes()}
        return min(rows) if rows else None

    def _compare_changed(self, item):
        if item.column() == 0:
            self._refresh_compare()

    def _refresh_compare(self):
        chosen = {
            r.name: r.series
            for i, r in enumerate(self.runs)
            if self.runs_table.item(i, 0).checkState() == QtCore.Qt.CheckState.Checked
        }
        self.compare_pane.set_figure(
            plots.compare(chosen, "Comparison of checked runs") if chosen else None
        )
        self.compare_info.setText(f"{len(chosen)} of {len(self.runs)} runs compared")
        self.compare_pane.placeholder.setText(
            "No runs selected: tick runs in the Runs tab or press 'Compare all runs'."
            if self.runs
            else "Run a simulation to see results here."
        )

    def _set_all_compared(self, on: bool):
        state = QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked
        t = self.runs_table
        t.blockSignals(True)
        for i in range(t.rowCount()):
            t.item(i, 0).setCheckState(state)
        t.blockSignals(False)
        self._refresh_compare()

    def _show_detail(self, i: int):
        run = self._run_at(i)
        self.detail_pane.set_figure(
            plots.launch_detail(run.name, run.series) if run else None
        )

    def _load_run_settings(self):
        i = self._selected_row()
        if i is not None:
            self.set_launch(self.runs[i].launch)
            self.num_form.set(self.runs[i].numerics)
            self.statusBar().showMessage(f"settings of {self.runs[i].name} loaded")

    def _export_series(self):
        i = self._selected_row()
        if i is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export time series", "launch.csv", "CSV (*.csv)"
        )
        if not fn:
            return
        s = self.runs[i].series
        keys = [k for k, v in s.items() if np.ndim(v) == 1]
        np.savetxt(
            fn,
            np.column_stack([s[k] for k in keys]),
            delimiter=",",
            header=",".join(keys),
            comments="",
        )
        self.statusBar().showMessage(f"time series (SI units) written to {fn}")

    def _remove_selected(self):
        i = self._selected_row()
        if i is None:
            return
        del self.runs[i]
        self.runs_table.removeRow(i)
        for combo in (self.detail_combo, self.rope_combo_run, self.sens_combo):
            combo.removeItem(i)
        self._refresh_compare()

    def _clear_runs(self):
        self.runs.clear()
        self.runs_table.setRowCount(0)
        for combo in (self.detail_combo, self.rope_combo_run, self.sens_combo):
            combo.clear()
        self._refresh_compare()


def main() -> None:
    # '.' as decimal separator everywhere, matching the input fields
    QtCore.QLocale.setDefault(QtCore.QLocale.c())
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
