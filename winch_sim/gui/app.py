"""Qt desktop app: edit every model setting, run launches, visualise and compare them.

Start with `uv run winch-gui` (or `uv run python -m winch_sim.gui`).
"""

import dataclasses
import json
import sys
import time
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
from ..measured import ALIGN_MODES, Measurement, load_launches, traces
from ..params import Env, Glider, Launch, Pilot, Rope, Winch
from ..report import COLUMNS, write_sensitivity_csv, write_series_csv
from ..sensitivity import sensitivity_unit
from ..winch import engine_winch, tension_winch
from .forms import FloatEdit, ModuleForm, scrolled
from .worker import (  # noqa: F401  (run_launch re-exported for scripts/tests)
    WARMUP,
    Numerics,
    RunResult,
    SimProcess,
    default_launch,
    run_launch,
)

WINCH_PRESETS = ("engine","tension") 
MEASURED_DIR = Path("data/tensiometer")  # example logs, see data/tensiometer/README.md
MEASURED_HEADS = (
    "show",
    "measurement",
    "glider",
    "clock",
    "pull → release s",
    "peak kN",
    "offset N",
    "shift s",
    "file",
)
SHIFT_COL = MEASURED_HEADS.index("shift s")


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
        "back_release": f"rope slack (hook tension {s['T_hook'][i] / 1e3:.2f} kN), "
        f"cable {np.degrees(s['cable_body'][i]):.0f}° below the fuselage axis "
        f"(hook trips at {np.degrees(float(L.pilot.back_release_angle)):.0f}°)",
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
        self.measurements: list[Measurement] = []
        self._run_counter = 0
        # Simulations run in a worker process, so a running one can be stopped.
        # It starts now, so JAX is imported while the user edits settings.
        self._sim = SimProcess()
        self._sim.start()
        self._job: tuple[int, str, float] | None = None  # (id, name, start time)
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll)
        self._warming = True  # the worker compiles the model on start-up
        self._poll_timer.start()

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
            "Compiling the model in the background… you can edit settings meanwhile."
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
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setToolTip("stop the running simulation (Esc)")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_run)
        self.busy = QtWidgets.QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumWidth(120)
        self.busy.hide()
        run_box.addWidget(self.name_edit, 1)
        run_box.addWidget(self.busy)
        run_box.addWidget(self.stop_btn)
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
        self.export_rope = QtWidgets.QCheckBox("export rope positions")
        self.export_rope.setToolTip(
            "add the x, z of every rope node to the exported time series "
            "(two columns per node, often doubles the file)"
        )
        btns.addWidget(self.export_rope)
        btns.addStretch(1)
        rl.addLayout(btns)
        self.results.addTab(runs, "Runs")
        self.measured_tab = self._measured_tab()
        self.results.addTab(self.measured_tab, "Measured")

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

    def _measured_tab(self) -> QtWidgets.QWidget:
        top = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        ctl = QtWidgets.QHBoxLayout()
        imp = QtWidgets.QPushButton("Import tensiometer CSV…")
        imp.clicked.connect(self._import_measured)
        self.align_combo = QtWidgets.QComboBox()
        for key, text in ALIGN_MODES.items():
            self.align_combo.addItem(text, key)
        self.align_combo.setToolTip(
            "where measured traces go on the time axis: their release (or start of "
            "the pull) at the same moment as in the run they are drawn with"
        )
        self.align_combo.currentIndexChanged.connect(self._measured_changed)
        self.zero_check = QtWidgets.QCheckBox("subtract the sensor offset")
        self.zero_check.setToolTip(
            "subtract the unloaded level (the logger reads ~70-90 N without load)"
        )
        self.zero_check.setChecked(True)
        self.zero_check.toggled.connect(self._measured_changed)
        ctl.addWidget(imp)
        ctl.addWidget(QtWidgets.QLabel("line up"))
        ctl.addWidget(self.align_combo)
        ctl.addWidget(self.zero_check)
        ctl.addStretch(1)
        tl.addLayout(ctl)
        t = self.meas_table = QtWidgets.QTableWidget(0, len(MEASURED_HEADS))
        t.setHorizontalHeaderLabels(list(MEASURED_HEADS))
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        t.itemChanged.connect(self._measured_item_changed)
        tl.addWidget(t, 1)
        btns = QtWidgets.QHBoxLayout()
        for text, slot in [
            ("Use launch conditions", self._use_conditions),
            ("Remove selected", self._remove_measured),
            ("Clear all", self._clear_measured),
        ]:
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            btns.addWidget(b)
        note = QtWidgets.QLabel(
            "Rope tension logged by the tensiometer at the glider end of the rope. "
            "Ticked launches are drawn over the simulated tension at the hook in "
            "the Comparison and Detail tabs. A log with several launches gives one "
            "row each. Double-click 'shift s' to move a trace in time. 'Use launch "
            "conditions' sets glider, rope, winch, rope length and wind as recorded "
            "with the selected launch."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        btns.addWidget(note, 1)
        tl.addLayout(btns)

        self.meas_pane = FigurePane()
        self.meas_pane.placeholder.setText(
            "Import a tensiometer log (File → Import measured tension) to see it here."
        )
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.meas_pane)
        split.setStretchFactor(1, 1)
        split.setSizes([230, 650])
        return split

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
            ("Import measured tension (CSV)…", self._import_measured, "Ctrl+I"),
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
        act = r.addAction("Stop simulation")
        act.setShortcut("Esc")
        act.triggered.connect(self._stop_run)

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
        if self._job is not None:
            return
        try:
            launch = self.current_launch()
            num = self.num_form.get()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid settings", str(e))
            return
        name = self._run_name()
        self._job = (self._run_counter, name, time.perf_counter())
        self._sim.submit(self._run_counter, name, launch, num)
        self._set_running(True)
        self._poll()

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.busy.setVisible(running)
        if running or self._warming:
            self._poll_timer.start()
        else:
            self._poll_timer.stop()

    def _poll(self):
        msg = self._sim.poll()
        if isinstance(msg, tuple) and msg[0] == WARMUP:
            self._warming = False
            _, status, payload = msg
            if self._job is None:
                self._set_running(False)
                self.statusBar().showMessage(
                    f"Ready: model compiled in {payload:.0f} s."
                    if status == "ok"
                    else "Background compilation failed; runs will compile on demand."
                )
            return
        if self._job is None:
            if msg == "dead":
                self._warming = False
                self._set_running(False)
            return
        job_id, name, t0 = self._job
        if msg is None:
            what = "sensitivities" if self.num_form.sens.isChecked() else ""
            self.statusBar().showMessage(
                f"{name}: simulating{' + ' + what if what else ''}… "
                f"{time.perf_counter() - t0:.0f} s  (Esc to stop)"
            )
            return
        if msg == "dead":
            self._job = None
            self._sim.start()
            self._warming = True
            self._set_running(False)
            self._run_failed("The simulation worker process died unexpectedly.")
            return
        done_id, status, payload = msg
        if done_id != job_id:  # a result of an earlier, stopped job
            return
        self._job = None
        self._set_running(False)
        if status == "ok":
            self._run_done(payload)
        else:
            self._run_failed(payload)

    def _stop_run(self):
        """Kill the worker process (and the running job), then start a fresh one."""
        if self._job is None:
            return
        _, name, t0 = self._job
        self._job = None
        self._sim.stop()
        self._sim.start()
        self._warming = True  # the new worker compiles the model again
        self._set_running(False)
        self.statusBar().showMessage(
            f"{name}: stopped after {time.perf_counter() - t0:.0f} s."
        )

    def _run_done(self, run: RunResult):
        self.add_run(run)
        s = run.summary
        self.statusBar().showMessage(
            f"{run.name}: {s['event']} at {s['release_height']:.0f} m after "
            f"{s['release_time']:.1f} s ({run.elapsed:.1f} s wall time). {run.note}"
        )

    def _run_failed(self, tb: str):
        self.statusBar().showMessage("simulation failed")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Simulation failed")
        box.setText(tb.strip().splitlines()[-1])
        box.setDetailedText(tb)
        box.exec()

    def closeEvent(self, event):
        self._poll_timer.stop()
        self._sim.shutdown()
        super().closeEvent(event)

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
        ref = next(iter(chosen.values()), None)
        measured = self._measured_traces(ref)
        self.compare_pane.set_figure(
            plots.compare(chosen, "Comparison of checked runs", measured)
            if chosen or measured
            else None
        )
        info = f"{len(chosen)} of {len(self.runs)} runs compared"
        if measured:
            info += (
                f"; {len(measured)} measured, lined up {self.align_combo.currentText()}"
            )
            info += f" with {next(iter(chosen))}" if chosen else " at t = 0"
        self.compare_info.setText(info)
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
            plots.launch_detail(
                run.name, run.series, measured=self._measured_traces(run.series)
            )
            if run
            else None
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
        rope = self.export_rope.isChecked()
        write_series_csv(fn, self.runs[i].series, rope=rope)
        what = "time series with rope positions" if rope else "time series"
        self.statusBar().showMessage(f"{what} (SI units) written to {fn}")

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

    # ---- measured tension ----------------------------------------------------------

    def _import_measured(self):
        start = str(MEASURED_DIR) if MEASURED_DIR.is_dir() else ""
        fns, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Import measured tension", start, "CSV (*.csv *.CSV);;All files (*)"
        )
        if not fns:
            return
        n, errors = self.import_measurements(fns)
        self.statusBar().showMessage(
            f"{n} measured launch{'es' if n != 1 else ''} imported from "
            f"{len(fns)} file{'s' if len(fns) != 1 else ''}"
        )
        if n:
            self.results.setCurrentWidget(self.measured_tab)
        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Import measured tension", "\n".join(errors)
            )

    def import_measurements(self, paths) -> tuple[int, list[str]]:
        """Add every launch found in the files; (number added, error messages)."""
        n, errors = 0, []
        for p in paths:
            try:
                found = load_launches(p)
            except (OSError, ValueError) as e:
                errors.append(str(e))
                continue
            for m in found:
                self.add_measurement(m)
                n += 1
        return n, errors

    def add_measurement(self, m: Measurement):
        names = {x.name for x in self.measurements}
        base, k = m.name, 2
        while m.name in names:
            m.name, k = f"{base} ({k})", k + 1
        self.measurements.append(m)
        t = self.meas_table
        t.blockSignals(True)
        i = t.rowCount()
        t.insertRow(i)
        chk = QtWidgets.QTableWidgetItem()
        chk.setFlags(chk.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        chk.setCheckState(QtCore.Qt.CheckState.Checked)
        duration = (
            f"{m.duration:.1f}"
            if m.start_recorded
            else f"> {m.duration:.0f} (start not recorded)"
        )
        items = [
            chk,
            QtWidgets.QTableWidgetItem(m.name),
            QtWidgets.QTableWidgetItem(m.glider),
            QtWidgets.QTableWidgetItem(m.clock),
            QtWidgets.QTableWidgetItem(duration),
            num_item(m.peak / 1e3, ".2f"),
            num_item(m.offset, ".0f"),
            QtWidgets.QTableWidgetItem(f"{m.shift:g}"),
            QtWidgets.QTableWidgetItem(Path(m.source).name),
        ]
        items[3].setToolTip("logger clock at the start of the pull (time since boot)")
        items[-1].setToolTip(m.source)
        for j, it in enumerate(items):
            if j != SHIFT_COL:
                it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            t.setItem(i, j, it)
        t.blockSignals(False)
        t.resizeColumnsToContents()
        self._measured_changed()

    def _checked_measurements(self) -> list[Measurement]:
        t = self.meas_table
        return [
            m
            for i, m in enumerate(self.measurements)
            if t.item(i, 0).checkState() == QtCore.Qt.CheckState.Checked
        ]

    def _measured_traces(self, ref) -> dict:
        """Ticked measurements lined up with the simulated series `ref` (or t = 0)."""
        return traces(
            self._checked_measurements(),
            self.align_combo.currentData(),
            ref,
            self.zero_check.isChecked(),
        )

    def _measured_item_changed(self, item):
        if item.column() == SHIFT_COL:
            m = self.measurements[item.row()]
            try:
                m.shift = float(item.text())
            except ValueError:
                pass  # keep the old shift
            self.meas_table.blockSignals(True)
            item.setText(f"{m.shift:g}")
            self.meas_table.blockSignals(False)
        if item.column() in (0, SHIFT_COL):
            self._measured_changed()

    def _measured_changed(self, *_):
        chosen = self._checked_measurements()
        self.meas_pane.set_figure(
            plots.measured_launches(
                chosen, self.align_combo.currentData(), self.zero_check.isChecked()
            )
            if chosen
            else None
        )
        self.meas_pane.placeholder.setText(
            "Tick a measured launch to show it."
            if self.measurements
            else "Import a tensiometer log (File → Import measured tension) to see "
            "it here."
        )
        self._refresh_compare()
        self._show_detail(self.detail_combo.currentIndex())

    def _use_conditions(self):
        """Settings from the launch conditions recorded with the selected launch."""
        rows = sorted({ix.row() for ix in self.meas_table.selectedIndexes()})
        if not rows:
            self.statusBar().showMessage("select a measured launch first")
            return
        m = self.measurements[rows[0]]
        self.apply_conditions(m)

    def apply_conditions(self, m: Measurement):
        c = m.conditions()
        done = []
        if c.get("glider") in CATALOGUE:  # also re-applies the winch preset
            self.glider_combo.setCurrentText(str(c["glider"]))
            self._apply_glider(str(c["glider"]))
            done.append(str(c["glider"]))
        if c.get("rope") in ROPES:
            self.rope_combo.setCurrentText(str(c["rope"]))
            self._apply_rope(str(c["rope"]))
            done.append(f"{c['rope']} rope")
        if "rope_length_m" in c:
            self.launch_form.set_si({"rope_length": c["rope_length_m"]})
            done.append(f"{c['rope_length_m']:g} m")
        if c.get("winch") in WINCH_PRESETS:
            self.winch_combo.setCurrentText(str(c["winch"]))
            self._apply_winch()
            done.append(f"{c['winch']} winch")
        if "headwind_m_s" in c:
            self.env_form.set_si({"wind_ref": c["headwind_m_s"]})
            done.append(f"headwind {c['headwind_m_s']:g} m/s")
        self.statusBar().showMessage(
            f"settings from {m.name}: {', '.join(done)}"
            if done
            else f"no launch conditions recorded in {Path(m.source).name}"
        )

    def _remove_measured(self):
        rows = {ix.row() for ix in self.meas_table.selectedIndexes()}
        for i in sorted(rows, reverse=True):
            del self.measurements[i]
            self.meas_table.removeRow(i)
        if rows:
            self._measured_changed()

    def _clear_measured(self):
        self.measurements.clear()
        self.meas_table.setRowCount(0)
        self._measured_changed()


def main() -> None:
    # '.' as decimal separator everywhere, matching the input fields
    QtCore.QLocale.setDefault(QtCore.QLocale.c())
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
