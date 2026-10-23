import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtCore = pytest.importorskip("PySide6.QtCore")

from winch_sim.gui.app import (
    MainWindow,
    Numerics,
    RunResult,
    default_launch,
    release_info,
    run_launch,
)
from winch_sim.winch import engine_winch


@pytest.fixture(scope="module")
def window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    yield w
    w.close()
    app.processEvents()


def test_forms_reproduce_the_default_launch(window):
    ref, got = default_launch(), window.current_launch()
    for part in ("glider", "rope", "winch", "pilot", "env"):
        a, b = getattr(ref, part), getattr(got, part)
        for f in a.__dataclass_fields__:
            np.testing.assert_allclose(
                np.asarray(getattr(b, f), float),
                np.asarray(getattr(a, f), float),
                rtol=1e-5,
                err_msg=f"{part}.{f}",
            )
    assert float(got.rope_length) == 1200.0


def test_preset_changes_glider_and_rescales_winch(window):
    try:
        window.glider_combo.setCurrentText("Ka 8")
        L = window.current_launch()
        assert float(L.glider.mass) == 290.0
        # the engine winch (default) keeps its curve and throttles to the weight
        ref = engine_winch(L.glider, pull_per_weight=1.1)
        assert float(L.winch.F_max) == pytest.approx(float(ref.F_max))
        assert float(L.winch.throttle) == pytest.approx(float(ref.throttle))
        window.winch_combo.setCurrentText("tension")
        L = window.current_launch()
        assert float(L.winch.F_max) == pytest.approx(1.1 * 290.0 * 9.81)
    finally:
        window.winch_combo.setCurrentText("engine")
        window.glider_combo.setCurrentText("ASK 21")


def test_run_and_settings_roundtrip(window):
    run = run_launch("test", window.current_launch(), Numerics(sensitivity=False))
    assert run.summary["event"] == "back_release"
    window.add_run(run)
    assert window.runs_table.rowCount() == 1
    assert window.detail_combo.count() == 1
    d = window.settings_dict()
    window.glider_form.set_si({"mass": 999.0})
    window.apply_settings_dict(d)
    assert float(window.current_launch().glider.mass) == 470.0


def test_invalid_input_is_reported(window):
    window.rope_form.editors["mu"][0].setText("abc")
    with pytest.raises(ValueError, match="mu"):
        window.current_launch()
    window.rope_form.editors["mu"][0].set_value(0.016)


def test_clear_and_restore_comparison(window):
    if not window.runs:
        window.add_run(
            run_launch("c", window.current_launch(), Numerics(sensitivity=False))
        )
    window._set_all_compared(False)
    assert window.compare_pane.canvas is None
    assert window.compare_info.text().startswith("0 of")
    window._set_all_compared(True)
    assert window.compare_pane.canvas is not None
    assert len(window.runs) >= 1  # clearing the comparison keeps the runs


def test_release_indicator(window):
    run = run_launch("r", window.current_launch(), Numerics(sensitivity=False))
    i, head, detail, _, _ = release_info(run)
    assert head == "BACK-RELEASE" and "fuselage axis" in detail
    assert run.series["released"][i] == 0 and run.series["released"][i + 1] == 1
    view = window.rope_view
    view.set_run(run)
    view.slider.setValue(0)
    assert not view.banner.get_visible()
    view.slider.setValue(view.slider.maximum())
    assert view.banner.get_visible() and view.release_pt.get_visible()
    # a weak-link failure gets its own label
    broken = RunResult(
        **{**run.__dict__, "summary": {**run.summary, "event": "weak_link"}}
    )
    assert release_info(broken)[1] == "WEAK LINK BREAK"


def test_stop_kills_the_running_simulation(window):
    import time

    app = QtWidgets.QApplication.instance()
    window._start_run()
    assert window._job is not None and window.stop_btn.isEnabled()
    old = window._sim._proc
    t0 = time.time()
    while time.time() - t0 < 1.0:
        app.processEvents()
        time.sleep(0.02)
    window._stop_run()
    assert window._job is None
    assert not old.is_alive()  # the job died with its worker
    assert window._sim._proc.is_alive()  # a fresh worker is ready
    assert window.run_btn.isEnabled() and not window.stop_btn.isEnabled()
    assert "stopped" in window.statusBar().currentMessage()


def test_import_measured_tension(window, tmp_path):
    from pathlib import Path

    from winch_sim.gui.app import SHIFT_COL

    data = Path(__file__).parent.parent / "data" / "tensiometer" / "2026-09-27"
    flat = tmp_path / "flat.csv"
    flat.write_text("t,force_n\n" + "".join(f"{x},80\n" for x in range(100)))
    defaults = window.settings_dict()
    n, errors = window.import_measurements([*sorted(data.glob("*.csv")), flat])
    assert n == 5 and len(errors) == 1 and "no launch" in errors[0]
    assert window.meas_table.rowCount() == 5
    assert window.meas_pane.canvas is not None
    if not window.runs:
        window.add_run(
            run_launch("m", window.current_launch(), Numerics(sensitivity=False))
        )
    window._set_all_compared(True)
    ax_T = window.compare_pane.canvas.figure.axes[3]
    labels = [ln.get_label() for ln in ax_T.get_lines()]
    assert sum(lb.startswith("measured:") for lb in labels) == 5
    assert "5 measured" in window.compare_info.text()
    # the shift column is editable; invalid text keeps the old value
    item = window.meas_table.item(0, SHIFT_COL)
    item.setText("2.5")
    assert window.measurements[0].shift == 2.5
    item.setText("abc")
    assert window.measurements[0].shift == 2.5 and item.text() == "2.5"
    # unticked measurements are not drawn
    window.meas_table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    ax_T = window.compare_pane.canvas.figure.axes[3]
    labels = [ln.get_label() for ln in ax_T.get_lines()]
    assert sum(lb.startswith("measured:") for lb in labels) == 4
    # importing the same file again gives unique names
    window.import_measurements([min(data.glob("*.csv"))])
    assert len({m.name for m in window.measurements}) == 6
    # settings from the recorded launch conditions
    ask13 = next(m for m in window.measurements if m.glider == "ASK 13")
    window.apply_conditions(ask13)
    L = window.current_launch()
    assert window.glider_combo.currentText() == "ASK 13"
    assert window.rope_combo.currentText() == "dyneema"
    assert window.winch_combo.currentText() == "engine"
    assert float(L.rope_length) == 1050.0 and float(L.env.wind_ref) == 6.0
    assert float(L.glider.mass) == 460.0 and float(L.winch.M_eff) == 250.0
    window._clear_measured()
    assert window.meas_table.rowCount() == 0 and window.meas_pane.canvas is None
    window.apply_settings_dict(defaults)
    window.glider_combo.setCurrentText("ASK 21")
    window.rope_combo.setCurrentText("dyneema")
    window.winch_combo.setCurrentText("tension")
    window.set_launch(default_launch())


def test_export_series_with_rope_positions(window, tmp_path, monkeypatch):
    if not window.runs:
        window.add_run(
            run_launch("e", window.current_launch(), Numerics(sensitivity=False))
        )
    window.runs_table.selectRow(0)
    fn = tmp_path / "launch.csv"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(fn), "")
    )
    series = window.runs[0].series
    for rope in (False, True):
        window.export_rope.setChecked(rope)
        window._export_series()
        head = fn.read_text().splitlines()[0].split(",")
        assert ("rope_x_0" in head) == rope
    assert f"rope_z_{series['rope_z'].shape[1] - 1}" in head
    data = np.loadtxt(fn, delimiter=",", skiprows=1)
    np.testing.assert_allclose(data[:, head.index("rope_x_0")], series["rope_x"][:, 0])
