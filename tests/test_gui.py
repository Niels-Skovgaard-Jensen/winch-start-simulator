import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from winch_sim.gui.app import (
    MainWindow,
    Numerics,
    default_launch,
    run_launch,
)


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
    window.glider_combo.setCurrentText("Ka 8")
    L = window.current_launch()
    assert float(L.glider.mass) == 290.0
    assert float(L.winch.F_max) == pytest.approx(1.1 * 290.0 * 9.81)
    window.glider_combo.setCurrentText("ASK 21")


def test_run_and_settings_roundtrip(window):
    run = run_launch("test", window.current_launch(), Numerics(sensitivity=False))
    assert run.summary["event"] == "release"
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
