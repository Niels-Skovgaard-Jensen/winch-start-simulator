import numpy as np
import pytest

from winch_sim.report import resample_series, write_series_csv


def _series(n_t=5, n_nodes=3):
    t = np.linspace(0.0, 1.0, n_t)
    return {
        "t": t,
        "height": 2 * t,
        "rope_x": np.arange(n_t * n_nodes, dtype=float).reshape(n_t, n_nodes),
        "rope_z": -np.arange(n_t * n_nodes, dtype=float).reshape(n_t, n_nodes),
    }


def test_series_csv_without_rope(tmp_path):
    fn = tmp_path / "s.csv"
    write_series_csv(fn, _series())
    assert fn.read_text().splitlines()[0] == "t,height"
    assert np.loadtxt(fn, delimiter=",", skiprows=1).shape == (5, 2)


def test_series_csv_with_rope_positions(tmp_path):
    s = _series()
    fn = tmp_path / "s.csv"
    write_series_csv(fn, s, rope=True)
    head = fn.read_text().splitlines()[0].split(",")
    assert head == ["t", "height"] + [f"rope_{c}_{j}" for c in "xz" for j in range(3)]
    data = np.loadtxt(fn, delimiter=",", skiprows=1)
    np.testing.assert_array_equal(data[:, 2:5], s["rope_x"])
    np.testing.assert_array_equal(data[:, 5:8], s["rope_z"])


def _launch_and_flight():
    # launch 0..2 s, free flight 2..3 s; the hook tension drops to 0 at release
    t = np.r_[np.linspace(0.0, 2.0, 5), np.linspace(2.0, 3.0, 3)]
    rel = (np.arange(t.size) >= 5).astype(float)
    return {
        "t": t,
        "released": rel,
        "x": 10 * t,
        "T_hook": np.where(rel > 0, 0.0, 1000.0),
        "rope_x": np.stack([0 * t, 5 * t], -1),
        "rope_z": np.stack([0 * t, -t], -1),
    }


def test_resample_with_time_step():
    r = resample_series(_launch_and_flight(), dt=0.3)
    np.testing.assert_allclose(r["t"], 0.3 * np.arange(11))  # 3.0 is not a step
    np.testing.assert_allclose(r["x"], 10 * r["t"])
    np.testing.assert_allclose(r["rope_x"][:, 1], 5 * r["t"])


def test_resample_with_number_of_points():
    r = resample_series(_launch_and_flight(), n=7)
    np.testing.assert_allclose(r["t"], np.linspace(0.0, 3.0, 7))
    assert r["rope_x"].shape == (7, 2)


def test_resample_does_not_blend_across_release():
    r = resample_series(_launch_and_flight(), dt=0.5)
    launch = r["t"] <= 2.0
    np.testing.assert_array_equal(r["released"], np.where(launch, 0.0, 1.0))
    np.testing.assert_array_equal(r["T_hook"], np.where(launch, 1000.0, 0.0))


def test_resample_rejects_bad_settings():
    s = _launch_and_flight()
    for kw in ({}, {"dt": 0.1, "n": 10}, {"dt": 0.0}, {"n": 1}):
        with pytest.raises(ValueError):
            resample_series(s, **kw)


def test_series_csv_resampled(tmp_path):
    fn = tmp_path / "s.csv"
    write_series_csv(fn, _launch_and_flight(), rope=True, n=4)
    head = fn.read_text().splitlines()[0].split(",")
    data = np.loadtxt(fn, delimiter=",", skiprows=1)
    assert data.shape == (4, len(head))
    np.testing.assert_allclose(data[:, head.index("t")], [0.0, 1.0, 2.0, 3.0])
