import numpy as np

from winch_sim.report import write_series_csv


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
