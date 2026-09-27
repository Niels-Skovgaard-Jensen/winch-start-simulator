from pathlib import Path

import numpy as np
import pytest

from winch_sim import plots
from winch_sim.measured import (
    SATURATED,
    find_launches,
    line_up,
    load_launches,
    read_csv,
    sim_pull_start,
    sim_release_time,
    traces,
)

DATA = Path(__file__).parent.parent / "data" / "tensiometer" / "2026-09-27"


def session(seed: int = 0):
    """A synthetic session at 12 Hz: unloaded level 80 N, a rope pull-out
    (~500 N for 3 min), a handling spike and one launch (pull at t = 1000 s, 4 kN
    at the top, release at 1040 s, then ~800 N while the rope is wound in)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 1800.0, 1 / 12)
    f = 80.0 + rng.normal(0.0, 1.0, t.size)
    f += np.where((t > 300) & (t < 480), 500.0, 0.0)  # pull-out
    f += np.where((t > 700) & (t < 700.5), 300.0, 0.0)  # spike
    climb = (t >= 1000) & (t < 1040)
    f[climb] += 1500.0 + 2500.0 * (t[climb] - 1000) / 40
    f += np.where((t >= 1041) & (t < 1060), 800.0, 0.0)  # parachute
    return t, f


def test_finds_only_the_launch():
    t, f = session()
    (m,) = find_launches(t, f, "session")
    assert m.name == "session"
    assert m.start_recorded
    assert m.t_pull == pytest.approx(1000.0, abs=0.1)
    assert m.t_release == pytest.approx(1040.0, abs=0.1)
    assert m.duration == pytest.approx(40.0, abs=0.2)
    assert m.peak == pytest.approx(4000.0, rel=0.01)
    assert m.offset == pytest.approx(80.0, abs=2.0)
    assert m.clock == "00:16:40"
    assert m.t[0] <= m.t_pull - 29 and m.t[-1] >= 1060 + 29  # context on both sides


def test_missing_start_is_flagged():
    t, f = session()
    keep = (t < 900) | (t > 1015)  # radio dropout during the ground run
    (m,) = find_launches(t[keep], f[keep])
    assert not m.start_recorded
    assert m.t_pull == pytest.approx(1015.0, abs=0.1)
    assert m.t_release == pytest.approx(1040.0, abs=0.1)


def test_several_launches_get_their_clock_in_the_name():
    t, f = session()
    t2 = np.concatenate([t, t + 1800.0])
    ms = find_launches(t2, np.concatenate([f, f]), "LOG00011")
    assert [m.name for m in ms] == ["LOG00011 00:16:40", "LOG00011 00:46:40"]


def test_tensiometer_csv(tmp_path):
    t, f = session()
    stamps = np.datetime64("2000-01-01T01:00:00", "ms") + (t * 1000).astype(
        "timedelta64[ms]"
    )
    raw = np.round(f * 219).astype(int)
    raw[5000] = SATURATED  # read glitch
    f[5000] = 1e5
    fn = tmp_path / "LOG00042.CSV"
    with open(fn, "w") as fh:
        fh.write("time_utc,raw_counts,force_n\n")
        fh.writelines(f"{s},{r},{x:.2f}\n" for s, r, x in zip(stamps, raw, f))
    log = read_csv(fn)
    assert log.is_clock and log.t.size == t.size - 1
    assert log.t[0] == pytest.approx(3600.0)
    (m,) = load_launches(fn)
    assert m.clock == "01:16:40" and m.source == str(fn)


def test_plain_csv_in_kn_and_errors(tmp_path):
    t, f = session()
    fn = tmp_path / "plain.csv"
    np.savetxt(fn, np.column_stack([t, f / 1e3]), delimiter=",", header="t,force_kN")
    fn.write_text(fn.read_text().removeprefix("# "))
    (m,) = load_launches(fn)
    assert m.clock == "1000.0 s" and m.peak == pytest.approx(4000.0, rel=0.01)

    flat = tmp_path / "flat.csv"
    flat.write_text("t,force_n\n" + "".join(f"{x},80\n" for x in range(100)))
    with pytest.raises(ValueError, match="no launch"):
        load_launches(flat)
    empty = tmp_path / "empty.csv"
    empty.write_text("time_utc,raw_counts,force_n\n")
    with pytest.raises(ValueError, match="no samples"):
        load_launches(empty)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="no column"):
        load_launches(bad)


def test_example_launches():
    files = sorted(DATA.glob("*.csv"))
    assert len(files) == 5
    ms = [m for fn in files for m in load_launches(fn)]
    assert len(ms) == 5
    assert [m.start_recorded for m in ms] == [False, True, True, False, True]
    for m in ms:
        assert 3.5e3 < m.peak < 6e3
        assert 70.0 < m.offset < 90.0  # from the full log (# unloaded_level_n)
        assert m.t_pull < m.t_release < m.t[-1]
        # the drop: tension a second after release is a fraction of the peak
        i = np.searchsorted(m.t, m.t_release + 1.0)
        assert m.tension()[i] < 0.25 * m.peak
    complete = [m.duration for m in ms if m.start_recorded]
    assert all(40.0 < d < 55.0 for d in complete)


def test_line_up_with_a_simulated_launch():
    (m,) = load_launches(DATA / "LOG00011_014330.csv")
    t = np.linspace(0.0, 50.0, 501)
    T = np.where(t < 2.0, 0.0, np.where(t < 40.0, 5e3, 100.0))
    s = {"t": t, "T_hook": T, "released": (t > 43.0).astype(float)}
    # same rules as for measurements: tension drop, first PULL_N
    assert sim_release_time(s) == pytest.approx(39.9)
    assert sim_pull_start(s) == pytest.approx(2.0)
    tr = line_up(m, "release", s)
    assert tr[np.searchsorted(m.t, m.t_release)] == pytest.approx(39.9)
    tp = line_up(m, "pull", s)
    assert tp[np.searchsorted(m.t, m.t_pull)] == pytest.approx(2.0)
    m.shift = 1.5
    assert line_up(m, "pull")[np.searchsorted(m.t, m.t_pull)] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        line_up(m, "sideways")
    (name, (_, TT)), *_ = traces([m], "release", s, zero=False).items()
    assert name == m.name and TT[0] == pytest.approx(m.force[0])


def test_measured_figures():
    ms = [m for fn in sorted(DATA.glob("*.csv")) for m in load_launches(fn)]
    fig = plots.measured_launches(ms, "pull")
    assert len(fig.axes[0].get_legend().get_texts()) == 5
    # overlay without any simulated run
    fig = plots.compare({}, "measured only", traces(ms))
    labels = [ln.get_label() for ln in fig.axes[3].get_lines()]
    assert sum(lb.startswith("measured:") for lb in labels) == 5
