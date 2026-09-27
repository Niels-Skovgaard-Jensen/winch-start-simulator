"""Measured rope tension: read tensiometer logs and line them up with simulations.

The tensiometer is a load cell in line with the rope at the glider end, so it
measures what the simulation calls the tension at the hook.  It logs CSV files at
about 12 Hz:

    time_utc,raw_counts,force_n
    2000-01-01T01:44:07.703,954044,4356.72

The boards have no real-time clock: `time_utc` counts from boot (2000-01-01T00:00).
Samples at ADC full scale (raw_counts 0x3FFFFF) are read glitches and are dropped.
One file can hold a whole session; `find_launches` picks out the launches (loaded
stretches peaking more than LAUNCH_PEAK_N above the unloaded level) and cuts a
window around each.  Rope pull-outs (a few hundred N for minutes) and handling
spikes stay below that.

Plain CSV files load too: a time column in seconds (`t`, `time`, `time_s`) and a
force column in N (`force_n`, `force`, `tension`, `T_hook`) or kN (`force_kn`).
Lines starting with `#` are comments.  `# key: value` lines are metadata:
`unloaded_level_n` gives the unloaded level (sensor offset) instead of estimating
it (see scripts/extract_launches.py), and the launch conditions (CONDITIONS, e.g.
`# glider: ASK 13`, `# rope_length_m: 1050`) let the GUI set up a matching
simulation.
"""

import csv
import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

SATURATED = 0x3FFFFF  # 22-bit ADC full scale: a read glitch, not a force
PULL_N = 75.0  # the pull has started once tension is this far above the unloaded level
LAUNCH_PEAK_N = 2000.0  # a loaded stretch is a launch if it peaks this far above it
RELEASE_FRACTION = 0.25  # release: tension drops below this fraction of the peak
MERGE_GAP_S = 60.0  # loaded samples closer than this belong to one launch
GAP_S = 2.0  # samples further apart than this: a radio dropout
BASELINE_WINDOW_S = 600.0  # rolling window for the unloaded level...
BASELINE_BIN_S = 5.0  # ...computed from medians over bins this long
PAD_S = 30.0  # context kept before the pull and after the tension has settled
OFFSET_KEY = "unloaded_level_n"  # metadata: known unloaded level [N]
CONDITIONS = {  # metadata key -> type: the launch conditions, if recorded
    "glider": str,  # catalogue name
    "rope": str,  # rope preset
    "winch": str,  # winch preset
    "rope_length_m": float,
    "headwind_m_s": float,  # at 10 m
}

TIME_COLUMNS = ("time_utc", "time", "timestamp", "t", "time_s")
FORCE_COLUMNS = {  # column name -> N per unit
    "force_n": 1.0,
    "force": 1.0,
    "tension_n": 1.0,
    "tension": 1.0,
    "t_hook": 1.0,
    "force_kn": 1e3,
    "tension_kn": 1e3,
}

ALIGN_MODES = {  # key -> description (GUI, CLI)
    "release": "at release (the drop in tension)",
    "pull": "at the start of the pull",
}


@dataclasses.dataclass(eq=False)
class Measurement:
    """One measured launch: rope tension vs time, with the detected key moments."""

    name: str
    source: str  # file it was read from
    t: np.ndarray  # [s] logger clock (since boot for the tensiometer)
    force: np.ndarray  # [N] as logged, sensor offset included
    offset: float  # [N] unloaded level at the start of the pull (sensor offset)
    t_pull: float  # [s] start of the pull: tension PULL_N above the unloaded level
    t_release: float  # [s] release: the last sample before the final drop
    peak: float  # [N] peak tension above the unloaded level
    start_recorded: bool  # False if the log only starts during the launch
    clock: str  # logger clock at the start of the pull, for display
    shift: float = 0.0  # [s] manual time shift, added when lining up
    meta: dict[str, str] = dataclasses.field(default_factory=dict)  # `# key: value`

    @property
    def duration(self) -> float:
        """Start of the pull to release [s]; a lower bound if the start is missing."""
        return self.t_release - self.t_pull

    def tension(self, zero: bool = True) -> np.ndarray:
        """Tension [N], with the sensor offset subtracted if `zero`."""
        return self.force - self.offset if zero else self.force

    @property
    def glider(self) -> str:
        return self.meta.get("glider", "")

    @property
    def label(self) -> str:
        return f"{self.name} ({self.glider})" if self.glider else self.name

    def conditions(self) -> dict[str, str | float]:
        """The recorded launch conditions (CONDITIONS keys only)."""
        out: dict[str, str | float] = {}
        for key, kind in CONDITIONS.items():
            if key in self.meta:
                out[key] = kind(self.meta[key])
        return out


def parse_times(values: list[str]) -> tuple[np.ndarray, bool]:
    """(seconds, is_clock): ISO timestamps as seconds since midnight of the first
    sample's day (= since boot for the tensiometer), or plain numbers in seconds."""
    try:
        return np.array(values, dtype=float), False
    except ValueError:
        stamps = np.array(values, dtype="datetime64[ms]")
        day = stamps[0].astype("datetime64[D]")
        return (stamps - day) / np.timedelta64(1, "s"), True


class Log(NamedTuple):
    t: np.ndarray  # [s]
    force: np.ndarray  # [N]
    is_clock: bool  # t from timestamps (logger clock) rather than plain seconds
    meta: dict[str, str]  # `# key: value` lines, e.g. unloaded_level_n


def read_rows(path: Path | str) -> tuple[list[str], list[list[str]], dict[str, str]]:
    """Header, data rows, and the `# key: value` comment lines (metadata)."""
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    meta = {}
    for line in lines:
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            meta[key.strip()] = value.strip()
    rows = [r for r in csv.reader(x for x in lines if not x.startswith("#")) if r]
    if not rows:
        raise ValueError(f"{Path(path).name}: empty file")
    return [h.strip() for h in rows[0]], rows[1:], meta


def find_column(header: list[str], names, path) -> int:
    lower = [h.lower() for h in header]
    for n in names:
        if n in lower:
            return lower.index(n)
    raise ValueError(
        f"{Path(path).name}: no column named {' / '.join(names)} "
        f"(columns: {', '.join(header)})"
    )


def read_csv(path: Path | str) -> Log:
    """A tensiometer log or a plain time/force CSV.

    Drops saturated and unparsable samples and any sample that goes back in time.
    """
    header, rows, meta = read_rows(path)
    it = find_column(header, TIME_COLUMNS, path)
    jf = find_column(header, FORCE_COLUMNS, path)
    scale = FORCE_COLUMNS[header[jf].lower()]
    lower = [h.lower() for h in header]
    jr = lower.index("raw_counts") if "raw_counts" in lower else None
    good = []
    for r in rows:
        if len(r) != len(header) or r[jf].strip() == "":
            continue
        if jr is not None and r[jr].strip() == str(SATURATED):
            continue
        good.append(r)
    if not good:
        raise ValueError(f"{Path(path).name}: no samples")
    t, is_clock = parse_times([r[it].strip() for r in good])
    force = np.array([float(r[jf]) for r in good]) * scale
    keep = np.isfinite(force) & np.isfinite(t)
    t, force = t[keep], force[keep]
    # a sample that goes back in time (e.g. the clock restarted) is dropped
    prev = np.maximum.accumulate(np.concatenate([[-np.inf], t[:-1]]))
    keep = t > prev
    return Log(t[keep], force[keep], is_clock, meta)


def unloaded_level(t: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Unloaded level (sensor offset and slow drift) underneath the launches.

    The 10 % quantile of BASELINE_BIN_S medians in a rolling BASELINE_WINDOW_S.
    """
    k = np.floor((t - t[0]) / BASELINE_BIN_S).astype(int)
    bins, first = np.unique(k, return_index=True)
    med = np.array([np.median(c) for c in np.split(force, first[1:])])
    tc = t[0] + (bins + 0.5) * BASELINE_BIN_S
    lo = np.searchsorted(tc, tc - BASELINE_WINDOW_S / 2)
    hi = np.searchsorted(tc, tc + BASELINE_WINDOW_S / 2, side="right")
    level = np.array([np.quantile(med[a:b], 0.1) for a, b in zip(lo, hi)])
    return np.interp(t, tc, level)


def release_index(T: np.ndarray, ip: int, end: int) -> int:
    """Release: the last sample before the tension drops below RELEASE_FRACTION of
    the peak at `ip` (`end` if it does not drop by then)."""
    drop = np.flatnonzero(T[ip : end + 1] < RELEASE_FRACTION * T[ip])
    return ip + int(drop[0]) - 1 if drop.size else end


def format_clock(t: float, is_clock: bool = True) -> str:
    if not is_clock:
        return f"{t:.1f} s"
    s = int(t)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def find_launches(
    t: np.ndarray,
    force: np.ndarray,
    name: str = "",
    source: str = "",
    is_clock: bool = True,
    offset: float | None = None,
    pad: float = PAD_S,
) -> list[Measurement]:
    """Every launch in a log, with `pad` seconds of context on either side.

    `offset`: the unloaded level [N] if known (e.g. from the full log a launch was
    cut from); by default it is estimated with `unloaded_level`.
    """
    if t.size < 2:
        return []
    base = unloaded_level(t, force) if offset is None else np.full_like(force, offset)
    above = force - base
    hot = np.flatnonzero(above > PULL_N)
    launches = []
    for g in np.split(hot, np.flatnonzero(np.diff(t[hot]) > MERGE_GAP_S) + 1):
        if g.size == 0:
            continue
        i0, i1 = int(g[0]), int(g[-1])
        ip = i0 + int(np.argmax(above[i0 : i1 + 1]))
        if above[ip] < LAUNCH_PEAK_N:
            continue
        # start of the pull: the first sample after the last unloaded one
        low = np.flatnonzero(above[:ip] <= PULL_N)
        j = int(low[-1]) + 1 if low.size else 0
        start_recorded = j > 0 and t[j] - t[j - 1] <= GAP_S
        ir = release_index(above, ip, i1)
        a = int(np.searchsorted(t, t[j] - pad))
        b = int(np.searchsorted(t, t[i1] + pad, side="right"))
        launches.append(
            Measurement(
                name=name,
                source=source,
                t=t[a:b],
                force=force[a:b],
                offset=float(base[j]),
                t_pull=float(t[j]),
                t_release=float(t[ir]),
                peak=float(above[ip]),
                start_recorded=bool(start_recorded),
                clock=format_clock(float(t[j]), is_clock),
            )
        )
    if len(launches) > 1:
        for m in launches:
            m.name = f"{name} {m.clock}"
    return launches


def load_launches(path: Path | str) -> list[Measurement]:
    """All launches in a CSV file (a full log or an extracted launch)."""
    path = Path(path)
    log = read_csv(path)
    level = log.meta.get(OFFSET_KEY)
    launches = find_launches(
        log.t,
        log.force,
        path.stem,
        str(path),
        log.is_clock,
        offset=None if level is None else float(level),
    )
    for m in launches:
        m.meta = dict(log.meta)
    if not launches:
        raise ValueError(
            f"{path.name}: no launch found (no stretch more than "
            f"{LAUNCH_PEAK_N / 1e3:.0f} kN above the unloaded level)"
        )
    return launches


# ---- lining up with a simulated launch ---------------------------------------------

Series = Mapping[str, np.ndarray]


def sim_release_time(s: Series) -> float:
    """Release by the rule used for measurements (`release_index`) on the hook
    tension.  After a back-release that is when the winch throttles down, a few
    seconds before the hook opens."""
    att = np.flatnonzero(s["released"] == 0)
    if not att.size:
        return float(s["t"][-1])
    T = np.asarray(s["T_hook"])[: att[-1] + 1]
    return float(s["t"][release_index(T, int(np.argmax(T)), len(T) - 1)])


def sim_pull_start(s: Series) -> float:
    """First time the hook tension is PULL_N (the same rule as for measurements)."""
    pulling = np.flatnonzero(s["T_hook"] > PULL_N)
    return float(s["t"][pulling[0]]) if pulling.size else 0.0


def line_up(m: Measurement, mode: str = "release", ref: Series | None = None):
    """Time axis of `m` [s] with its release (or start of the pull) at the moment
    of the simulated launch `ref`, or at t = 0 without one; plus `m.shift`."""
    if mode == "release":
        return (
            m.t
            - m.t_release
            + (sim_release_time(ref) if ref is not None else 0.0)
            + m.shift
        )
    if mode == "pull":
        return (
            m.t - m.t_pull + (sim_pull_start(ref) if ref is not None else 0.0) + m.shift
        )
    raise ValueError(f"unknown alignment {mode!r}; use one of {', '.join(ALIGN_MODES)}")


def traces(
    measurements, mode: str = "release", ref: Series | None = None, zero: bool = True
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{label: (lined-up time [s], tension [N])} for plots.compare / launch_detail."""
    return {m.label: (line_up(m, mode, ref), m.tension(zero)) for m in measurements}
