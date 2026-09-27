"""Cut the launches out of tensiometer logs: one CSV per launch, original rows.

    uv run scripts/extract_launches.py ../tensiometer/logs/2026-09-27/RXLOGS \
        --out data/tensiometer/2026-09-27

Every launch that `winch_sim.measured.find_launches` finds in a log file is written
to <out>/<log file>_<HHMMSS>.csv (logger clock at the start of the pull): the log's
rows unchanged, from CONTEXT_S before the pull to CONTEXT_S after the tension has
settled.  Comment lines on top record the source and the unloaded level of the full
log (the sensor offset), which a short cut-out may not show: the radio link often
drops out while the rope end is on the ground.  The GUI (File -> Import measured
tension) and `main.py --measured` read these files like the full logs.
"""

import argparse
import csv
from pathlib import Path

from winch_sim.measured import (
    OFFSET_KEY,
    TIME_COLUMNS,
    find_column,
    find_launches,
    parse_times,
    read_csv,
    read_rows,
)

CONTEXT_S = 60.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="+", type=Path, help="log files or directories")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = []
    for p in a.logs:
        files += sorted(p.glob("*.[cC][sS][vV]")) if p.is_dir() else [p]
    a.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            log = read_csv(f)
        except ValueError as e:
            print(f"skipping {e}")
            continue
        header, rows, _ = read_rows(f)
        rows = [r for r in rows if len(r) == len(header)]
        it = find_column(header, TIME_COLUMNS, f)
        times, _ = parse_times([r[it].strip() for r in rows])
        found = find_launches(
            log.t, log.force, f.stem, str(f), log.is_clock, pad=CONTEXT_S
        )
        for m in found:
            keep = [r for r, x in zip(rows, times) if m.t[0] <= x <= m.t[-1]]
            out = a.out / f"{f.stem}_{m.clock.replace(':', '')}.csv"
            with open(out, "w", newline="") as fh:
                fh.write(
                    f"# source: {f}\n"
                    f"# extracted by: scripts/extract_launches.py\n"
                    f"# {OFFSET_KEY}: {m.offset:.2f}\n"
                )
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(header)
                w.writerows(keep)
            print(
                f"{out}: pull at {m.clock}, peak {m.peak / 1e3:.2f} kN, "
                f"{m.duration:.1f} s to release, unloaded level {m.offset:.0f} N"
                + ("" if m.start_recorded else " (start of the pull not recorded)")
            )


if __name__ == "__main__":
    main()
