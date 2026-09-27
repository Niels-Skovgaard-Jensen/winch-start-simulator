# Measured launches

Rope tension logged by the club's tensiometer: a load cell in line with the rope at
the glider end, read at about 12 Hz and sent by radio to a receiver at the winch
(firmware and the full logs are in the separate `tensiometer` repository). Each CSV
here is one launch cut out of the receiver's logs by
`scripts/extract_launches.py`. The rows are unchanged (`time_utc, raw_counts,
force_n`). Comment lines at the top give the source log and its unloaded level,
i.e. the sensor offset. The boards have no real-time clock, so the times are the
logger's time since boot.

Import them in the desktop app (File → Import measured tension, or the *Measured*
tab), or on the command line:

```sh
uv run main.py --gliders "Discus CS" "ASK 13" --rope steel --rope-length 1000 \
    --winch engine --measured data/tensiometer/2026-09-27/*.csv
```

## 2026-09-27

Glider: a Discus CS or an ASK 13 (which one flew each launch was not recorded).
1000 m of winch wire. The winch settings were not logged.

| file | event in `plot_logs.py` | clock at the pull | peak | pull → release |
|---|---|---|---|---|
| `LOG00011_011410.csv` | #6 | 01:14:10 | 3.83 kN | > 24 s, start not recorded |
| `LOG00011_014330.csv` | #8 | 01:43:30 | 4.28 kN | 43.1 s |
| `LOG00014_002756.csv` | #10 | 00:27:56 | 4.96 kN | 52.0 s |
| `LOG00014_004933.csv` | #12 | 00:49:33 | 4.32 kN | > 27 s, start not recorded |
| `LOG00015_012257.csv` | #15 | 01:22:57 | 5.69 kN | 43.6 s |

Peak is above the unloaded level. "Start not recorded": the radio link dropped out
during the ground run, so the log only starts during the climb. After release the
rope end is wound in and the gauge reads about 0.5 to 1.3 kN for 15 to 20 s.

`plot_logs.py` marks 16 events on that day. The other 11 peak at 0.24 to 1.3 kN
and are not launches. The long ones (2 to 4 min at 400 to 700 N) look like the rope
being towed out, and the short ones look like handling. The importer keeps only
stretches that peak more than 2 kN above the unloaded level.

Re-create the files with

```sh
uv run scripts/extract_launches.py ../tensiometer/logs/2026-09-27/RXLOGS \
    --out data/tensiometer/2026-09-27
```
