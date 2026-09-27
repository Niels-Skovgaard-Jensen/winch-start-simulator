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
uv run main.py --gliders "Discus CS" "ASK 13" --rope-length 1050 --winch engine \
    --wind 6 --measured data/tensiometer/2026-09-27/*.csv
```

The launch conditions are recorded in each file as `# key: value` lines (`glider`,
`rope`, `winch`, `rope_length_m`, `headwind_m_s`). In the desktop app, *Use launch
conditions* on the *Measured* tab puts them into the settings.

## 2026-09-27

1050 m of Dyneema, engine winch, 6 m/s headwind. Two gliders flew: a Discus CS and
an ASK 13. Which one flew each launch was not written down. The estimate made
afterwards is that the launches logged at 5 kN and above were the ASK 13 (5.04 and
5.78 kN logged, including the sensor offset). The files record that assignment.

| file | event in `plot_logs.py` | glider | clock at the pull | peak | pull → release |
|---|---|---|---|---|---|
| `LOG00011_011410.csv` | #6 | Discus CS | 01:14:10 | 3.83 kN | > 24 s, start not recorded |
| `LOG00011_014330.csv` | #8 | Discus CS | 01:43:30 | 4.28 kN | 43.1 s |
| `LOG00014_002756.csv` | #10 | ASK 13 | 00:27:56 | 4.96 kN | 52.0 s |
| `LOG00014_004933.csv` | #12 | Discus CS | 00:49:33 | 4.32 kN | > 27 s, start not recorded |
| `LOG00015_012257.csv` | #15 | ASK 13 | 01:22:57 | 5.69 kN | 43.6 s |

Peak is above the unloaded level. "Start not recorded": the radio link dropped out
during the ground run, so the log only starts during the climb. After release the
rope end is wound in and the gauge reads about 0.5 to 1.3 kN for 15 to 20 s.

`plot_logs.py` marks 16 events on that day. The other 11 peak at 0.24 to 1.3 kN
and are not launches. The long ones (2 to 4 min at 400 to 700 N) look like the rope
being towed out, and the short ones look like handling. The importer keeps only
stretches that peak more than 2 kN above the unloaded level.

Re-create the files with the command below, then add the `# glider:` lines from
the table by hand:

```sh
uv run scripts/extract_launches.py ../tensiometer/logs/2026-09-27/RXLOGS \
    --out data/tensiometer/2026-09-27 --meta rope=dyneema --meta winch=engine \
    --meta rope_length_m=1050 --meta headwind_m_s=6
```
