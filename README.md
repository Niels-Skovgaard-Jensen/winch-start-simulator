# winch-simulator

Simulator for glider winch launches, written in JAX and solved with
[diffrax](https://docs.kidger.site/diffrax/).

* 3-DOF longitudinal glider (translation + pitch) with aerodynamic polar, stall,
  pitch damping, tow-hook moment and wheel/skid ground contact
* lumped-mass rope with elasticity, weight, aerodynamic drag and rope-on-grass
  friction (steel and Dyneema presets)
* two winch models: tension-controlled, and power-limited engine
* pilot (attitude PID with rotation / climb / top-of-launch schedule) and winch driver
* release, back-release, weak-link and rope-in events; free flight after release
* sensitivities of the release height to all ~80 parameters, with units, via
  `jax.grad` through the ODE solve
* ropes from datasheet values (diameter, mass per 100 m, breaking load,
  elongation or EA) or TOML files
* ISA air density vs height (airfield elevation, ISA temperature offset);
  indicated vs true airspeed
* catalogue of gliders: Ka 8, ASK 13, LS4, Discus CS, ASK 21, ASG 29, DG-1000
  (approximate data)
* measured rope tension from tensiometer logs (CSV), drawn over the simulated
  tension at the hook

The physics and the numerical approach are written up in Typst in
[`docs/physics.typ`](docs/physics.typ) (bibliography in `docs/refs.yml`). Build the
PDF with

```sh
typst compile docs/physics.typ    # -> docs/physics.pdf
```

## Example results

Seven gliders on 1200 m of Dyneema with a power-limited engine winch
(`uv run main.py --winch engine`). Every launch ends the usual way: the winch
driver throttles down near the top, the rope goes slack and the tow hook's
back-release opens.

![Comparison of seven gliders](docs/images/compare_gliders.png)

One launch in detail (ASK 21, tension winch): rope shapes during the climb,
tension at the winch and at the hook, speeds, angles, winch power and the pilot's
elevator input.

![ASK 21 launch in detail](docs/images/detail_ask21.png)

## Sensitivity of the release height

Every run also reports how much each of the ~80 parameters matters for the
release height. `jax.grad` differentiates straight through the ODE solve,
including the moment the launch ends. One backward pass gives ∂h/∂p for every
parameter, and the results agree with finite differences to within 0.5 %. Each
row shows the parameter's value and unit, ∂h/∂p (e.g. metres per mm of rope
diameter), the height change for +10 %, and the *elasticity* (% height per %
parameter). The table is sorted by elasticity. These are local values: for big
changes, such as steel vs Dyneema, run the simulation instead.

ASK 21, 1200 m Dyneema, tension winch (h = 489 m), largest effects:

| parameter | ∂h/∂p | Δh for +10 % | elasticity |
|---|---|---|---|
| rope length (1200 m) | 0.39 m/m | +47 m | 0.97 |
| winch pull (5.07 kN) | 50 m/kN | +26 m | 0.52 |
| glider mass (470 kg) | −0.48 m/kg | −22 m | −0.46 |
| pilot: end of nose lowering (70°) | 1.1 m/deg | +8 m | 0.15 |
| pilot: climb attitude (40°) | 1.5 m/deg | +6 m | 0.12 |
| pilot: target climb speed (105 km/h) | −0.55 m/(km/h) | −6 m | −0.12 |
| rope diameter (5 mm) | −6.8 m/mm | −3 m | −0.07 |
| rope drag coefficient (1.2) | −28 m | −3 m | −0.07 |

Rope length, winch pull and glider mass dominate. After them come the pilot's
technique and the rope's drag. Rope stretch (EA) hardly matters for height. The
full table is written to `results/sensitivity_<glider>_<rope>_<winch>.csv`, and
`--only rope` filters it.

## Measured launches

Rope tension measured on real launches can be drawn over the simulated tension at
the hook. The club's tensiometer, a load cell in line with the rope at the glider
end, logs CSV files. `winch_sim/measured.py` reads them and finds the launches in a
log: stretches that peak more than 2 kN above the unloaded level. It subtracts the
sensor offset and detects the start of the pull and the release. Measured and
simulated launches are lined up at release or at the start of the pull, with the
same rule for both. Release is the moment the tension drops below 25 % of its peak.
Five launches from 2026-09-27 (Discus CS and ASK 13, 1050 m of Dyneema, engine
winch, 6 m/s headwind) are in [`data/tensiometer/`](data/tensiometer/README.md). The
files record these launch conditions, and *Use launch conditions* in the desktop
app sets up a matching simulation.

```sh
uv run main.py --gliders "Discus CS" "ASK 13" --rope-length 1050 --winch engine \
    --wind 6 --measured data/tensiometer/2026-09-27/*.csv
```

![Measured launches over simulated ones](docs/images/compare_measured.png)

*Thin lines: the five measured launches, lined up at release with the Discus CS.
As with the power-limited engine winch, the measured tension builds up during the
climb while the reel speed falls. The default start follows these logs: the rope
has no slack (the driver has taken it up) and the throttle opens over 6 s. The
measured tension builds up over about 7 s. With the default winch setting (pull
1.1 × weight at 15 m/s) the simulation pulls harder than the winch did that day.
In the climb it reaches 4.6 kN for the Discus CS (3.8 to 4.3 kN measured) and
6.6 kN for the ASK 13 (5.0 to 5.7 kN measured).*

## Desktop app

```sh
uv run winch-gui          # or: uv run python -m winch_sim.gui
```

A Qt (PySide6) window with:

* **settings:** every model parameter in tabs (launch & air, glider, pilot, rope,
  winch, numerics), shown in display units with the descriptions from
  `params.py`. Presets fill them in: a glider + pilot from the catalogue, a rope
  preset or TOML file, and a tension/engine winch scaled to the glider weight.
  Any value can then be edited.
* **Run simulation** (Ctrl+R) and **Stop** (Esc). Simulations run in a separate
  worker process, so a running simulation can be stopped at once; a fresh worker
  starts right away. The worker compiles the model in the background when the app
  starts (about 15 s, shown in the status bar). After that a run takes a couple of
  seconds, and it only recompiles when the rope-segment count or the solver
  settings change, not when physical values change.
* **results:**
  * *Runs*: summary table of all runs; tick runs to compare them, reload a run's
    settings, export a time series as CSV.
  * *Measured*: imported tensiometer launches (File → Import measured tension,
    Ctrl+I; a log with several launches gives one row each), with the detected
    start of the pull and release. Ticked launches are drawn over the hook tension
    in *Comparison* and *Detail*. Choose whether to line them up at release or at
    the start of the pull, and shift single traces in time. *Use launch
    conditions* sets glider, rope, winch, rope length and wind as recorded in the
    file.
  * *Comparison* and *Detail*: the plots, with zoom/pan.
  * *Rope & glider*: rope shape and glider position with a time slider and playback.
  * *Sensitivity*: sortable, filterable table, exportable as CSV.
* **File → Save/Load settings** as JSON (SI units).

![Rope & glider view of the desktop app](docs/images/gui_rope_view.png)

*Rope & glider view: a Ka 8 with the winch pulling 2.2 × its weight. The weak
link breaks just after lift-off, and the banner shows why.*

## Usage

```sh
uv run main.py                                    # all gliders, Dyneema, tension winch
uv run main.py --rope steel --winch engine
uv run main.py --gliders "ASK 21" LS4 --pull 1.0 --rope-length 1000 --wind 5
uv run main.py --rope-file ropes/example_dyneema_6mm.toml    # rope from a datasheet
uv run main.py --rope steel --rope-param mu=0.09 --rope-param EA=1.2e6
uv run main.py --gliders "ASK 13" --only rope   # sensitivity table: rope parameters only
uv run main.py --field-elevation 1500 --isa-dt 15 # hot, high airfield (ISA atmosphere)
uv run main.py --rope-length 10000 --segments-per-km 10 --t-max 20000
uv run main.py --gliders "ASK 13" --measured data/tensiometer/2026-09-27/*.csv
uv run main.py --measured my_log.CSV --align pull  # line up at the start of the pull
uv run pytest
```

`main.py` prints a comparison table, writes plots to `results/`, and prints the
sensitivity of the release height to every parameter (CSV in `results/`;
`--no-sensitivity` skips it).

From Python:

```python
import winch_sim
from winch_sim.gliders import CATALOGUE
from winch_sim.cable import STEEL
from winch_sim.winch import tension_winch
from winch_sim.params import Env, Launch
from winch_sim.simulate import solve_launch, summarize, time_series

glider, pilot = CATALOGUE["ASK 21"]
launch = Launch(
    glider=glider,
    rope=STEEL,
    winch=tension_winch(glider, 1.1),
    pilot=pilot,
    env=Env(wind_ref=3.0),
    rope_length=1200.0,
    slack=0.0,
)
sol = solve_launch(launch, n_segments=12)
print(summarize(launch, sol))
series = time_series(launch, sol)  # numpy arrays for plotting
```

`simulate.solve_batch([...])` runs many launches as one `vmap`-ed program.

## Layout

| file | contents |
|---|---|
| `winch_sim/params.py` | parameter pytrees (`Glider`, `Rope`, `Winch`, `Pilot`, `Env`, `Launch`) |
| `winch_sim/gliders.py` | glider catalogue and handbook-data → coefficients |
| `winch_sim/aero.py` | wind, air data, lift/drag/moment |
| `winch_sim/cable.py` | lumped-mass rope, rope presets, elastic catenary (validation) |
| `winch_sim/winch.py` | winch models and presets |
| `winch_sim/pilot.py` | pilot model |
| `winch_sim/ground.py` | smooth ground contact |
| `winch_sim/dynamics.py` | state, vector field, diagnostics |
| `winch_sim/simulate.py` | diffrax solve with events, batching, summaries |
| `winch_sim/sensitivity.py` | d(release height)/d(parameter) via `jax.grad` |
| `ropes/*.toml` | example rope definitions |
| `winch_sim/plots.py` | figures |
| `winch_sim/report.py` | summary columns, sensitivity CSV (shared by CLI and GUI) |
| `winch_sim/measured.py` | measured tension: read tensiometer logs, find launches, line up with simulations |
| `winch_sim/gui/` | Qt desktop app (`app.py`), auto-generated settings forms (`forms.py`) |
| `data/tensiometer/` | measured example launches (see its README) |
| `scripts/extract_launches.py` | cut the launches out of tensiometer logs, one CSV each |
