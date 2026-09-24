#import "../template.typ": *

// Operation, numerics and results: pilot, events, solver, results,
// sensitivities, limitations, literature.

= Pilot <sec-pilot>

*What a real pilot does.* During the ground run the pilot holds a little back
stick, so the glider lifts off cleanly and does not nose over. After lift-off
the glider must not climb steeply while it is still close to the ground, so the
pilot lets it rotate into the climb gradually as it gains height. In the full
climb the pilot watches the airspeed: when the glider is too fast the pilot
pulls the nose higher, which puts more of the rope pull into climbing and slows
the glider down; when it is too slow the pilot lowers the nose. Near the top,
when the glider is high above the winch, the pilot lowers the nose. After
release the pilot flies away in a normal glide.

*What the model does.* The pilot model has two parts. The first part says which
pitch attitude the pilot wants, $theta_"ref"$. It is one smooth formula of the
flight state, with no switches between phases. The second part moves the
elevator to reach that attitude.

*Assumptions* (see also @sec-overview). The pilot is a simple, smooth control
law, not a model of a human:

- The pilot knows the pitch attitude $theta$, the pitch rate $q$, the height
  $h$, the indicated airspeed $V_"IAS"$ and the elevation angle $beta$ of the
  glider seen from the winch, exactly and without noise.
- There is no reaction time. The only delay is a first-order stick lag of
  0.25 s.
- Every glider is flown with the same schedule. Only the climb attitude and the
  target speed differ between gliders (@o-tab-pilot).
- The pilot does not use the release knob by default (see @sec-events).

*Target attitude.* The target attitude is

$
  theta_"ref" = (1 - s_"top") [theta_g + s_"rot" (theta_"climb" - theta_g)
    + a(h) thin K_V thin c(V_"IAS" - V_"target")] + s_"top" theta_"top"
$ <eq-pilot-ref>

while the glider is on the rope, and $theta_"ref" = theta_"glide"$ after
release. The symbols are:

- $theta_g = theta_"rest" + 3 degree$: the ground-run attitude. This is the
  attitude at rest on the main wheel and tail, plus 3° of back stick.
- $s_"rot" = S((h - 2 "m") \/ (30 "m" - 2 "m"))$: the rotation into the climb.
  It goes from 0 at 2 m height to 1 at 30 m. Here $S(x) = 3x^2 - 2x^3$ for $x$
  clipped to $[0, 1]$ is the "smoothstep" function, a smooth ramp from 0 to 1.
- $theta_"climb"$: the full-climb attitude, 35° or 40°.
- $c(Delta V)$: the airspeed error $Delta V = V_"IAS" - V_"target"$, limited
  softly to the range $-4$ to $+2$ m/s (the corners are rounded over 1 m/s).
  With $K_V = 2.5 degree$ per m/s the pilot therefore raises the nose by up to
  5° when too fast and lowers it by up to 10° when too slow.
- $a(h) = sigma((h - 0.5 "m") \/ 0.1 "m")$, with $sigma$ the logistic sigmoid:
  a smooth "airborne" factor. It is 0 on the ground and 1 from about 1 m
  height, so the speed correction only acts in the air.
- $s_"top" = S((beta - 55 degree) \/ (70 degree - 55 degree))$: lowering the
  nose at the top. $beta$ is the elevation of the glider's centre of gravity
  seen from the winch (the straight line from the winch to the glider, not the
  rope direction). As $beta$ goes from 55° to 70° the target moves to
  $theta_"top" = 5 degree$.
- $theta_"glide" = 0 degree$: the attitude after release.

*Elevator.* The pilot moves the elevator like a PID controller (proportional,
integral and derivative action), with a deflection limit and a lag:

$
  delta_(e,"cmd") = delta_"max" tanh((-(K_p e + K_i e_I) + K_d q) / delta_"max"),
  quad dot(delta)_e = (delta_(e,"cmd") - delta_e) / tau_p,
  quad dot(e)_I = a(h) e - (1 - a(h)) e_I
$ <eq-elevator>

Here $e = theta_"ref" - theta$ is the attitude error and $e_I$ its time
integral. A negative elevator deflection (trailing edge up) pitches the nose up,
hence the minus sign. The $K_d q$ term damps the pitch rate. The $tanh$ limits
the command smoothly to $plus.minus delta_"max"$, and $tau_p$ is the stick lag.
The integral only builds up in the air; on the ground ($a = 0$) it decays back
to zero, so it cannot wind up while the tail is still on the grass. The default
values are in @o-tab-pilot.

#figure(
  booktab(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    ([symbol], [default], [meaning]),
    [$theta_g - theta_"rest"$], [3°], [back stick in the ground run],
    [$h_"rot0"$, $h_"rot1"$], [2 m, 30 m], [rotation into the climb starts, ends],
    [$theta_"climb"$], [35° or 40°], [full-climb attitude (per glider)],
    [$V_"target"$], [85–115 km/h], [target climb speed, IAS (per glider)],

    [$K_V$], [2.5° per m/s], [attitude change per unit of speed error],
    [$beta_"top0"$, $beta_"top1"$], [55°, 70°], [lowering the nose starts, ends],
    [$theta_"top"$, $theta_"glide"$], [5°, 0°], [attitude at the top, after release],
    [$K_p$, $K_i$, $K_d$], [1.2, 0.8 s#super[−1], 0.6 s], [elevator gains],
    [$tau_p$, $delta_"max"$], [0.25 s, 25°], [stick lag, elevator limit],
    [`release_angle`], [180° (off)], [optional pilot release (@sec-events)],
    [`back_release_angle`], [110°], [back-release of the hook (@sec-events)],
  ),
  caption: [Pilot parameters (`gliders.make_pilot`). Per glider: Ka 8 35°,
    85 km/h; ASK 13 35°, 100 km/h; LS4 and ASK 21 40°, 105 km/h; ASG 29 and
    DG-1000 40°, 115 km/h.],
) <o-tab-pilot>

= End of the launch: events <sec-events>

*How a launch ends in reality.* When the glider is high above the winch, the
winch driver throttles back. The rope pull stops, and the pilot lowers the nose.
The rope now hangs from the hook, and its weight and the drogue parachute pull
the hook end down and back. When the rope pulls from far enough below and behind,
the back-release in the tow hook opens by itself and drops the rope. The pilot
can also pull the release knob at any time. If the rope pull gets too large, the
weak link in the rope breaks.

*How it ends in the model.* The driver's throttle-down is part of the winch
model (@sec-winch): as the elevation $beta$ goes from 65° to 70°, the throttle
goes smoothly down to zero (idle). The rope slackens and sags, and the
back-release trips. This is the normal end of a launch in the model, and it
needs no decision by the pilot.

The launch ends at the first of the events in @tab-events. Each event is a
function of the time and the state that is negative during the launch; the event
fires when it becomes positive (`no_liftoff` is a true/false condition
instead).

#figure(
  booktab(
    columns: (auto, 1fr),
    ([event], [condition]),
    [`release`],
    [Pilot release: the local rope angle at the hook, $phi_c$, exceeds
      `Pilot.release_angle`. $phi_c$ is measured _below the horizontal_, along
      the rope from the hook to the nearest rope node. Off by default (180°),
      because pilots cannot judge the rope angle; set e.g. 72–90° to model a
      pilot who releases on purpose near the top.],

    [`back_release`],
    [Back-release of the hook: the angle $phi_b$ between the rope (hook to
      nearest node) and the fuselage axis, measured downwards, exceeds
      `Pilot.back_release_angle` = 110°. Roughly $phi_b = phi_c + theta$. *The
      normal end of a launch.*],

    [`weak_link`],
    [The tension in the rope segment at the hook exceeds the glider's weak-link
      rating (the parachute drag and the rope-end weight are not included).],

    [`rope_in`], [Less than 30 m of (unstretched) rope is left out.],
    [`no_liftoff`],
    [Still less than 1 m above the rest position after 120 s, e.g. because the
      winch cannot drag a heavy rope. A failed launch: no sensitivities are
      computed.],
  ),
  placement: auto,
  caption: [End-of-launch events, in the order of `simulate.EVENTS`. The first
    four (`LAUNCH_END_EVENTS`) end a launch normally.],
) <tab-events>

If no event fires before the time limit $t_"max"$ (@sec-solver), the event is
reported as `t_max`. The _release height_ in all results is the height of the
glider's centre of gravity above its rest height, at the instant of the event.

*What happens at the top.* In all the default runs of @sec-results the launch
ends with the back-release. Once the throttle-down starts at 65°, the glider
gains only another 5–17 m. The back-release trips between 0 and 6 s after the
glider passes 70° elevation, at a hook tension of 0.1–0.9 kN and 64–109 km/h
IAS. There are two patterns:

- The Ka 8, ASK 13, ASK 21 and (on steel) DG-1000 still have time to lower the
  nose to 1–10° before the pull stops. The rope then has to sag to 100–109°
  below the horizontal before it is 110° below the fuselage axis. This takes
  1.5–2.5 s on Dyneema and about 5–6 s with the heavier steel rope. (With the
  engine winch the ASK 21 lies in between.)
- The LS4, ASG 29 and (on Dyneema) DG-1000 climb faster at the top, and their
  nose is still 24–30° up when the pull stops. With this attitude the rope is
  already 110° below the fuselage axis when it hangs 80–86° below the
  horizontal, so the back-release trips almost as soon as the throttle is
  closed, at about 70° elevation.

*Optional pilot release.* A pilot who releases at a set rope angle
(`release_angle` < 180°) ends the launch slightly earlier and lower. For the
ASK 21 on 1200 m Dyneema with the tension winch, a release at 72°, 76°, 80°,
85° or 90° gives 481, 483, 484, 486 or 488 m, against 491 m with the
back-release. At 72° the glider is still being pulled (2.0 kN at the hook),
because the driver has only just started to throttle down; from 80° on the hook
tension at release is about 0.3 kN or less. The release height depends only a little
on exactly when the launch ends near the top.

= Solving it with diffrax <sec-solver>

*The idea.* The whole system (glider, rope nodes, winch drum, pilot) is one set
of ordinary differential equations, $dot(y) = f(t, y, p)$, with the state $y$
of @tab-state and the parameters $p$. It is solved with the JAX library diffrax.
Three choices shape the numerics:

+ *Everything is smooth.* Every kink in the physics (rope going slack, ground
  contact, friction, stall, the pilot's schedule, the winch throttle) is
  rounded off with softplus, sigmoid, tanh or smoothstep functions. So there is
  a single continuous ODE for the whole launch, and an adaptive solver never
  has to stop at a discontinuity or switch between modes.
+ *Everything is a JAX function.* The vector field and all parameters are JAX
  pytrees. The whole solve can be compiled (`jit`), run for many launches at
  once (`vmap`), and differentiated (`grad`).
+ *The end of the launch is an event.* The solver finds the exact instant when
  an event function of @tab-events changes sign.

The details:

/ Vector field: `dynamics.vector_field(t, y, args)`. `y` is an `equinox.Module`
  (`State`); `args` holds all parameters as `equinox.Module`s with float leaves
  (`params.py`) and a flag `attached` (1 on the rope, 0 after release).
/ Solver: `Tsit5`, an explicit Runge–Kutta method of order 5. A
  `PIDController` sets the step size, with rtol = atol = $10^(-6)$,
  `pcoeff` = 0.3 and `icoeff` = 0.4; the first step is $10^(-3)$ s. The fastest motion is the axial wave in
  the rope, with a frequency of about $omega approx sqrt(E A \/ mu) \/ ell_0$,
  where $ell_0 = L_0 \/ N$ is the rest length of one rope segment. For the
  Dyneema rope with 100 m segments this is about 60 rad/s at the start; it
  rises to about 150 rad/s near the top, as reel-in shortens the segments. The
  catalogue launches need 2200–4300 steps. The implicit solver `Kvaerno5`
  (`solver="kvaerno5"`) gives the same release height and time, but for the
  ASK 21 it takes 14 700 steps instead of 3100 and several times as long:
  the problem is not stiff enough for an implicit method to pay off.
/ Events: `diffrax.Event` with the condition functions of @tab-events, only
  for crossings from negative to positive (`direction=True`). An
  `optimistix.Newton` root finder (rtol = atol = $10^(-8)$) locates the event
  time, and `sol.event_mask` says which event fired.
/ Two passes: Pass 1 runs to the event and saves only the final state
  (`SaveAt(t1=True)`). So the time limit can be huge at no cost: $t_"max" =
  10^5$ s by default (`simulate.T_MAX`, `--t-max`), with at most
  $5 dot 10^6$ steps (`MAX_STEPS`). Pass 2 repeats the same, deterministic
  solve from 0 to the event time and saves 2000 evenly spaced points for plots
  and summaries. Long and short launches thus get the same number of points.
/ Free flight: A second `diffeqsolve` starts from the state at release with
  `attached = 0` and runs for 8 s (200 points). The glider flies on with the
  glide attitude, the rope falls with a free end, and the winch driver brakes
  the drum.
/ Derived quantities: Tensions, angles, load factor, power and so on are
  recomputed from the saved states with the same `evaluate` function that
  gives the derivatives, `vmap`-ed over time. So the plots can never disagree
  with the dynamics.
/ Gradients: `sensitivity.release_height` is the same pass-1 solve (always
  with `Tsit5`) with `RecursiveCheckpointAdjoint` for reverse-mode
  differentiation. `eqx.filter_value_and_grad` over the whole `Launch` pytree
  gives the sensitivity to every parameter at once (@sec-sensitivity).
/ Batching: `simulate.solve_batch(launches)` stacks the parameter pytrees and
  `vmap`s the solve. The six catalogue gliders run as one compiled program: about
  5 s the first time (mostly compilation), then about 1 s.

= Example results <sec-results>

*Set-up.* 1200 m rope, no wind, sea level with the standard atmosphere (ISA),
rope slack 0.2 %, rope modelled with 12 segments of 100 m (the default of 10
segments per km, `--segments-per-km`). The winch pull is 1.1 × the glider's
weight. For the tension winch this is the constant pull $F_"max"$. For the
engine winch (200 kW, 12 kN) the driver sets the throttle so that the pull is
1.1 × weight at 15 m/s reel speed (@sec-winch). Reproduce @tab-results with

```sh
uv run main.py --no-sensitivity [--rope steel] [--winch engine]
```

Other options change the conditions: `--rope-length`, `--wind` (headwind at
10 m), `--pull`, `--field-elevation`, `--isa-dt`, `--segments-per-km`, `--t-max`,
`--rope-file` and `--rope-param` (@sec-rope-input). Plots go to `results/`.

#let grp(body) = table.cell(colspan: 11, align: left, emph(body))
#figure(
  text(8.5pt, booktab(
    columns: 11,
    align: (left, right, right, right, right, right, right, right, right, right, right),
    (
      [glider],
      [$V_W$ \ km/h],
      [$h_"rel"$ \ m],
      [time \ s],
      [roll \ m],
      [max $V$ \ km/h],
      [min $V$ \ km/h],
      [max $T_"hook"$ \ kN],
      [max $n$],
      [max $P$ \ kW],
      [rope SF],
    ),
    grp[Dyneema, tension winch],
    [Ka 8], [100], [462], [52.1], [46], [*128*], [64], [3.10], [2.59], [92], [8.7],
    [ASK 13], [120], [478], [44.9], [52], [*149*], [78], [5.00], [2.81], [173], [5.4],
    [LS4], [130], [459], [42.8], [69], [*150*], [93], [4.07], [2.96], [124], [6.6],
    [ASK 21], [150], [491], [46.1], [51], [150], [72], [5.35], [3.18], [181], [5.0],
    [ASG 29], [150], [467], [40.0], [74], [*157*], [102], [4.75], [2.97], [154], [5.7],
    [DG-1000], [150], [496], [40.1], [62], [*164*], [95], [6.97], [3.26], [258], [3.9],
    table.hline(stroke: 0.3pt),
    grp[Steel, tension winch],
    [Ka 8], [100], [438], [57.0], [56], [*121*], [64], [3.34], [2.51], [85], [4.5],
    [ASK 13], [120], [464], [48.8], [59], [*143*], [75], [5.24], [2.71], [162], [3.0],
    [LS4], [130], [444], [43.5], [81], [*145*], [93], [4.07], [2.96], [129], [3.8],
    [ASK 21], [150], [474], [50.4], [57], [146], [70], [5.73], [3.09], [170], [3.0],
    [ASG 29], [150], [454], [40.6], [86], [*153*], [101], [4.72], [3.07], [160], [3.3],
    [DG-1000], [150], [494], [45.1], [68], [*161*], [75], [7.27], [3.19], [251], [2.3],
    table.hline(stroke: 0.3pt),
    grp[Dyneema, engine winch],
    [Ka 8], [100], [452], [53.6], [70], [*107*], [67], [3.97], [2.33], [68], [6.8],
    [ASK 13], [120], [448], [47.0], [70], [119], [82], [6.02], [2.26], [136], [4.4],
    [LS4], [130], [422], [45.0], [95], [124], [96], [4.80], [2.36], [104], [5.6],
    [ASK 21], [150], [455], [46.2], [64], [121], [88], [6.29], [2.58], [149], [4.3],
    [ASG 29], [150], [416], [42.3], [104], [130], [103], [5.40], [2.45], [121], [4.9],
    [DG-1000], [150], [448], [41.9], [80], [132], [105], [8.04], [2.53], [197], [3.3],
  )),
  placement: auto,
  caption: [Launch results for the catalogue gliders, 1200 m rope, no wind.
    Every launch ends with the back-release. $V_W$: placarded maximum winch
    speed; $h_"rel"$: release height; time: launch time; roll: ground run to
    lift-off; max/min $V$: highest IAS, and lowest IAS once above 5 m; max
    $T_"hook"$: highest tension at the hook; max $n$: highest wing load factor
    $L \/ W$; max $P$: highest power at the drum; rope SF: breaking load divided
    by the highest rope tension. Bold: peak speed more than 1 % above $V_W$.],
) <tab-results>

*Observations.*

- *Height.* On Dyneema with the tension winch the release heights are
  459–496 m, 38–41 % of the rope length, after 40–52 s.
- *Steel* costs 2–24 m of height (most for the light Ka 8, least for the heavy
  DG-1000), gives a 6–12 m longer ground run, and lowers the rope safety
  factor: the steel rope is heavier and has a lower breaking load (17 kN
  against 27 kN). The DG-1000 on steel uses 43 % of the rope's strength.
- *Speed.* With the constant pull of the tension winch, the glider accelerates
  hard in the ground run and the early climb. The speed peaks about 9–11 s
  after the start, at 34–44 m height, during the rotation into the climb. This
  peak is above $V_W$ for five of the six gliders: by 28 % for the Ka 8, 24 %
  for the ASK 13, 15 % for the LS4, 10 % for the DG-1000 and 4 % for the
  ASG 29; the ASK 21 just reaches its $V_W$. A real driver would set a lower
  pull for the Ka 8 and open the throttle more gently.
- *Engine winch.* The pull of the power-limited engine winch falls as the reel
  speeds up, so it limits the speed by itself. The peak speeds (107–132 km/h)
  and load factors (2.3–2.6) are lower, and only the Ka 8 is still above its
  $V_W$ (by 7 %). The price is 10–51 m of release height; the fast gliders
  (ASG 29, DG-1000) lose most. The engine pulls harder when the glider slows
  the rope in the rotation, so the peak hook tension is 14–28 % _higher_ than
  with the tension winch: about 80 % of the weak-link rating for the Ka 8
  (4.0 of 5 kN) and the DG-1000 (8.0 of 10 kN).
- *Load factor.* With the tension winch the highest wing load factor, 2.5–3.3,
  occurs at 31–34 m height, during the rotation, which the simple pilot flies
  rather briskly.
- *Top of the launch.* The lowest airspeed (64–105 km/h) is usually reached at
  the very end, as the glider slows after the throttle-down. For the Ka 8 this
  is 64 km/h, about 12 % above its 1 g stall speed in the model (57 km/h).
- *Power.* The ideal tension winch needs up to 258 kW at the drum for the
  DG-1000, more than the 200 kW of the engine winch.

= Sensitivities of the release height <sec-sensitivity>

*What is computed.* How much would the release height $h$ change if one
parameter $p$ changed a little? Every `uv run main.py` run (unless
`--no-sensitivity`) answers this for all 81 model parameters at once: glider,
rope, winch, pilot, environment and layout. It computes the derivative
$partial h \/ partial p$ with one reverse-mode pass of `jax.grad` through the
diffrax solve (`sensitivity.py`, `RecursiveCheckpointAdjoint`). The event time
is found by the root finder, and diffrax differentiates it implicitly, so the
change of the release instant is included. The gradients agree with central
finite differences to within 0.5 % (`tests/test_sensitivity.py`). Very small
sensitivities are only as accurate as the solver tolerance, so this solve uses
rtol = atol = $10^(-8)$. The gradient takes about 10 s to compile and then about
1 s per launch; a whole `main.py` run for one glider takes about 30 s.

*How to read the table.* For each parameter the table gives its value,
$partial h \/ partial p$ in metres per unit of the parameter (per degree for
angles), the height change for a +10 % change of the parameter, and the
_elasticity_ $(p \/ h) thin partial h \/ partial p$: the percentage change of
height for a 1 % change of the parameter. Rows are sorted by the size of the
elasticity. All other parameters are held fixed. For example, the winch pull,
preset as 1.1 × weight, does not follow a change of glider mass. These are
local, linear estimates. `--only rope` restricts the printed table to one group,
`--top` sets the number of rows, and the full table goes to
`results/sensitivity_<glider>_<rope>_<winch>.csv`.

#figure(
  text(9pt, booktab(
    columns: 6,
    align: (left, left, right, right, right, right),
    (
      [parameter],
      [name],
      [value],
      [$partial h \/ partial p$],
      [$Delta h$ (+10 %)],
      [elasticity],
    ),
    [rope length], [`rope_length`], [1200 m], [0.393 m/m], [+47.2 m], [0.96],
    [winch pull], [`winch.F_max`], [5.07 kN], [51.1 m/kN], [+25.9 m], [0.53],
    [winch throttle], [`winch.throttle`], [1], [259 m], [+25.9 m], [0.53],
    [glider mass], [`glider.mass`], [470 kg], [−0.48 m/kg], [−22.6 m], [−0.46],
    [gravity], [`env.g`], [9.81 m/s²], [−22.7 m/(m/s²)], [−22.3 m], [−0.45],
    [end of nose lowering], [`pilot.top_beta1`], [70°], [1.08 m/deg], [+7.6 m], [0.15],
    [climb attitude], [`pilot.theta_climb`], [40°], [1.52 m/deg], [+6.1 m], [0.12],
    [target climb speed], [`pilot.V_target`], [105 km/h], [−0.57 m/(km/h)], [−6.0 m], [−0.12],
    [start of nose lowering], [`pilot.top_beta0`], [55°], [0.82 m/deg], [+4.5 m], [0.09],
    [back-release angle], [`pilot.back_release_angle`], [110°], [0.38 m/deg], [+4.2 m], [0.08],
    [rope diameter], [`rope.diameter`], [5 mm], [−6.8 m/mm], [−3.4 m], [−0.07],
    [rope normal drag coeff.], [`rope.CDn`], [1.2], [−28 m], [−3.4 m], [−0.07],
  )),
  caption: [The 12 largest sensitivities: ASK 21, 1200 m Dyneema, tension winch
    ($h = 491$ m). `uv run main.py --gliders "ASK 21" --top 12`.],
) <tab-sensitivity>

*What the table shows.*

- The rope length matters most. The height is almost proportional to it
  (elasticity 0.96): +120 m of rope gives +47 m of height.
- Next come the winch pull and the glider's weight, with about equal and
  opposite effect (+0.53 and −0.46): what counts is the pull per weight.
  `winch.F_max` and `winch.throttle` give the same row because the pull is
  their product, and `env.g` acts through the weight.
- Then the pilot. How late the pilot lowers the nose at the top
  (`top_beta0`, `top_beta1`), the climb attitude and the target speed each
  change the height by 4–8 m for a 10 % change. So does the angle at which the
  back-release trips. These control-law parameters matter more than any single
  aerodynamic parameter of the glider: the largest, the wing area, comes next
  with +3.1 m for +10 %.
- Rope drag (diameter and normal drag coefficient) costs about 3.4 m each for
  +10 %.
- The driver's throttle-down angles have small effects: moving the start from
  65° to 71.5° (+10 %) costs 2.0 m.

Parameters whose default value is zero have zero elasticity and so do not
appear at the top, although their derivative can be large. The most important
one is the headwind: $partial h \/ partial W_"ref" = 22$ m per m/s of headwind
at 10 m height. Another is the throttle floor after the throttle-down
(`fade_floor` = 0): keeping 10 % of the power would add about 5 m (linear
estimate).

*The rope.* For the steel rope with the ASK 13 (`--rope steel --only rope`,
$h = 464$ m) the rope properties change the height by, for +10 %: diameter
−2.4 m, normal drag coefficient `CDn` −2.3 m, mass per length −1.5 m,
parachute drag area −0.5 m, ground friction −0.4 m, rope-end mass −0.4 m,
tangential friction `Cf` −0.1 m. Rope drag and rope weight thus matter about
equally, with drag somewhat ahead. For the much lighter Dyneema rope the weight
hardly matters (−0.2 m for +10 % on the ASK 21). The axial stiffness $E A$ has
essentially _no_ influence on the release height: +10 % changes it by about
1.5 cm. Rope stretch shapes the tension oscillations in the ground run
(@sec-rope), not the energy that reaches the glider.

= Limitations and possible extensions <sec-limits>

- *Pilot and driver.* Both are simple, smooth control laws with perfect
  information and no reaction time (@sec-pilot, @sec-winch). The driver works
  open loop: the throttle depends only on time and on the glider's elevation,
  not on its speed or the rope tension. The driver does not react to
  overspeed; $V_W$ is only reported, not respected. The launch ends by the
  back-release after the throttle-down on a visual cue; a pilot release when
  the pull fades (a tension cue) would be an alternative. As
  @sec-sensitivity shows, the parameters of these control laws affect the
  release height more than any single aerodynamic parameter, just as pilot and
  driver technique do in real life.
- *Gliders.* The glider data are approximate estimates, not manufacturer data.
  The model is longitudinal only: no wing drop, roll or yaw. There is no ground
  effect and no flaps (flap settings matter for the ASG 29 and similar gliders).
- *Rope end.* During the launch the parachute is a fixed, closed drag area
  plus a mass at the hook. After release both disappear: the rope falls with a
  bare free end. The parachute does not open.
- *Reel-in.* Reeling in removes mass evenly from all rope nodes
  (@sec-rope-disc). A "moving-node" formulation that removes mass only at the
  drum would be exact.
- *Winch.* The drum physics is intentionally minimal (one effective mass).
- *Atmosphere.* Steady wind with a power-law profile only; no gusts or
  thermals.
- *Sensitivities* (@sec-sensitivity) are local. For large changes (e.g. steel
  to Dyneema) run the simulation itself. The same gradients could also drive an
  optimisation of the tension profile or the pilot schedule, subject to
  $V <= V_W$, $T <=$ weak link and $n <= n_"max"$ @aq1977.

= Related literature <sec-literature>

*Winch-launch simulation.* Gäb and Santel @gab2011 (RWTH Aachen,
Matlab/Simulink) model the aircraft, pilot, winch, winch operator, cable,
atmosphere and terrain. Their cable is a chain of point masses joined by
spring–damper links, with drag, weight and ground reaction: the same approach as
@sec-rope here. Their pilot and winch operator are PID controllers with reaction
time and neuromuscular delay. They study a reference launch, wind, and too steep
initial climbs. *It is the best reference to validate this model against.* It
grew out of an RWTH student thesis @rwth2008
(#link("https://core.ac.uk/download/pdf/36588266.pdf")[CORE PDF]). That thesis
found that opening the throttle quickly destabilises the phugoid (the slow
pitch–speed oscillation), that the tow-hook position can be optimised, and that
the release height depends on cable type, tow distance, maximum winch force and
especially cable drag. It lists ground effect and ground/cable friction as
missing. Santel's diploma thesis @santel studies winch-launch accidents with
multipoint aerodynamics. Bogan's simpler model @bogan (prescribed flight path,
MathCAD, not peer-reviewed) gives a good feel for how the cable drag grows as the
rope turns broadside to the flow.

*Optimal launch trajectories.* With limits on the lift coefficient and on the
cable acceleration, the optimal trajectory is climb–dive–climb; a limit on the
reel-in speed makes it more realistic @aq1977. This is the benchmark for an
optimisation with `jax.grad` (@sec-limits). See also Eppler on the optimal
release height @eppler and the follow-up work of Pierson and Chen @pierson.

*Safety and winch engineering.* Browning @browning2007 and Hills @hills2007 on
safe winch launching; design formulae for winch hardware @winchdesign; the
Soaring Safety Foundation notes by Smith @smith
(#link("https://www.soaringsafety.org/publications/winches.pdf")[Winches]).

*Cable and tether modelling.* Irvine's elastic catenary @irvine1981 is used to
validate the rope model in @sec-catenary. Modelling of tethers with variable
length @williams2007 relates to the reel-in approach of @sec-rope-disc; the
SkySails model @skysails does optimal control with tether force and reel-out
speed as variables.

*How this model compares with @gab2011.* The structure is the same: a
lumped-mass cable, a PID pilot and a tension-controlled winch operator. The
differences: @gab2011 includes human reaction and neuromuscular delays and a
more detailed winch and operator model. This model keeps the number of rope
segments fixed during reel-in, uses smooth contact models (one continuous ODE
instead of switching between modes), and uses JAX/diffrax for batching and
gradients. Obvious next steps: reproduce the reference launch of @gab2011, and
add reaction delays to the pilot and the winch driver.
