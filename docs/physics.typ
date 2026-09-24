// Physics write-up of the winch-launch simulator.
// Build:  typst compile docs/physics.typ    (-> docs/physics.pdf)

#set document(title: [Physics of a glider winch launch])
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show link: set text(fill: rgb("#1f5fbf"))
#show raw.where(block: false): box.with(
  fill: luma(245),
  inset: (x: 1.5pt),
  outset: (y: 2pt),
  radius: 2pt,
)
#show heading.where(level: 1): set block(above: 1.6em, below: 0.9em)
#show figure.where(kind: "note"): set align(left)
#show table: set par(justify: false)
#show figure.where(kind: table): set figure.caption(position: top)

// booktabs-style table: header row plus body cells
#let booktab(columns: auto, align: left, header, ..cells) = table(
  columns: columns,
  align: align,
  stroke: none,
  inset: (x: 5pt, y: 3.5pt),
  table.hline(stroke: 0.8pt),
  table.header(..header.map(h => strong(h))),
  table.hline(stroke: 0.4pt),
  ..cells,
  table.hline(stroke: 0.8pt),
)

#let note(body) = block(
  fill: luma(246),
  stroke: (left: 2pt + luma(170)),
  inset: 8pt,
  width: 100%,
  body,
)

#align(center)[
  #text(17pt, weight: "bold")[Physics of a glider winch launch]
  #v(0.2em)
  #text(11pt)[Model and numerics of `winch_sim`]
]
#v(1em)

This document derives the model implemented in `winch_sim/` and explains how it
is solved with #link("https://docs.kidger.site/diffrax/")[diffrax]. The system
has three coupled parts:

+ the *glider*, a rigid body moving in the vertical plane (3 degrees of freedom:
  two translations and pitch),
+ the *rope*, 1000–1500 m of steel wire or synthetic (Dyneema) rope with its own
  mass, elasticity and aerodynamic drag, partly lying on the runway at the start,
+ the *winch*, a drum that reels the rope in, driven either by an ideal tension
  controller or a power-limited engine,

plus two "controllers": the *pilot* (elevator) and the *winch driver* (throttle).

#figure(
  box(width: 14cm, height: 5.4cm, {
    let ground = 4.7cm
    let hook = (4.7cm, 1.25cm)
    let winch = (12.9cm, ground - 0.2cm)
    // runway
    place(dx: 0cm, dy: ground, line(length: 14cm, stroke: 1.6pt + luma(140)))
    place(dx: 0.1cm, dy: ground + 0.12cm, text(8pt, fill: luma(90))[runway, $z = 0$])
    // glider path (dashed) from the start to the hook position
    place(curve(
      stroke: (paint: luma(120), thickness: 0.8pt, dash: "dashed"),
      curve.move((0.4cm, ground - 0.05cm)),
      curve.cubic((2.2cm, ground - 0.05cm), (3.3cm, 2.6cm), hook),
    ))
    place(dx: 0.25cm, dy: ground - 0.55cm, text(8pt)[start])
    // straight line of sight winch - glider, defines beta
    place(line(start: winch, end: hook, stroke: (paint: luma(150), thickness: 0.6pt, dash: "dotted")))
    // rope: sags below the line of sight
    place(curve(
      stroke: 1.3pt + rgb("#2a78d6"),
      curve.move(hook),
      curve.quad((9.8cm, 3.9cm), winch),
    ))
    place(dx: 5.2cm, dy: 3.55cm, text(8pt, fill: rgb("#2a78d6"))[rope: sags under its weight,])
    place(dx: 5.2cm, dy: 3.9cm, text(8pt, fill: rgb("#2a78d6"))[drag pushes it back])
    // winch drum
    place(dx: winch.at(0) - 0.2cm, dy: winch.at(1) - 0.2cm, rect(width: 0.4cm, height: 0.4cm, fill: black))
    place(dx: winch.at(0) - 0.4cm, dy: ground + 0.12cm, text(8pt)[winch])
    place(dx: 10.6cm, dy: 3.35cm, text(9pt)[$beta$])
    // glider (fuselage + wing) pitched nose-up
    place(dx: hook.at(0) - 0.9cm, dy: hook.at(1) - 0.32cm, rotate(-30deg, origin: center, {
      box(width: 1.8cm, height: 0.5cm, {
        place(dx: 0cm, dy: 0.22cm, line(length: 1.8cm, stroke: 2pt + rgb("#eb6834")))
        place(dx: 0.05cm, dy: 0cm, line(length: 0.25cm, angle: 90deg, stroke: 2pt + rgb("#eb6834")))
      })
    }))
    place(dx: hook.at(0) - 0.08cm, dy: hook.at(1) - 0.08cm, circle(radius: 0.08cm, fill: black))
    place(dx: hook.at(0) + 0.25cm, dy: hook.at(1) - 0.05cm, text(8pt)[hook])
    place(dx: hook.at(0) - 1.0cm, dy: hook.at(1) - 1.15cm, text(8pt)[glider, pitch $theta$])
    // axes
    place(dx: 13.2cm, dy: ground - 0.35cm, text(9pt)[$x$])
    place(dx: 0.05cm, dy: 0cm, text(9pt)[$z$ (up)])
  }),
  caption: [Geometry of a winch launch. The rope sags below the line of sight
    from the winch; $beta$ is the elevation of the glider seen from the winch.],
) <fig-geometry>

= Frames, state and conventions <sec-frames>

- Earth frame: $x$ horizontal, from the glider's start towards the winch, $z$ up.
  The runway is $z = 0$, the winch at $bold(r)_w = (x_w, z_w)$ with
  $x_w approx$ laid-out rope length and $z_w approx 1$ m.
- Body frame: $x_b$ forward along the fuselage, $z_b$ towards the canopy. Unit
  vectors in earth coordinates:
  $hat(bold(x))_b = (cos theta, sin theta)$,
  $hat(bold(z))_b = (-sin theta, cos theta)$.
- Pitch attitude $theta$ and pitching moment $M$ positive nose-up. In the plane,
  the moment of a force $bold(F)$ applied at offset $bold(r)$ from the CG is
  $M = bold(r) times bold(F) = r_x F_z - r_z F_x$.
- A point with body coordinates $bold(b)$ sits at
  $bold(r)_"cg" + R(theta) bold(b)$ and moves with $bold(v) + q bold(r)^perp$,
  where $bold(r)^perp = (-r_z, r_x)$.

#figure(
  booktab(
    columns: 2,
    ([symbol], [meaning]),
    [$bold(r) = (x, z)$, $bold(v)$], [glider CG position and velocity (earth frame)],
    [$theta$, $q$], [pitch attitude and pitch rate],
    [$bold(p)_i$, $bold(u)_i$, $i = 1 .. N - 1$], [rope node positions and velocities],
    [$L_0$], [unstretched length of rope still paid out],
    [$v_r$], [reel-in speed at the drum],
    [$delta_e$], [elevator deflection],
    [$e_I$], [pilot's integrated attitude error],
  ),
  caption: [State vector $y$ (SI units).],
) <tab-state>

With $N = 12$ rope segments that is $6 + 4 dot 11 + 4 = 54$ states. Using
earth-frame instead of body-frame translational velocities is equivalent for a
3-DOF model and makes the coupling with the rope nodes simpler.

= Glider rigid body <sec-glider>

$
  m_"tot" dot(bold(v)) = bold(F)_"aero" + sum_k bold(F)_("gnd", k)
  + bold(F)_"hook" - m g hat(bold(z))
$ <eq-translation>
$
  I_(y y) dot(q) = M_"aero" + sum_k bold(r)_k times bold(F)_("gnd", k)
  + bold(r)_"hook" times bold(F)_"hook",
  quad dot(theta) = q,
  quad dot(bold(r)) = bold(v)
$ <eq-rotation>

$m_"tot" = m + m_"end"$ includes the rope end assembly carried at the hook
(parachute, strop, weak link, half the top rope segment); its weight is part of
$bold(F)_"hook"$.

== Aerodynamics <sec-aero>

Air-relative velocity $bold(v)_a = bold(v) - bold(w)(z)$ with a power-law wind
profile $bold(w) = (-W_10 (z\/10)^(1\/7), 0)$ (a headwind blows towards $-x$).
True airspeed $V = abs(bold(v)_a)$, and

$
  alpha = op("atan2")(-bold(v)_a dot hat(bold(z))_b, space bold(v)_a dot hat(bold(x))_b),
  quad macron(q) = 1/2 rho(z) V^2,
  quad hat(q) = (q macron(c)) / (2 V).
$

*Air density* follows the International Standard Atmosphere (`atmosphere.py`)
at the pressure height $h = h_"field" + z$, with a temperature offset $Delta T$
(`Env.field_elevation`, `Env.isa_dT`; CLI `--field-elevation`, `--isa-dt`):

$
  T = T_0 - lambda h + Delta T,
  quad p = p_0 ((T_0 - lambda h) / T_0)^(g_0 \/ (R lambda)),
  quad rho = p / (R T)
$ <eq-isa>

($T_0 = 288.15$ K, $p_0 = 101325$ Pa, $lambda = 6.5$ K/km; isothermal above
11 km). The same $rho(z)$ is used for the glider, for every rope segment (at its
mid-height) and for the parachute. At 3000 m, $rho$ is 26 % lower than at sea
level.

*Indicated vs true airspeed.* Instruments, placards and pilots use indicated
airspeed. Neglecting instrument error and compressibility, it equals the
equivalent airspeed:

$ V_"IAS" = V sqrt(rho(z) \/ rho_0), quad rho_0 = 1.225 thin "kg/m"^3. $ <eq-ias>

Stall and the aerodynamic loads depend on $macron(q)$, i.e. on IAS. The pilot's
target speed and the winch speed limit $V_W$ are therefore IAS values, while the
kinematics use TAS. Summaries report max/min IAS (against $V_W$ and stall) and max
TAS. On a hot, high airfield the glider needs more TAS, i.e. a longer ground
roll, for the same IAS.

Coefficients (the fuselage-referenced $C_(L 0)$ contains the wing incidence):

$
  C_L & = op("softclip")(C_(L 0) + C_(L alpha) alpha + C_(L delta) delta_e
          + C_(L q) hat(q), space C_(L,"min"), space C_(L,"max")) \
  C_D & = C_(D 0) + k C_L^2 + 1.3 sin^2 (alpha - alpha_"stall")_+,
          quad k = 1 / (pi e "AR") \
  C_m & = C_(m 0) + C_(m alpha) alpha + C_(m q) hat(q) + C_(m delta) delta_e
$ <eq-coefficients>

Lift acts perpendicular to $bold(v)_a$, drag against it:

$
  bold(F)_"aero" = macron(q) S (C_L hat(bold(e))_v^perp - C_D hat(bold(e))_v),
  quad hat(bold(e))_v = bold(v)_a \/ V,
  quad M_"aero" = macron(q) S macron(c) C_m.
$ <eq-aero-force>

The soft clip gives a lift plateau at $C_(L,"max")$ (stall) while staying
differentiable. Per-glider coefficients are generated from handbook data
(`gliders.make_glider`):

- $C_(D 0)$ from the best glide ratio: $(L\/D)_"max" = 1 \/ (2 sqrt(C_(D 0) k))$,
- $C_(L alpha)$ from the Helmbold formula for the aspect ratio, +8 % for the tail,
- $C_(m alpha) = -C_(L alpha) dot$ static margin (15 % $macron(c)$),
- $C_(m q)$, $C_(m delta)$, $C_(L delta)$, $C_(L q)$ from tail volume and tail arm,
- $C_(m 0)$ so the glider trims at best-glide $C_L$ with neutral elevator,
- $I_(y y) = m k_y^2$ with a radius of gyration of 1.1–1.4 m.

== Tow hook <sec-hook>

The winch ("CG") hook sits slightly ahead of and below the CG
($bold(b)_"hook" approx (0.25 "to" 0.35, -0.45)$ m). With cable tension $T$
pulling at angle $phi$ below the fuselage axis the hook moment is

$ M_"hook" = T (abs(b_z) cos phi - b_x sin phi), $ <eq-hook-moment>

nose-up while the cable is shallow (the well-known pitch-up tendency right after
lift-off) and nose-down once $tan phi > abs(b_z) \/ b_x$ in the steep part of the
climb, which is why the pilot holds more and more back stick as the launch
proceeds.

== Ground contact <sec-ground>

Nose skid, main wheel and tail wheel are points $bold(b)_k$ with a smooth
spring–damper that can only push, and regularised Coulomb friction:

$
  N_k = sigma(delta_k / (2 "mm")) op("softplus")(k_k delta_k - c_k dot(z)_k),
  quad F_(x, k) = -mu_k N_k tanh(dot(x)_k \/ 0.1)
$ <eq-ground>

with penetration $delta_k = -z_k$. This handles standing, the ground roll,
lift-off (and a touchdown, should it happen) with _one_ continuous vector field.
There is no mode switching, which keeps the ODE smooth and makes
`jit`/`vmap`/`grad` straightforward.

= The rope: lumped masses with elasticity, weight and drag <sec-rope>

Rope weight is large (steel $approx 0.08$ kg/m, i.e. about 100 kg of rope for
1200 m, heavier than the pilot), its drag at 30 m/s is hundreds of newtons, and
its stretch (1–2 % at working load) acts like a bungee. All three shape the
tension that actually reaches the hook, the cable angle at the hook, and the
oscillations during the ground run.

== Discretisation <sec-rope-disc>

The paid-out rope is split into $N$ segments with equal rest length
$ell_0 = L_0 \/ N$. Node 0 is fixed at the winch, nodes $1 .. N - 1$ are free
point masses $m_n = mu ell_0$, node $N$ is the hook (a point on the glider).

Reel-in is $dot(L)_0 = -v_r$. Because _all_ rest lengths and masses shrink
uniformly, the number of states stays fixed, which JAX needs. The price is a
small inconsistency in momentum bookkeeping (mass leaves every node instead of
only at the drum); it vanishes as $N -> oo$ and the test suite checks that the
release height converges with $N$ (12 vs 24 segments differ by less than 2 %).

The resolution is set in *segments per km* of laid-out rope
(`--segments-per-km`, default 10, i.e. 100 m segments, minimum 4 segments), so
short and long ropes are discretised alike. Too coarse a rope is badly wrong for
long ropes: 50 km of Dyneema gave about 3000 m release height at 20 segments but
2170 m at 500 segments (10/km).

== Segment tension <sec-tension>

For segment $j$ from node $a$ to node $b$: $bold(d) = bold(p)_b - bold(p)_a$,
$ell = abs(bold(d))$, $hat(bold(e)) = bold(d) \/ ell$,
$dot(ell) = (bold(u)_b - bold(u)_a) dot hat(bold(e))$.

$
  T_j = op("softplus")(E A (ell - ell_0) / ell_0
    + c (dot(ell) - ell dot(ell)_0 / ell_0)),
  quad c = 2 zeta sqrt(E A mu)
$ <eq-tension>

- softplus (width 2 N): a rope cannot push; a slack rope carries about 0 N.
- The damping term uses the _strain rate_, so reeling in does not by itself
  create damping forces. With $c = 2 zeta sqrt(E A mu)$ each segment's axial
  mode has damping ratio $zeta$ independent of $ell_0$.

== Aerodynamic drag (cross-flow principle) <sec-rope-drag>

With the segment's mean velocity relative to the wind
$bold(v)_"rel" = 1/2 (bold(u)_a + bold(u)_b) - bold(w)$, split into tangential
$bold(v)_t = (bold(v)_"rel" dot hat(bold(e))) hat(bold(e))$ and normal
$bold(v)_n = bold(v)_"rel" - bold(v)_t$ parts:

$
  bold(F)_("drag", j) = -1/2 rho d ell (C_(D n) abs(bold(v)_n) bold(v)_n
    + pi C_f abs(bold(v)_t) bold(v)_t)
$ <eq-rope-drag>

($C_(D n) approx 1.2$ for a cylinder, $C_f approx 0.01$–$0.02$). Half goes to
each end of the segment. Near the top of the launch the upper rope swings round
the winch at high speed, so this term grows as the launch progresses.

== Node equations <sec-nodes>

$
  m_n dot(bold(u))_i = T_(i+1) hat(bold(e))_(i+1) - T_i hat(bold(e))_i
  + 1/2 (bold(F)_("drag", i) + bold(F)_("drag", i+1))
  - m_n g hat(bold(z)) + bold(F)_("gnd", i),
  quad dot(bold(p))_i = bold(u)_i
$ <eq-nodes>

Ground contact of the rope lying on the grass uses the same spring–damper and
friction law as the wheels ($mu approx 0.4$–$0.5$), with the stiffness scaled
to the node mass. Dragging 100 kg of steel rope over grass costs about 500 N
during the ground run.

== Force on the glider <sec-hook-force>

$
  bold(F)_"hook" = -T_N hat(bold(e))_N + 1/2 bold(F)_("drag", N)
  - 1/2 rho (C_D A)_"chute" abs(bold(v)_(a,h)) bold(v)_(a,h)
  - m_"end" g hat(bold(z))
$ <eq-hook-force>

The *local* rope direction at the hook, $-hat(bold(e))_N$, is steeper than the
straight line to the winch because of sag (@fig-geometry); it is this angle that
triggers the back-release and that the pilot sees.

== Validation: elastic catenary <sec-catenary>

For a rope hanging still between two points, the exact solution is the elastic
catenary @irvine1981, parametrised by unstretched arc length $s$ with constant
horizontal tension $H$ and vertical tension $V(s) = V_0 + w s$:

$
  x(s) & = (H s) / (E A) + H / w [sinh^(-1) V / H - sinh^(-1) V_0 / H], \
  z(s) & = (w s^2 \/ 2 + V_0 s) / (E A)
         + H / w [sqrt(1 + V^2 / H^2) - sqrt(1 + V_0^2 / H^2)]
$ <eq-catenary>

`cable.elastic_catenary` solves for $V_0$ and the unstretched length, and the
test suite checks that the lumped rope ($N = 20$) settles onto it within 1 % of
the sag. A consequence visible in the launch results: in the climb the tension
at the hook exceeds the tension at the winch by roughly $mu g Delta z$ (about
300 N for steel at 400 m).

== Rope presets and rope input <sec-rope-input>

#figure(
  booktab(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    ([], [diameter], [$mu$], [$E A$], [breaking load], [$C_(D n)$], [$C_f$]),
    [steel wire], [4.5 mm], [0.080 kg/m], [1.0 MN], [17 kN], [1.2], [0.02],
    [Dyneema], [5.0 mm], [0.016 kg/m], [0.6 MN], [27 kN], [1.2], [0.01],
  ),
  caption: [Generic rope presets (approximate values).],
) <tab-ropes>

A specific rope can be given in three ways:

- `cable.rope_from_datasheet(diameter_mm, mass_kg_per_100m, breaking_load_kN,
  elongation_at_break_pct | EA_N, CDn, Cf, ground_mu, end_mass_kg, end_CdA_m2,
  ...)`. Without `EA_N`, the stiffness is the secant value
  $E A = F_"break" \/ epsilon_"break"$. Synthetic ropes are stiffer at working
  loads, so give `EA_N` from the load–elongation curve at 20–30 % of the breaking
  load when you have it.
- A TOML file with the same keys (`--rope-file ropes/example_dyneema_6mm.toml`),
  or `base = "steel"` plus SI overrides of `Rope` fields
  (`ropes/example_steel_override.toml`).
- Single-field SI overrides on the command line: `--rope-param mu=0.09`.

The breaking load does not affect the dynamics. It is used to report the rope
safety factor $F_"break" \/ max T$ in the summary table.

== Long ropes: why the hook tension does not grow <sec-long-ropes>

Rope weight enters the tension balance through height, not length. Along a
hanging rope, $d T \/ d z = mu g$ (plus drag), so the hook feels at most the
tension at the rope's lowest point plus $mu g Delta z$ ($0.16 "N/m" times 3000
"m" approx 0.5$ kN for Dyneema). The remaining rope weight rests on the winch
end, the sag, or the ground. What a long rope does is *eat* tension between winch
and hook: friction of the part still lying on the grass ($mu_g m_"rope" g$) and
drag of the long airborne part (@tab-long-ropes).

#figure(
  booktab(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    (
      [rope],
      [length],
      [rope mass],
      [release height],
      [max $T$ winch],
      [max $T$ hook],
      [on ground#super[a]],
    ),
    [Dyneema], [1.2 km], [19 kg], [450 m], [6.2 kN], [6.1 kN], [0 %],
    [Dyneema], [10 km], [160 kg], [2353 m], [5.6 kN], [4.8 kN], [5 %],
    [Dyneema], [50 km], [800 kg], [2169 m], [5.6 kN], [2.9 kN], [81 %],
    [steel], [10 km], [800 kg], [1500 m], [8.6 kN], [6.0 kN], [29 %],
    [steel], [50 km], [4000 kg], [no lift-off], [6.9 kN], [0.2 kN], [100 %],
  ),
  caption: [Long ropes, ASK 21, engine winch, 10 segments/km.
    #super[a]Share of the rope on the ground at 60 % of the release height.
    Computed with the earlier end-of-launch model (@note-old-model).],
) <tab-long-ropes>

Beyond some length, extra rope only adds drag and friction: the 50 km Dyneema
launch goes _lower_ than the 10 km one. The 50 km steel rope (4 t, about 20 kN
of friction on grass) cannot be dragged by this winch at all.

= Winch <sec-winch>

Both winch models share one equation for the drum, lumped as an effective mass
$M_"eff" = J \/ r_d^2$ at the rope:

$
  M_"eff" dot(v)_r = tau(t, beta) F_"avail" (v_r) - T_1 - b v_r,
  quad F_"avail" (v) = P_"max" / sqrt(v^2 + (P_"max" \/ F_"max")^2)
$ <eq-winch>

$F_"avail"$ is a smooth $min(F_"max", P_"max" \/ v)$:

- *tension-controlled winch*: $P_"max" -> oo$, so the winch pulls with a
  prescribed tension $F_"max" = k_T m g$ (default $k_T = 1.1$). This isolates the
  glider's behaviour.
- *engine winch*: finite power (default 200 kW, 12 kN), so the pull _drops_ as
  the reel speeds up and _rises_ when the glider slows the rope down (e.g. during
  rotation), a real mechanism behind weak-link failures of light gliders.

The driver's throttle $tau$ ramps up over 3 s after $t = 1$ s. At the top the
driver *throttles right down* (to `fade_floor` = 0) as the glider elevation seen
from the winch, $beta = op("atan2")(z - z_w, x_w - x)$, goes from 65° to 70°. The
drum stops pulling, the rope slackens and its weight (plus parachute) drags the
hook end down and back until the back-release trips (@sec-events). For the
engine winch the driver sets the throttle so that the pull at 15 m/s reel speed
is $k_T m g$. After release the drum is braked.

= Pilot <sec-pilot>

The pilot flies pitch attitude. The reference is a smooth function of the state
(again no discrete phases):

$
  theta_"ref" = (1 - s_"top") [theta_g + s_"rot" (h) (theta_"climb" - theta_g)
    + K_V op("clip")(V_"IAS" - V_"target")] + s_"top" (beta) theta_"top"
$ <eq-pilot-ref>

- ground roll: rest attitude + 3° of back stick ($theta_g$),
- rotation into the climb between 2 and 30 m height ($s_"rot"$, smoothstep),
- climb attitude 35–40°, raised when faster than the target speed (IAS),
  lowered when slower,
- nose lowered to 5° as $beta$ goes from 55° to 70° ($s_"top"$),
- after release: glide attitude 0°.

Elevator: PID with saturation and a first-order stick lag,

$
  delta_(e,"cmd") = delta_"max" tanh((-(K_p e + K_i e_I) + K_d q) / delta_"max"),
  quad dot(delta)_e = (delta_(e,"cmd") - delta_e) / tau_p,
  quad dot(e)_I = a(h) e - (1 - a(h)) e_I
$ <eq-elevator>

with $e = theta_"ref" - theta$ and $a(h)$ a smooth "airborne" factor that stops
integrator wind-up while the tail is on the ground.

= End of the launch: events <sec-events>

The launch ends at the first of the events in @tab-events.

#figure(
  booktab(
    columns: (auto, 1fr),
    ([event], [condition]),
    [`back_release`],
    [cable more than 110° below the fuselage axis: the hook's back-release trips.
      *The normal end of a launch*: after the winch throttles down the rope goes
      slack, the glider flies on over the rope end and the cable pulls from
      behind.],

    [`release`],
    [pilot release at a local cable angle `Pilot.release_angle`. Off by default
      (180°), since pilots cannot judge the cable angle; set e.g. 72–85° to model
      a pilot who releases early.],

    [`weak_link`], [hook tension exceeds the glider's weak-link rating.],
    [`rope_in`], [less than 30 m of rope left out.],
    [`no_liftoff`],
    [still on the ground after 120 s (e.g. rope too heavy to drag); a failed
      launch, no sensitivities.],
  ),
  caption: [End-of-launch events.],
) <tab-events>

= Solving it with diffrax <sec-solver>

/ Pure-JAX vector field: `dynamics.vector_field(t, y, args)`. `y` is an
  `equinox.Module` pytree (`State`), `args` holds all parameters as
  `equinox.Module`s with float leaves, so the whole solve can be `jit`-compiled,
  `vmap`-ed over gliders, ropes or winch settings, and differentiated (e.g.
  release height w.r.t. winch tension).
/ Smoothness by construction: Every kink (rope slack, ground contact, friction,
  stall, pilot phases) is replaced by softplus, sigmoid, tanh or smoothstep, so
  an adaptive explicit solver never has to stop at a discontinuity.
/ Solver: `Tsit5` with a `PIDController(rtol=atol=1e-6)`. The stiffest mode is
  the axial rope wave, $omega approx sqrt(E A \/ mu) \/ ell_0$ (about 50 rad/s
  at the start, a few hundred near the top), which is resolved in 2–3k steps.
  `Kvaerno5` (implicit) is available via `solver="kvaerno5"`; it gives the same
  answer but takes more steps, i.e. the problem is not stiff enough to benefit.
/ Events: `diffrax.Event` with the condition functions of @tab-events (sign
  change, `direction=True`) and an `optimistix.Newton` root finder to locate the
  release instant exactly. `sol.event_mask` says which one fired.
/ Saving (two passes): Pass 1 runs to the event with `SaveAt(t1=True)` only, so
  the time limit `t_max` (default $10^5$ s, `--t-max`) costs nothing, and
  `max_steps` is $5 dot 10^6$. Pass 2 repeats the deterministic solve from 0 to
  the release time and saves 2000 evenly spaced points for plotting and
  diagnostics. Long and short launches get the same number of plot points.
/ Two stages: A second `diffeqsolve` from the release state with `attached = 0`
  continues the glider in free flight (push-over, pick-up of speed) while the
  rope falls.
/ Derived quantities: Tensions, angles, load factor and power are recomputed
  from saved states with the same `evaluate` function, `vmap`-ed over time, so
  plots can never disagree with the dynamics.
/ Gradients: `sensitivity.release_height` is the same stage-1 solve with
  `SaveAt(t1=True)` and `RecursiveCheckpointAdjoint`.
  `eqx.filter_value_and_grad` over the whole `Launch` pytree gives every
  parameter's sensitivity at once (@sec-sensitivity).
/ Batching: `simulate.solve_batch(launches)` stacks the parameter pytrees and
  `vmap`s the solve: all catalogue gliders run as one compiled program in about
  0.5 s after a few seconds of compilation.

= Example results <sec-results>

1200 m rope, no wind, winch pull 1.1 × glider weight. Speeds are IAS; bold marks
a peak speed above the glider's placarded winch speed $V_W$. Reproduce with

```sh
uv run main.py [--rope steel] [--winch engine]
```

#figure(
  text(8.5pt, booktab(
    columns: 11,
    align: (left, right, right, left, left, right, right, right, right, right, right),
    (
      [glider],
      [mass],
      [$V_W$],
      [rope],
      [winch],
      [$h_"rel"$],
      [time],
      [roll],
      [max $V$],
      [max $T_"hook"$],
      [max $L\/W$],
    ),
    [Ka 8], [290 kg], [100 km/h], [Dyneema], [tension], [454 m], [50 s], [46 m], [*128 km/h*], [3.0 kN], [2.6],
    [ASK 13], [460 kg], [120 km/h], [Dyneema], [tension], [470 m], [43 s], [52 m], [*149 km/h*], [5.0 kN], [2.8],
    [LS4], [360 kg], [130 km/h], [Dyneema], [tension], [456 m], [42 s], [69 m], [*150 km/h*], [4.1 kN], [3.0],
    [ASK 21], [470 kg], [150 km/h], [Dyneema], [tension], [481 m], [45 s], [52 m], [*151 km/h*], [5.4 kN], [3.2],
    [ASG 29], [420 kg], [150 km/h], [Dyneema], [tension], [465 m], [39 s], [75 m], [*157 km/h*], [4.8 kN], [3.0],
    [DG-1000], [620 kg], [150 km/h], [Dyneema], [tension], [493 m], [40 s], [64 m], [*165 km/h*], [7.0 kN], [3.3],
    [Ka 8], [], [], [steel], [tension], [434 m], [50 s], [57 m], [121 km/h], [3.3 kN], [2.5],
    [ASK 13], [], [], [steel], [tension], [459 m], [44 s], [58 m], [*144 km/h*], [5.2 kN], [2.7],
    [ASK 21], [], [], [steel], [tension], [468 m], [45 s], [59 m], [146 km/h], [5.7 kN], [3.1],
    [DG-1000], [], [], [steel], [tension], [484 m], [40 s], [70 m], [161 km/h], [7.3 kN], [3.2],
    [Ka 8], [], [], [Dyneema], [engine], [444 m], [51 s], [70 m], [107 km/h], [3.7 kN], [2.2],
    [ASK 13], [], [], [Dyneema], [engine], [441 m], [45 s], [69 m], [119 km/h], [5.9 kN], [2.3],
    [ASK 21], [], [], [Dyneema], [engine], [452 m], [46 s], [65 m], [122 km/h], [6.0 kN], [2.6],
    [DG-1000], [], [], [Dyneema], [engine], [449 m], [42 s], [81 m], [132 km/h], [7.9 kN], [2.5],
  )),
  caption: [Launch results across the catalogue. Computed with the earlier
    end-of-launch model (@note-old-model).],
) <tab-results>

Observations:

- Release heights of about 40 % of the rope length, 40–50 s launches and climb
  speeds of #box[90–110 km/h] are in line with typical club winch launches.
- Steel costs 10–20 m of height and lengthens the ground run versus Dyneema.
- A constant tension (tension winch) accelerates the glider hard in the ground
  run and initial climb; the speed peak just before full rotation exceeds the
  placarded winch speed $V_W$ for every glider (barely for the ASK 21, by 28 %
  for the Ka 8). A real driver would use a lower setting for the Ka 8 and ramp
  more gently. The power-limited engine winch is self-limiting (pull falls with
  reel speed): lower peak speeds and load factors, at the cost of 20–50 m of
  release height.
- The peak load factor occurs during the rotation, which the simple pilot flies
  somewhat aggressively.

= Sensitivities of the release height <sec-sensitivity>

Every `uv run main.py` run (unless `--no-sensitivity`; `--only rope` filters the
table) computes $partial h \/ partial p$ for all model parameters (glider, rope,
winch, pilot, environment, layout) in one reverse-mode pass of `jax.grad` through
the diffrax solve (`sensitivity.py`, `RecursiveCheckpointAdjoint`). The release
time is found by the event root finder, and its dependence on the parameters is
differentiated implicitly. The gradients agree with central finite differences
to within 0.5 % (`tests/test_sensitivity.py`). Tiny sensitivities are only as
accurate as the solver tolerance, so the default is rtol = atol = $10^(-8)$. A
run takes about 10 s.

For each parameter the table gives its value and unit, $partial h \/ partial p$
in m per unit (angles per degree), the height change for a +10 % change, and the
elasticity $(p \/ h) thin partial h \/ partial p$ (% height per % parameter).
Everything else is held fixed: e.g. the winch pull, preset as 1.1 × weight, does
not follow a change of glider mass. These are local linearisations. The full
table goes to `results/sensitivity_<glider>_<rope>_<winch>.csv`.

#figure(
  booktab(
    columns: 5,
    align: (left, right, right, right, right),
    ([parameter], [value], [$partial h \/ partial p$], [$Delta h$ for +10 %], [elasticity]),
    [rope length], [1200 m], [0.380 m/m], [+45.6 m], [0.95],
    [winch pull `F_max`], [5072 N], [0.0488 m/N], [+24.7 m], [0.51],
    [glider mass], [470 kg], [−0.464 m/kg], [−21.8 m], [−0.45],
    [winch fade start `fade_beta0`], [50°], [0.75 m/deg], [+3.8 m], [0.08],
    [wing area], [17.95 m²], [1.92 m/m²], [+3.4 m], [0.07],
    [rope diameter], [5 mm], [−6.6 m/mm], [−3.3 m], [−0.07],
    [rope normal drag coefficient `CDn`], [1.2], [−27 m], [−3.2 m], [−0.07],
    [pilot target speed], [105 km/h], [−1.08 m/(m/s)], [−3.1 m], [−0.07],
    [release cable angle], [72°], [0.41 m/deg], [+3.0 m], [0.06],
    [climb attitude], [40°], [0.63 m/deg], [+2.5 m], [0.05],
  ),
  caption: [Largest sensitivities, ASK 21, 1200 m Dyneema, tension winch
    ($h = 481$ m). Computed with the earlier end-of-launch model
    (@note-old-model).],
) <tab-sensitivity>

Rope characteristics for the steel rope and the ASK 13 (`--only rope`): diameter
−2.8 m, `CDn` −2.7 m, mass per length −1.7 m, parachute drag area −0.4 m, ground
friction −0.4 m per +10 %. The axial stiffness $E A$ has essentially *no*
influence on release height (less than 1 cm for +10 %). Rope stretch shapes the
tension oscillations in the ground run (@sec-rope), not the energy that reaches
the glider. Rope drag (diameter × `CDn`) matters about twice as much as rope
weight.

#figure(kind: "note", supplement: [Note], note[
  *Note: earlier end-of-launch model.*
  @tab-long-ropes, @tab-results and @tab-sensitivity were computed with an
  earlier end-of-launch model: the driver faded the winch to 40 % between 50°
  and 70° elevation, and the pilot released at 72° local cable angle. The current
  default (throttle down to idle at 65–70°, launch ended by the back-release)
  gives release heights within about ±10 m of those values (catalogue, 1200 m
  Dyneema, tension winch: Ka 8 462 m, ASK 13 478 m, LS4 459 m, ASK 21 491 m,
  ASG 29 467 m, DG-1000 496 m). The back-release trips 1–7 s after the
  throttle-down, at 0.1–0.5 kN hook tension and 64–105 km/h IAS.

  For reference, a sweep of the (optional) pilot release angle with the old
  winch fade, ASK 21 on 1200 m Dyneema, tension winch:

  #align(center, booktab(
    columns: 6,
    align: (left, right, right, right, right, right),
    ([release angle], [72°], [76°], [80°], [85°], [95°]),
    [release height], [480 m], [481 m], [479 m], [477 m], [473 m],
    [launch time], [44.2 s], [45.7 s], [47.1 s], [48.8 s], [52.3 s],
    [IAS at release], [74 km/h], [77 km/h], [82 km/h], [86 km/h], [86 km/h],
  ))

  Release height hardly depends on when exactly the launch ends near the top.
  With the old 40 % power floor the rope stayed taut, so the back-release never
  tripped; with a real throttle-down it does.
]) <note-old-model>

= Limitations and possible extensions <sec-limits>

- Pilot and winch driver are simple smooth control laws, not human models. The
  launch ends by the hook's back-release after the driver throttles down on a
  visual cue (glider elevation). A pilot release when the pull fades (a tension
  cue) would be an alternative. Their parameters (rotation height, climb
  attitude, tension profile) dominate the results as much as the glider's
  aerodynamics do, just as in real life.
- The glider data are approximate estimates, not manufacturer data.
- No ground effect, no flaps (flap settings matter for the ASG 29/LS-type
  gliders), no wing drop or lateral dynamics, no cable parachute opening after
  release.
- Rope reel-in removes mass uniformly from all nodes (@sec-rope-disc). A
  "moving-node" formulation that removes mass only at the drum would be exact.
- Winch drum physics is intentionally minimal (one effective mass).
- The sensitivities (@sec-sensitivity) are local. For large changes (e.g. steel
  to Dyneema) run the simulation directly. The same gradients could also drive an
  optimisation of the tension profile or pilot schedule, subject to
  $V <= V_W$, $T <=$ weak link and $n <= n_"max"$ @aq1977.

= Related literature <sec-literature>

*Winch-launch simulation.* Gäb and Santel @gab2011 (RWTH Aachen,
Matlab/Simulink) model aircraft, pilot, winch, winch operator, cable, atmosphere
and terrain. Their cable is mass points joined by spring–damper links with drag,
weight and ground reaction, the same approach as @sec-rope here; pilot and winch
operator are PID controllers with reaction time and neuromuscular delay. They
analyse a reference launch, wind, and overly steep initial climbs. *It is the
best reference to validate this model against.* Its origin is an RWTH study
thesis @rwth2008 (#link("https://core.ac.uk/download/pdf/36588266.pdf")[CORE
PDF]), which found that opening the throttle quickly destabilises the phugoid,
that the tow-hook position can be optimised, and that release height depends on
cable type, tow distance, maximum winch force and especially cable drag; it lists
ground effect and ground/cable friction as missing. Santel's diploma thesis
@santel investigates winch launch accidents with multipoint aerodynamics. Bogan's
simpler model @bogan (prescribed flight path, MathCAD, not peer-reviewed) gives a
good intuition for cable drag growing as the rope turns broadside.

*Optimal launch trajectories.* With limits on $C_L$ and cable acceleration, the
optimal trajectory is climb–dive–climb; a reel-in speed limit makes it more
realistic @aq1977. This is the benchmark for a `jax.grad`-based optimisation
(@sec-limits). See also Eppler on the optimal release height @eppler and the
follow-up work of Pierson and Chen @pierson.

*Safety and winch engineering.* Browning @browning2007 and Hills @hills2007 on
safe winch launching; design formulae for winch hardware @winchdesign; the
Soaring Safety Foundation notes by Smith @smith
(#link("https://www.soaringsafety.org/publications/winches.pdf")[Winches]).

*Cable and tether modelling.* Irvine's elastic catenary @irvine1981 is used for
validation in @sec-catenary. Variable-length tether modelling @williams2007
relates to the reel-in approach of @sec-rope-disc; the SkySails model
@skysails does optimal control with tether force and reel-out speed as
variables.

*How this model compares with @gab2011.* The structure is the same
(lumped-mass cable, PID pilot, tension-controlled winch operator). Differences:
@gab2011 includes human reaction and neuromuscular delays and a more detailed
winch/operator model. This model uses a fixed-size reel-in discretisation,
smooth contact models (one continuous ODE instead of mode switching), and
JAX/diffrax for batching and gradients. Obvious next steps: reproduce the
reference launch of @gab2011, and add reaction delays to the pilot and winch
driver.

#bibliography("refs.yml", style: "ieee")
