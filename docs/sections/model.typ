#import "../template.typ": *

// Model: conventions, glider, rope, winch.

= Frames, state and conventions <sec-frames>

The whole launch happens in one vertical plane, along the runway. We assume
that nothing moves sideways: there is no crosswind, no roll and no yaw. The
runway is flat and level.

/ Earth frame: $x$ is horizontal and points from the glider's start towards the
  winch; $z$ points up. The runway is $z = 0$. The rope leaves the winch at
  $bold(r)_w = (x_w, z_w)$, where $z_w = 1$ m (`Winch.height`) and $x_w$ is the
  hook's start position plus the laid-out rope length (`Launch.rope_length`).
/ Body frame: $x_b$ points forward along the fuselage, $z_b$ points up towards
  the canopy. In earth coordinates the body axes are
  $hat(bold(x))_b = (cos theta, sin theta)$ and
  $hat(bold(z))_b = (-sin theta, cos theta)$.
/ Signs: The pitch attitude $theta$, the pitch rate $q$ and the pitching moment
  $M$ are positive nose-up. In the plane, a force $bold(F)$ that acts at offset
  $bold(r)$ from the centre of gravity (CG) gives the moment
  $M = bold(r) times bold(F) = r_x F_z - r_z F_x$.
/ Points on the glider: A point with body coordinates $bold(b)$ (hook, wheel,
  skid) sits at $bold(r)_"cg" + bold(r)$ with $bold(r) = R(theta) bold(b)$, the
  body vector rotated by $theta$. It moves with velocity
  $bold(v) + q bold(r)^perp$, where $bold(r)^perp = (-r_z, r_x)$ is $bold(r)$
  turned by $+90°$.

*Smooth functions.* Many things in a launch switch on or off: a rope goes
slack, a wheel leaves the ground, the wing stalls. Instead of sharp switches the
model uses smooth versions, so the equations have no kinks (@sec-solver). They
are:

- $op("softplus")_s (x) = s ln(1 + e^(x \/ s))$, a rounded $max(x, 0)$; the
  width $s$ sets how sharp the corner is;
- $op("softclip")(x, a, b) = a + op("softplus")_s (x - a) - op("softplus")_s (x - b)$,
  a rounded clip of $x$ to $[a, b]$;
- the logistic function $sigma(x) = 1 \/ (1 + e^(-x))$, a smooth step from 0
  to 1;
- the smoothstep $S(x) = 3 x^2 - 2 x^3$ for $0 <= x <= 1$ (0 below, 1 above),
  used for gradual changes such as the throttle ramp.

#figure(
  booktab(
    columns: 2,
    ([symbol], [meaning]),
    [$bold(r) = (x, z)$, $bold(v)$], [glider CG position and velocity (earth frame)],
    [$theta$, $q$], [pitch attitude and pitch rate],
    [$bold(p)_i$, $bold(u)_i$, $i = 1, ..., N - 1$], [positions and velocities of the free rope nodes],
    [$L_0$], [unstretched length of the rope still paid out],
    [$v_r$], [reel-in speed at the drum],
    [$delta_e$], [elevator deflection],
    [$e_I$], [time integral of the pilot's attitude error],
  ),
  caption: [State vector $y$ (SI units).],
) <tab-state>

With $N$ rope segments the state has $6 + 4 (N - 1) + 4$ entries. For the
default 1200 m rope ($N = 12$) that is 54. We use earth-frame velocities for
the glider instead of the more usual body-frame velocities. For motion in a
plane the two are equivalent, and the earth frame makes the coupling to the rope
nodes simpler.

*Initial state.* The glider stands still on its main wheel and tail skid, tail
down. The rope lies straight on the runway between the hook and the winch. Its
unstretched length is the distance between the two ends plus 0.2 % slack
(`Launch.slack` = 0.002), so at $t = 0$ it carries no tension.

= Glider rigid body <sec-glider>

The glider is pulled by the rope, lifted and slowed by the air, pulled down by
gravity and, at first, held up by the ground. These forces set how fast it
accelerates and climbs. Where they act relative to the CG sets how it pitches.

*Assumptions.* The glider is a rigid body of constant mass; the wings and
fuselage do not bend. Only the longitudinal motion is modelled (three degrees
of freedom: $x$, $z$, $theta$). The rope end assembly at the hook (parachute,
strop, weak link) moves with the glider: its mass is added to the glider's mass
for translation, but not to the moment of inertia. Its weight and the parachute
drag act at the hook, so they do pitch the glider.

Newton's laws for translation and rotation read

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

with

- $m$: launch mass of the glider (with pilots); $g = 9.81 thin "m/s"^2$
  (`Env.g`; the ISA formulas below use the standard $g_0 = 9.80665 thin "m/s"^2$);
  $hat(bold(z)) = (0, 1)$;
- $m_"tot" = m + m_"end"$: the glider plus the rope mass carried at the hook,
  $m_"end" = m_"assy" + 1/2 m_n$ (end assembly of 8 kg plus half a rope
  segment, @sec-hook-force). The weight of $m_"end"$ is part of
  $bold(F)_"hook"$; after release $m_"end" = 0$;
- $I_(y y)$: pitch moment of inertia about the CG;
- $bold(F)_"aero"$, $M_"aero"$: aerodynamic force and pitching moment
  (@sec-aero);
- $bold(F)_("gnd", k)$: force of the ground on contact point $k$ at offset
  $bold(r)_k$ (@sec-ground);
- $bold(F)_"hook"$: force of the rope on the hook at offset $bold(r)_"hook"$
  (@sec-hook, @sec-hook-force).

== Aerodynamics <sec-aero>

The wing needs airspeed to lift. The glider's speed through the air, and the
angle at which the air meets the wing, set the lift, the drag and the pitching
moment.

*Wind.* A headwind shortens the ground run. It grows with height as a power law,

$ bold(w)(z) = (-W_10 (z_c \/ 10 thin "m")^p, 0), quad z_c = op("softplus")_(0.5 thin "m") (z) + 0.1 thin "m", $

where $W_10$ is the headwind at 10 m (`Env.wind_ref`, default 0) and $p = 1\/7$
(`Env.wind_exp`). A headwind blows towards $-x$. The height $z_c$ is kept just
above zero so the power law stays finite on the ground. The glider feels the
wind at its CG height.

*Air data.* The air-relative velocity is $bold(v)_a = bold(v) - bold(w)(z)$.
The true airspeed (TAS) is $V = abs(bold(v)_a)$. With $u = bold(v)_a dot
hat(bold(x))_b$ and $w_b = bold(v)_a dot hat(bold(z))_b$ the airflow components
along the body axes,

$
  alpha = op("atan2")(-w_b, u),
  quad macron(q) = 1/2 rho(z) V^2,
  quad hat(q) = (q macron(c)) / (2 sqrt(V^2 + 1 thin "m"^2"/s"^2)),
$ <m-eq-airdata>

where $alpha$ is the angle of attack of the fuselage axis, $macron(q)$ the
dynamic pressure, $hat(q)$ the non-dimensional pitch rate and $macron(c)$ the
mean chord. The $1 thin "m"^2"/s"^2$ under the root keeps $hat(q)$ finite at rest.

*Air density.* Thinner air gives less lift at the same true airspeed. The
density follows the International Standard Atmosphere (ISA, `atmosphere.py`)
at the pressure height $h = h_"field" + z$, with a temperature offset $Delta T$
for hot or cold days (`Env.field_elevation`, `Env.isa_dT`; CLI
`--field-elevation`, `--isa-dt`):

$
  T = T_0 - lambda h + Delta T,
  quad p = p_0 ((T_0 - lambda h) / T_0)^(g_0 \/ (R lambda)),
  quad rho = p / (R T)
$ <eq-isa>

with $T_0 = 288.15$ K, $p_0 = 101325$ Pa, lapse rate $lambda = 6.5$ K/km,
$g_0 = 9.80665 thin "m/s"^2$ and gas constant $R = 287.05 thin "J/(kg K)"$. Above 11 km
the temperature is constant. The offset $Delta T$ changes the temperature and
so the density, but not the pressure. The same $rho(z)$ is used for the glider,
for every rope segment (at its mid-height) and for the parachute. At 3000 m the
density is 26 % lower than at sea level.

*Indicated and true airspeed.* Pilots, instruments and placards use the
indicated airspeed (IAS). We neglect instrument error and compressibility (fine
below about 300 km/h). Then IAS equals the equivalent airspeed, the speed that
gives the same dynamic pressure at sea-level density:

$ V_"IAS" = V sqrt(rho(z) \/ rho_0), quad rho_0 = 1.225 thin "kg/m"^3. $ <eq-ias>

Stall and the aerodynamic loads depend on $macron(q)$, and so on IAS. The pilot's
target speed and the placarded maximum winch-launch speed $V_W$ are therefore
IAS values, while the motion itself uses TAS. The summaries report maximum and
minimum IAS (to compare with $V_W$ and the stall speed) and maximum TAS. On a
hot, high airfield the glider needs more TAS for the same IAS, and so a longer
ground run.

*Aerodynamic coefficients.* The lift coefficient grows linearly with the angle
of attack until it levels off at stall. Drag has a constant part and a part that
grows with lift (induced drag); past the stall it rises steeply. The pitching
moment keeps the glider stable and is changed with the elevator:

$
  C_L & = op("softclip")(C_(L 0) + C_(L alpha) alpha + C_(L delta) delta_e
          + C_(L q) hat(q), space C_(L,"min"), space C_(L,"max")) \
  C_D & = C_(D 0) + k C_L^2 + 1.3 sin^2 (min(Delta alpha, 1.5)),
          quad k = 1 / (pi e "AR") \
  C_m & = C_(m 0) + C_(m alpha) alpha_m + C_(m q) hat(q) + C_(m delta) delta_e
$ <eq-coefficients>

- $C_(L 0)$, $C_(L alpha)$: lift at zero angle of attack and lift slope. The
  angle $alpha$ is measured from the fuselage axis, so $C_(L 0)$ contains the
  wing's incidence.
- $C_(L delta)$, $C_(L q)$: lift from elevator deflection and pitch rate.
- $C_(L,"min")$, $C_(L,"max")$: negative and positive stall limits. The soft
  clip (width 0.05) gives a flat top in the lift curve at stall while staying
  differentiable.
- $C_(D 0)$: zero-lift drag. $k C_L^2$ is induced drag, with Oswald factor
  $e = 0.85$ and aspect ratio $"AR" = b^2 \/ S$.
- $Delta alpha$: how far $alpha$ is past the stall, in either direction,
  $Delta alpha = op("softplus")_(0.02)(alpha - alpha_s)
  + op("softplus")_(0.02)(alpha_n - alpha)$ with
  $alpha_s = (C_(L,"max") - C_(L 0)) \/ C_(L alpha)$ and
  $alpha_n = (C_(L,"min") - C_(L 0)) \/ C_(L alpha)$. The post-stall term
  $1.3 sin^2$ is flat-plate drag; it is capped at $Delta alpha = 1.5$ rad.
- $C_(m 0)$, $C_(m alpha)$, $C_(m q)$, $C_(m delta)$: pitching-moment
  coefficients about the CG. The moment uses
  $alpha_m = op("softclip")(alpha, -0.4, 0.4)$ (about $plus.minus 23°$), so the
  moment curve goes flat far beyond the stall.
- $delta_e$: elevator deflection. Negative $delta_e$ (trailing edge up)
  pitches the nose up, since $C_(m delta) < 0$.

Lift acts perpendicular to the airflow, drag along it:

$
  bold(F)_"aero" = macron(q) S (C_L hat(bold(e))_v^perp - C_D hat(bold(e))_v),
  quad hat(bold(e))_v = bold(v)_a \/ V,
  quad M_"aero" = macron(q) S macron(c) C_m,
$ <eq-aero-force>

where $S$ is the wing area and $hat(bold(e))_v^perp$ is $hat(bold(e))_v$ turned
by $+90°$. There is no ground effect.

*Coefficients from handbook data.* Flight manuals give mass, span, wing area,
best glide ratio and the placarded winch speed, but not the stability
derivatives. `gliders.make_glider` fills these in with standard estimates:

- mean chord $macron(c) = S \/ b$;
- $C_(D 0)$ from the best glide ratio: $(L\/D)_"max" = 1 \/ (2 sqrt(C_(D 0) k))$;
- $C_(L alpha)$ from the Helmbold formula
  $2 pi "AR" \/ (2 + sqrt("AR"^2 + 4))$, plus 8 % for the tailplane;
- $C_(L 0) = 0.35$ and $C_(L,"min") = -0.6$ for all gliders;
- $C_(m alpha) = -C_(L alpha) dot 0.15$ (static margin 15 % of $macron(c)$);
- $C_(m q)$, $C_(m delta)$, $C_(L delta)$, $C_(L q)$ from a tailplane with
  lift slope 4/rad, tail volume 0.5, tail arm $l_t = 4.5$ m and elevator
  effectiveness 0.5: $C_(m q) = -3.6 thin l_t \/ macron(c)$,
  $C_(m delta) = -1.0$, $C_(L delta) = macron(c) \/ l_t$, $C_(L q) = 3.6$;
- $C_(m 0)$ so the glider trims at its best-glide lift coefficient
  $sqrt(C_(D 0) \/ k)$ with neutral elevator;
- $I_(y y) = m k_y^2$ with a radius of gyration $k_y$ of 1.1–1.4 m.

@m-tab-gliders lists the catalogue. The values are approximate and meant to
show the differences between classes of gliders, not to certify anything.

#figure(
  booktab(
    columns: 9,
    align: (left, right, right, right, right, right, right, right, right),
    ([glider], [$m$], [$b$], [$S$], [$(L\/D)_"max"$], [$C_(L,"max")$], [$V_W$], [weak link], [$k_y$]),
    [Ka 8], [290 kg], [15 m], [14.15 m²], [27], [1.30], [100 km/h], [5.0 kN], [1.10 m],
    [ASK 13], [460 kg], [16 m], [17.50 m²], [27], [1.40], [120 km/h], [10.0 kN], [1.30 m],
    [LS4], [360 kg], [15 m], [10.50 m²], [40], [1.40], [130 km/h], [6.0 kN], [1.15 m],
    [ASK 21], [470 kg], [17 m], [17.95 m²], [34], [1.35], [150 km/h], [10.0 kN], [1.35 m],
    [ASG 29], [420 kg], [18 m], [10.50 m²], [52], [1.45], [150 km/h], [8.5 kN], [1.20 m],
    [DG-1000], [620 kg], [20 m], [17.53 m²], [46], [1.40], [150 km/h], [10.0 kN], [1.40 m],
  ),
  caption: [Glider catalogue (`gliders.CATALOGUE`, approximate data): launch
    mass, span, wing area, best glide ratio, maximum lift coefficient,
    placarded winch speed (IAS), weak-link rating and radius of gyration.],
) <m-tab-gliders>

== Tow hook <sec-hook>

The rope is attached to the winch hook (often called the CG hook). It sits a
little ahead of and below the CG, at body coordinates
$bold(b)_"hook" = (b_x, b_z) = (0.25, -0.45)$ m (0.35 m forward for the
two-seaters ASK 13, ASK 21 and DG-1000). Because the hook is not at the CG, the
rope pull also pitches the glider. With rope tension $T$ pulling at angle
$phi$ below the fuselage axis, the moment of the rope pull about the CG is

$ M_"hook" = T (abs(b_z) cos phi - b_x sin phi). $ <eq-hook-moment>

While the rope is shallow ($phi$ small) this moment is nose-up. This is the
well-known pitch-up tendency right after lift-off. Once
$tan phi > abs(b_z) \/ b_x$ (above about 52° to 61° here), in the steep part
of the climb, the moment turns nose-down. So the pilot has to hold more and more
back stick as the launch goes on.

The hook also has a *back-release*: it opens by itself when the rope pulls
from far behind, more than 110° below the fuselage axis. This ends the launch
(@sec-events).

== Ground contact <sec-ground>

Before lift-off the glider rests on the ground. It touches it at three points:
the nose skid, the main wheel and the tail wheel or skid. The ground carries the
weight and the wheels and skids add friction, which the rope has to overcome
during the ground run.

*Assumptions.* Each contact point is a spring with a damper that can only push,
not pull. Friction follows Coulomb's law (friction force proportional to the
normal force), rounded near zero speed:

$
  N_k = sigma(delta_k / (2 thin "mm")) op("softplus")_(1 thin "N")(k_k delta_k - c_k dot(z)_k),
  quad F_(x, k) = -mu_k N_k tanh(dot(x)_k / (0.1 thin "m/s"))
$ <eq-ground>

- $delta_k = -z_k$: how far point $k$ is below the runway;
- $dot(x)_k$, $dot(z)_k$: velocity of the point;
- $k_k$, $c_k$: spring stiffness and damping. Skids: $4 dot 10^4$ N/m and
  $2000$ N s/m; main wheel: $25 m g$ per metre and $20 m$ N s/m;
- $mu_k$: friction coefficient, 0.4 (nose skid), 0.05 (main wheel, rolling),
  0.3 (tail);
- the logistic factor $sigma$ fades the force out within a few millimetres
  above the ground.

The contact points sit at body coordinates $(1.8, -0.55)$ m (nose skid),
$(0.2, -0.8)$ m (main wheel) and $(-5.0, -0.4)$ m (tail). At rest the glider
sits tail down on the main wheel and tail, at about 4.4° nose-up. This one
smooth force law covers standing, the ground roll, lift-off and a touchdown,
should it happen. There is no switching between modes, which keeps the
equations smooth and makes `jit`, `vmap` and `grad` straightforward.

= The rope: lumped masses with elasticity, weight and drag <sec-rope>

The rope is a large part of the launch. A steel rope weighs about 0.08 kg/m,
so 1200 m of it weigh about 100 kg, more than the pilot. At 30 m/s its air drag
is hundreds of newtons. It stretches by about 0.5–1 % at a working load of 5–6 kN, so it acts like a
long bungee. All three effects change the tension that reaches the hook, the
angle at which the rope pulls on the hook, and the oscillations in the ground
run.

== Discretisation <sec-rope-disc>

*Assumptions.* We model the rope as a chain of point masses (nodes) joined by
straight, massless segments that act as springs with dampers. All the rope's
mass sits in the nodes. The rope does not resist bending. Air drag is worked
out per segment and shared between its two end nodes.

The paid-out rope is split into $N$ segments of equal rest length
$ell_0 = L_0 \/ N$. Node 0 is fixed at the winch. Nodes $1, ..., N - 1$ are free
point masses $m_n = mu ell_0$, where $mu$ is the rope's mass per metre. Node
$N$ is the hook, a point on the glider.

The winch reels in at speed $v_r$, so $dot(L)_0 = -v_r$. In the model, _all_
rest lengths and node masses shrink together. This keeps the number of states
fixed, which JAX needs. The price is a small error in the momentum balance:
mass leaves every node, not only the one at the drum. The error vanishes as
$N -> oo$, and the test suite checks that the release height converges: with 12
and 24 segments it differs by less than 2 %.

The resolution is set in *segments per km* of laid-out rope
(`--segments-per-km`, default 10, so 100~m segments; at least 4 segments in
total). Short and long ropes are then resolved alike. The release height is not
very sensitive to the resolution. For the ASK 21 on the engine winch with
1200 m of Dyneema, 6, 12 and 24 segments give 463, 455 and 453 m. For the 10 km
rope of @sec-long-ropes, 20, 40 and 100 segments all give 2414–2415 m. A finer
rope mainly shows better where the rope lifts off the ground and how it
oscillates.

== Segment tension <sec-tension>

A rope pulls when stretched but cannot push. Its tension follows from how much
each segment is stretched, plus a little internal damping that calms the
oscillations along the rope.

For segment $j$, from node $j - 1$ to node $j$, let
$bold(d) = bold(p)_j - bold(p)_(j-1)$ be the vector along it,
$ell = abs(bold(d))$ its length, $hat(bold(e))_j = bold(d) \/ ell$ its
direction and $dot(ell) = (bold(u)_j - bold(u)_(j-1)) dot hat(bold(e))_j$ its
rate of stretching. Then

$
  T_j = op("softplus")_(2 thin "N")(E A (ell - ell_0) / ell_0
    + c (dot(ell) - ell dot(ell)_0 / ell_0)),
  quad c = 2 zeta sqrt(E A mu)
$ <eq-tension>

- $E A$: axial stiffness (Young's modulus times cross-section), in N;
- $ell_0 = L_0 \/ N$ and $dot(ell)_0 = -v_r \/ N$: rest length and its rate
  of change;
- $zeta = 0.2$: damping ratio;
- the softplus (width 2 N) makes a slack rope carry almost no tension.

The damping term is proportional to the rate of _strain_,
$dot(ell) - ell dot(ell)_0 \/ ell_0 = ell_0 thin d(ell \/ ell_0) \/ d t$.
So reeling in does not by itself create a damping force. With
$c = 2 zeta sqrt(E A mu)$ each segment's stretching mode has damping ratio
$zeta$, whatever its length. After release the top segment has a free end and
carries no tension.

== Aerodynamic drag (cross-flow principle) <sec-rope-drag>

A rope moving sideways through the air feels much more drag than one sliding
along its own length. We use the cross-flow principle: split the air-relative
velocity into a part across the rope and a part along it, and give each its own
drag coefficient.

For segment $j$, the mean velocity relative to the wind is
$bold(v)_"rel" = 1/2 (bold(u)_(j-1) + bold(u)_j) - bold(w)$, with the wind at
the segment's mid-height. Its part along the segment is
$bold(v)_t = (bold(v)_"rel" dot hat(bold(e))_j) hat(bold(e))_j$ and its part
across is $bold(v)_n = bold(v)_"rel" - bold(v)_t$. The drag is

$
  bold(F)_("drag", j) = -1/2 rho d ell (C_(D n) abs(bold(v)_n) bold(v)_n
    + pi C_f abs(bold(v)_t) bold(v)_t)
$ <eq-rope-drag>

- $rho$: air density at the segment's mid-height;
- $d$: rope diameter; $ell$: stretched segment length;
- $C_(D n)$: cross-flow drag coefficient, about 1.2 for a cylinder;
- $C_f$: skin-friction coefficient along the rope, 0.01–0.02; the factor
  $pi$ turns the frontal area $d ell$ into the surface area $pi d ell$.

Half of each segment's drag goes to each of its end nodes. Near the top of the
launch the upper rope swings round the winch at high speed, so rope drag grows
as the launch goes on.

== Node equations <sec-nodes>

Each free node is pulled by the two segments on either side, feels half the
drag of each, its weight and, while it lies on the runway, the ground:

$
  m_n dot(bold(u))_i = T_(i+1) hat(bold(e))_(i+1) - T_i hat(bold(e))_i
  + 1/2 (bold(F)_("drag", i) + bold(F)_("drag", i+1))
  - m_n g hat(bold(z)) + bold(F)_("gnd", i),
  quad dot(bold(p))_i = bold(u)_i
$ <eq-nodes>

for $i = 1, ..., N - 1$. Rope on the grass uses the same spring–damper and
friction law as the glider's wheels (@eq-ground). The spring is set by the node
weight, $k_g = m_n g \/ s_g$ with $s_g = 2$ cm (the rope sinks 2 cm into the
grass under its own weight), and the damping is critical,
$c_g = 2 m_n sqrt(g \/ s_g)$. The friction coefficient $mu_g$ is 0.5 for steel
and 0.4 for Dyneema. Dragging 100 kg of steel rope over grass costs about
500 N during the ground run. Ground contact is checked at the nodes only.

== Force on the glider <sec-hook-force>

The rope acts on the glider through the hook. The force is the pull of the top
segment, half its drag, the drag of the rope parachute and the weight of the
rope end:

$
  bold(F)_"hook" = -T_N hat(bold(e))_N + 1/2 bold(F)_("drag", N)
  - 1/2 rho (C_D A)_"chute" abs(bold(v)_(a,h)) bold(v)_(a,h)
  - m_"end" g hat(bold(z))
$ <eq-hook-force>

- $T_N$, $hat(bold(e))_N$: tension and direction of the top segment (from node
  $N - 1$ to the hook), so $-hat(bold(e))_N$ points from the hook down along
  the rope;
- $(C_D A)_"chute" = 0.1 thin "m"^2$: drag area of the closed parachute trailing
  in the rope end;
- $bold(v)_(a,h)$: velocity of the hook relative to the air;
- $m_"end" = m_"assy" + 1/2 m_n$: end assembly ($m_"assy" = 8$ kg: parachute,
  strop, weak link) plus half the top segment.

After release $bold(F)_"hook" = 0$. The *local* rope direction at the hook,
from the hook to the last free node, is steeper than the straight line to the
winch because the rope sags (@fig-geometry). Two angles are measured on this
local direction: its angle below the fuselage axis, which trips the
back-release when it exceeds 110°, and its angle below the horizontal, which
triggers the optional pilot release (@sec-events). The local direction also
sets the hook moment (@eq-hook-moment). The pilot's attitude schedule and the
winch driver do not use the rope angle; they use the elevation $beta$ of the
glider's CG seen from the winch.

== Validation: elastic catenary <sec-catenary>

A rope hanging still between two points takes the shape of an elastic
catenary. Its exact solution @irvine1981 is a good check of the lumped rope. It
is written in terms of the unstretched arc length $s$ from the first end. The
horizontal tension $H$ is constant along the rope and the vertical tension is
$V(s) = V_0 + w s$:

$
  x(s) & = (H s) / (E A) + H / w [sinh^(-1) V / H - sinh^(-1) V_0 / H], \
  z(s) & = (w s^2 \/ 2 + V_0 s) / (E A)
         + H / w [sqrt(1 + V^2 / H^2) - sqrt(1 + V_0^2 / H^2)]
$ <eq-catenary>

where $w = mu g$ is the rope weight per unstretched metre and $V_0$ the
vertical tension at $s = 0$ (negative where the rope sags below its end).
`cable.elastic_catenary` solves for $V_0$ and the unstretched length that fit
the given end points. The test suite hangs a steel rope of 20 segments between
two points 300~m apart and 80~m different in height, and checks that it settles
onto the catenary to within 1 % of the sag.

A consequence visible in the launch results: in the climb, the tension at the
hook is higher than the tension at the winch by roughly $mu g Delta z$, the
weight of a rope as long as the height difference $Delta z$ (about 300 N for
steel at 400~m).

== Rope presets and rope input <sec-rope-input>

Two generic presets are built in (@tab-ropes). They are typical values, not a
specific product.

#figure(
  booktab(
    columns: 3,
    align: (left, right, right),
    ([], [steel wire], [Dyneema]),
    [diameter $d$], [4.5 mm], [5.0 mm],
    [mass per length $mu$], [0.080 kg/m], [0.016 kg/m],
    [axial stiffness $E A$], [1.0 MN], [0.6 MN],
    [breaking load], [17 kN], [27 kN],
    [damping ratio $zeta$], [0.2], [0.2],
    [cross-flow drag $C_(D n)$], [1.2], [1.2],
    [skin friction $C_f$], [0.02], [0.01],
    [friction on grass $mu_g$], [0.5], [0.4],
    [end assembly mass $m_"assy"$], [8 kg], [8 kg],
    [parachute drag area $(C_D A)_"chute"$], [0.1 m²], [0.1 m²],
    [sinking into the grass $s_g$], [2 cm], [2 cm],
  ),
  caption: [Generic rope presets (`cable.ROPES`, approximate values).],
) <tab-ropes>

A specific rope can be given in three ways:

#[
#set par(justify: false)
- *From datasheet values* with `cable.rope_from_datasheet`. It takes
  `diameter_mm`, `mass_kg_per_100m`, `breaking_load_kN` and either
  `elongation_at_break_pct` or `EA_N`, plus the optional `CDn`, `Cf`,
  `ground_mu`, `zeta`, `end_mass_kg` and `end_CdA_m2`. Without `EA_N`, the
  stiffness is the secant value $E A = F_"break" \/ epsilon_"break"$, which
  assumes a straight load–elongation curve. Synthetic ropes are stiffer at
  working loads, so give `EA_N` from the load–elongation curve at 20–30 % of
  the breaking load when you have it.
- *From a TOML file* (`--rope-file`), either with the same datasheet keys
  (example: `ropes/example_dyneema_6mm.toml`), or with `base = "steel"` (or
  `"dyneema"`) plus overrides of `Rope` fields in SI units (example:
  `ropes/example_steel_override.toml`).
- *Single fields on the command line*, in SI units: `--rope-param mu=0.09`
  (repeatable).
]

The breaking load does not enter the dynamics. It is only used to report the
rope safety factor, $F_"break"$ divided by the highest tension anywhere in the
rope, in the summary table.

== Long ropes: why the hook tension does not grow <sec-long-ropes>

Rope weight enters the tension balance through height, not length. Along a
rope hanging in still air the tension changes by $d T \/ d z = mu g$. The hook
therefore feels at most the tension at the rope's lowest point plus
$mu g Delta z$, which is small: $0.16 thin "N/m" times 3000 thin "m" approx 0.5$ kN for
Dyneema. The rest of the rope's weight is carried by the winch end, by the sag,
or by the ground. What a long rope does is *use up* the winch's pull before it
reaches the hook: friction of the part still lying on the grass (up to
$mu_g m_"rope" g$) and drag of the long airborne part. @tab-long-ropes shows
this for the ASK 21 on the engine winch.

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
    [Dyneema], [1.2 km], [19 kg], [455 m], [6.2 kN], [6.3 kN], [0 %],
    [Dyneema], [10 km], [160 kg], [2414 m], [5.6 kN], [5.6 kN], [2 %],
    [Dyneema], [50 km], [800 kg], [no lift-off#super[b]], [5.6 kN], [1.7 kN], [99 %],
    [steel], [10 km], [800 kg], [no lift-off#super[b]], [8.6 kN], [3.0 kN], [98 %],
    [steel], [50 km], [4000 kg], [no lift-off], [7.0 kN], [1.0 kN], [100 %],
  ),
  caption: [Long ropes: ASK 21, engine winch, 10 segments/km, no wind.
    Tensions are maxima while the glider is on the rope.
    #super[a]Share of the rope nodes on the ground at 60 % of the release
    height; for failed launches at the end of the ground run (120 s).
    #super[b]Lifts off later if the 120 s limit is raised (see text).],
) <tab-long-ropes>

A longer rope first gives more height: 10 km of Dyneema take the ASK 21 to
2414~m instead of 455~m. The hook tension does not grow with the rope; it is
about the same as on the 1.2 km rope. But beyond some length, the extra rope
only adds friction and drag. At 50 km of Dyneema (800 kg) and at 10 km of steel
(also 800 kg) almost all of the engine's pull is spent dragging the rope over
the grass. Only a few hundred newtons reach the hook during the ground run. The glider
rolls at about the reel-in speed, 70–85 km/h, well below the pilot's target
speed. Whenever it hops off the ground, the pilot's speed correction lowers the
nose again (@sec-pilot), so it stays on the runway, and the launch counts as
failed after 120 s (`no_liftoff`, @sec-events). If this limit is
raised, both do lift off in the end, after a very long ground run: 10 km of
steel after about 175~s and 3.7~km, with release at 1540~m; 50 km of Dyneema
after about 12 minutes and 17~km, with release at 2710~m. The 50 km steel rope
(4 t, about 20 kN of friction on grass) cannot be dragged fast enough by this
winch at all; the glider reaches only 28 km/h.

= Winch <sec-winch>

The winch turns engine power into rope pull. How the pull changes with reel-in
speed decides how the launch feels: a constant pull, or one that fades as the
rope speeds up.

*Assumptions.* The drum, gearbox and engine are lumped into one effective mass
$M_"eff" = J \/ r_d^2$ at the rope ($J$ the moment of inertia of the rotating
parts, $r_d$ the drum radius). The drum radius does not change as rope winds
on. Both winch models share one equation for the reel-in speed:

$
  M_"eff" dot(v)_r = tau(t, beta) F_"avail" (v_r) - T_1 - b v_r,
  quad F_"avail" (v) = P_"max" / sqrt(v^2 + (P_"max" \/ F_"max")^2)
$ <eq-winch>

- $T_1$: tension of the rope segment at the winch;
- $b = 20$ N s/m: viscous losses in the drum;
- $F_"avail"$: the pull the winch can give at full throttle. It is a smooth
  $min(F_"max", P_"max" \/ v)$: at low speed the pull is limited to $F_"max"$
  (torque limit), at high speed by the power $P_"max"$;
- $tau$: the driver's throttle (@m-eq-throttle). It scales the whole pull
  curve, including the power limit.

The two winch types differ only in their parameters:

- *Tension-controlled winch* (`winch.tension_winch`): $P_"max" = 10^12$ W,
  which is practically infinite, so the winch pulls with a prescribed force
  $F_"max" = k_T m g$ (default $k_T = 1.1$, CLI `--pull`), whatever the reel
  speed. $M_"eff" = 60$ kg. This ideal winch isolates the glider's behaviour.
- *Engine winch* (`winch.engine_winch`): $P_"max" = 200$ kW,
  $F_"max" = 12$ kN and $M_"eff" = 250$ kg. The pull _drops_ as the reel speeds
  up and _rises_ when the glider slows the rope down (for example during the
  rotation into the climb). This is a real mechanism behind weak-link failures
  on light gliders. The driver sets the full throttle $tau_0$ so that the pull
  at 15 m/s reel speed is $k_T m g$ (a heavier glider gets more throttle),
  capped at $tau_0 = 1$. For the ASK 21, $tau_0 approx 0.57$.

*Winch driver.* The driver opens the throttle smoothly over 3 s, starting at
$t = 1$ s. Near the top the driver *throttles right down*: as the glider's
elevation seen from the winch, $beta$, goes from 65° to 70°, the throttle falls
to `fade_floor` $= 0$ of full:

$
  tau(t, beta) = tau_0 thin S((t - t_0) / t_r)
    [1 - (1 - f) thin S((beta - beta_0) / (beta_1 - beta_0))],
  quad beta = op("atan2")(z - z_w, x_w - x)
$ <m-eq-throttle>

with $S$ the smoothstep, $t_0 = 1$ s, $t_r = 3$ s, $beta_0 = 65°$,
$beta_1 = 70°$, $f = 0$ (`Winch.t_start`, `t_ramp`, `fade_beta0`,
`fade_beta1`, `fade_floor`), and $(x, z)$ the glider's CG. The drum stops
pulling, the rope slackens and its weight (plus the parachute drag) pulls the
rope end down and back. The glider flies on over the rope end until the rope
pulls from far behind and the hook's back-release trips (@sec-events). After
release the driver brakes the drum, $dot(v)_r = -v_r \/ (2 thin "s")$, and the
rope falls.

The driver's schedule is fixed in time and elevation. The driver does not react
to the glider's speed or the rope tension. The placarded winch speed $V_W$ is
only reported, never enforced.
