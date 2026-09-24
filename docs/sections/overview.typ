#import "../template.typ": *

// Key physics and assumptions at a glance: the big picture before the details.

= Key physics and assumptions <sec-overview>

This section describes the model in words: what happens during a launch, which
effects are included, what is simplified or left out, and which inputs matter
most for the release height. The equations are in the sections that follow.

== A winch launch in brief

At the start the glider rests tail-down on its main wheel and tail skid. The
rope lies straight along the runway between the winch and the hook, almost taut
(0.2 % slack by default). After one second the winch driver opens the throttle
over three seconds. The rope tightens and stretches, and the glider accelerates.
The rope that is still on the grass is dragged along with it, which costs some
of the pull.

The pilot holds a little back stick during the ground roll. After lift-off the
pilot rotates gradually into a steep climb, reaching a pitch attitude of 35–40°
by about 30 m height. In the climb the wing carries the glider's weight _and_
the downward pull of the rope, so the wing load factor rises to about 3. The
pilot controls the speed with the pitch attitude: nose up if too fast, nose
down if too slow.

As the glider gets high above the winch, its elevation seen from the winch,
$beta$ (@fig-geometry), increases. From $beta = 55 degree$ the pilot starts to
lower the nose. Between 65° and 70° the winch driver closes the throttle
completely. The drum stops pulling and the rope goes slack. Its weight and the
drag of the parachute at its end pull the hook down and back. The glider flies
on over the rope end until the cable pulls from behind, at more than 110° below
the fuselage axis, and the hook's back-release opens. This is the normal end of
a launch in the model. A launch can also end when the weak link breaks, when
the rope is almost fully reeled in, or (optionally) when the pilot pulls the
release at a chosen cable angle. If the glider has not left the ground after
120 s, the launch is counted as failed (@sec-events).

== What the model includes

*The glider* is a rigid body that moves in the vertical plane: forward, up and
in pitch (@sec-glider). Pitch matters because the rope does not pull through the
centre of gravity (CG). The hook sits about 0.3 m ahead of and 0.45 m below the
CG (@sec-hook). While the rope is shallow, its pull below the CG pitches the nose
up; this is the pitch-up tendency just after lift-off. Once the rope is steeper
than about 50–60° below the fuselage axis, the moment changes sign and pitches the
nose down, so the pilot needs more and more back stick in the climb. The glider
touches the ground at three points (nose skid, main wheel, tail) with springs,
dampers and friction (@sec-ground). The wheel rolls with little friction; the
skids slide.

*Aerodynamics* (@sec-aero). Lift grows linearly with the angle of attack, the
elevator deflection and the pitch rate, up to a smooth ceiling at the maximum
lift coefficient: the wing stalls, but gently. Drag is a parabolic polar with
extra drag beyond the stall. The pitching moment gives static stability and
pitch damping. All coefficients are estimated from handbook data (span, wing
area, best glide ratio, maximum lift coefficient) and generic rules, not
measured.

*The air.* The density falls with height following the International Standard
Atmosphere, with an optional airfield elevation and temperature offset. The
aerodynamic forces depend on the dynamic pressure, i.e. on the indicated
airspeed (IAS); the motion depends on the true airspeed (TAS). The pilot flies
IAS, as a real pilot does. An optional headwind grows with height (a power law)
and acts on the glider, the rope and the parachute.

*The rope* is a chain of point masses joined by straight, elastic segments,
about 100 m long by default (@sec-rope, @sec-rope-disc). Four effects are
included:
- _Weight._ The rope sags below the straight line to the winch, so the local
  rope angle at the hook is steeper than the line of sight. The hanging rope
  pulls the hook down, and the tension at the hook is larger than at the winch
  by roughly the weight of the rope between the two heights (@sec-catenary).
- _Stretch._ The rope acts like a stiff bungee (@sec-tension). It shapes the
  tension oscillations during the ground run, but hardly changes the energy
  that reaches the glider.
- _Aerodynamic drag_ across and along each segment (@sec-rope-drag). It grows
  towards the top of the launch, when the upper rope swings round the winch at
  high speed, broadside to the air.
- _Ground friction_ of the rope lying on the grass, which matters for heavy
  and long ropes (@sec-long-ropes).
The rope end at the glider (parachute, strop and weak link, 8 kg with 0.1 m² of
drag area) hangs at the hook. The rope can be steel, Dyneema or any rope given
by its datasheet (@sec-rope-input).

*The winch* is a drum with an effective mass that reels the rope in against
the rope tension and some drum friction (@sec-winch). Two pull curves are
available. The _tension winch_ pulls with a fixed force, by default 1.1 times
the glider's weight. The _engine winch_ has a power limit (200 kW), so its pull
drops as the rope speeds up and rises when the glider slows the rope down.

*Pilot and winch driver* are simple, smooth control laws (@sec-winch,
@sec-pilot). The pilot aims for a pitch attitude that depends on height, IAS
and $beta$, and moves the elevator through a PID controller with a deflection
limit and a 0.25 s lag. The winch driver follows a fixed throttle schedule:
ramp up at the start, close as $beta$ passes 65–70°. The driver does not watch
the glider's speed or the rope tension. After release the drum is braked.

== Main assumptions and simplifications

@tab-ov-assumptions lists the main simplifications. Most are standard for a
longitudinal flight model. The pilot and winch-driver laws are the weakest part:
real people react later, less smoothly and less predictably.

#figure(
  booktab(
    columns: (auto, 1fr),
    ([topic], [assumption in the model]),
    [motion], [2-D, longitudinal only: no roll, yaw, sideslip or crosswind],
    [glider], [rigid body; no wing bending, no flaps, no ground effect],
    [aerodynamics],
    [quasi-steady coefficients, linear up to a smooth stall; generic estimates
      from handbook data, not manufacturer data],
    [air],
    [ISA density with height; IAS = equivalent airspeed (no instrument or
      position error, no compressibility); steady wind, no gusts or thermals],
    [rope],
    [point masses joined by massless, straight, linearly elastic segments; rope
      reeled in by shrinking every segment equally, so mass leaves all nodes
      instead of only at the drum],
    [rope end],
    [parachute, strop and weak link lumped at the hook; closed parachute with a
      constant drag area; not modelled after release],
    [contact],
    [smooth spring–damper ground contact and smoothed Coulomb friction, for the
      glider and the rope],
    [winch], [one effective drum mass; a smooth pull curve (constant force or
      power-limited)],
    [pilot],
    [attitude control with a smooth schedule and a PID elevator law; no
      reaction delay beyond the 0.25 s stick lag],
    [winch driver],
    [open-loop throttle schedule in time and glider elevation $beta$; no
      reaction to speed or tension],
    [end of launch],
    [back-release at a fixed cable angle of 110° below the fuselage axis; weak
      link breaks at a fixed tension],
  ),
  caption: [Main assumptions of the model.],
) <tab-ov-assumptions>

Every switch-like effect (the rope going slack, a wheel leaving the ground,
friction, the stall, the phases of the pilot's schedule) is rounded off
smoothly. This keeps the equations one continuous system without mode changes,
which makes batching and exact gradients possible (@sec-solver). Near the
switch points, the smoothing is a small approximation.

*Not modelled:* ground effect, flaps, lateral motion and wing drop, cable
breaks other than the weak link, the parachute opening after release, gusts and
turbulence, instrument errors, human reaction delays and mistakes, and the
detailed mechanics of the winch drum and engine. The glider data are
approximations. See @sec-limits for more.

== What matters most for the release height

The simulator computes the gradient of the release height with respect to every
parameter (@sec-sensitivity). @tab-ov-sens shows the largest ones for the
default launch: ASK 21, 1200 m Dyneema rope, tension winch, no wind. The
release height is 491 m.

#figure(
  booktab(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    ([parameter (default)], [$Delta h$ for +10 %], [elasticity]),
    [rope length (1200 m)], [+47 m], [0.96],
    [winch pull (1.1 × weight)], [+26 m], [0.53],
    [glider mass (470 kg)], [−23 m], [−0.46],
    [pilot: $beta$ where nose-lowering ends (70°)], [+8 m], [0.16],
    [pilot: climb attitude (40°)], [+6 m], [0.12],
    [pilot: target speed (105 km/h IAS)], [−6 m], [−0.12],
    [pilot: $beta$ where nose-lowering starts (55°)], [+4 m], [0.09],
    [back-release angle (110°)], [+4 m], [0.09],
    [rope diameter (5 mm)], [−3 m], [−0.07],
    [rope normal drag coefficient (1.2)], [−3 m], [−0.07],
    [wing area (17.95 m²)], [+3 m], [0.06],
  ),
  caption: [Largest sensitivities of the release height; ASK 21, 1200 m
    Dyneema, tension winch ($h = 491$ m). Elasticity: % change in height per %
    change in the parameter. Local values; each parameter is changed on its
    own.],
) <tab-ov-sens>

The picture is simple. The energy the winch puts into the rope is roughly its
pull times the length of rope reeled in, so rope length and winch pull come
first. A heavier glider needs more of that energy per metre of height, so mass
comes next, with the opposite sign. Then come the pilot's and the driver's
choices near the top: when to lower the nose, how steep to climb, how fast to
fly. These change the height by a few metres each, and their signs can differ
between set-ups (with a steel rope, for example, the back-release angle has the
opposite effect). The rope's drag (diameter × drag coefficient) is the most
important rope property. The weight of a Dyneema rope costs well under a metre;
for steel, five times heavier, it is comparable to the drag (about −3 m for
+10 %). Rope stiffness has practically no effect. The glider's aerodynamics
(polar, maximum lift) change the height by only about 1–2 m for +10 %.

Some inputs have no simple percentage because their default is zero. A headwind
adds about 22 m of height per m/s at 10 m. With the tension winch, a high or hot
airfield costs very little height (about 2 m per 1000 m of elevation), because
the pull is the same and the pilot flies the same IAS; the glider only needs
more true airspeed and a longer ground roll. These are local derivatives around
the default launch. For large changes, run the simulation directly.
