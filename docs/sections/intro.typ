#import "../template.typ": *

// Introduction and launch-geometry figure.

In a winch launch a glider is pulled into the air by a long rope. A winch at the
far end of the runway reels the rope in. The glider rolls a short distance,
lifts off, climbs steeply on the rope and releases near the top, typically at
about 40 % of the rope length above the ground.

This document describes the physics behind the simulator in `winch_sim/` and
how the equations are solved with
#link("https://docs.kidger.site/diffrax/")[diffrax]. The model has three
coupled parts (@fig-geometry):

+ the *glider*, a rigid body that moves in the vertical plane. It has three
  degrees of freedom: forward and upward motion, and pitch;
+ the *rope*, about 1–1.5 km of steel wire or synthetic (Dyneema) rope. It has
  mass, it stretches, it feels air drag, and at the start it lies on the runway;
+ the *winch*, a drum that reels the rope in. It is either an ideal
  tension-controlled winch or a winch with a power-limited engine.

At the start the glider stands still on the runway and the whole rope lies
straight on the grass between the glider and the winch.

Two simple control laws close the loop: the *pilot* moves the elevator, and the
*winch driver* sets the throttle.

#figure(
  box(width: 14cm, height: 5.4cm, {
    let ground = 4.7cm
    let hook = (4.7cm, 1.25cm)
    let winch = (12.9cm, ground - 0.2cm)
    let blue = rgb("#2a78d6")
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
    // rope: sags below the line of sight (sag exaggerated)
    place(curve(
      stroke: 1.3pt + blue,
      curve.move(hook),
      curve.quad((8.6cm, 4.0cm), winch),
    ))
    place(dx: 5.6cm, dy: 3.65cm, text(8pt, fill: blue)[rope: sags under its weight,])
    place(dx: 5.6cm, dy: 4.0cm, text(8pt, fill: blue)[drag pushes it back])
    // local rope direction at the hook (tangent of the rope curve)
    place(line(start: hook, end: (hook.at(0) + 2.0cm, hook.at(1) + 1.41cm),
      stroke: (paint: blue, thickness: 0.6pt, dash: "dashed")))
    place(dx: 3.35cm, dy: 2.05cm, box(width: 2.1cm, align(right,
      text(7.5pt, fill: blue)[local rope \ direction])))
    // winch drum
    place(dx: winch.at(0) - 0.2cm, dy: winch.at(1) - 0.2cm, rect(width: 0.4cm, height: 0.4cm, fill: black))
    place(dx: winch.at(0) - 0.4cm, dy: ground + 0.12cm, text(8pt)[winch])
    place(dx: winch.at(0) - 0.4cm, dy: winch.at(1) - 0.65cm, text(8pt)[$bold(r)_w$])
    place(dx: 10.6cm, dy: 3.35cm, text(9pt)[$beta$])
    // glider (fuselage + fin) pitched nose-up
    place(dx: hook.at(0) - 0.9cm, dy: hook.at(1) - 0.32cm, rotate(-30deg, origin: center, {
      box(width: 1.8cm, height: 0.5cm, {
        place(dx: 0cm, dy: 0.3cm, line(length: 1.8cm, stroke: 2pt + rgb("#eb6834")))
        place(dx: 0.05cm, dy: 0.3cm, line(length: 0.3cm, angle: -90deg, stroke: 2pt + rgb("#eb6834")))
      })
    }))
    place(dx: hook.at(0) - 0.08cm, dy: hook.at(1) - 0.08cm, circle(radius: 0.08cm, fill: black))
    place(dx: hook.at(0) + 0.2cm, dy: hook.at(1) - 0.4cm, text(8pt)[hook])
    place(dx: hook.at(0) - 1.0cm, dy: hook.at(1) - 1.15cm, text(8pt)[glider, pitch $theta$])
    // axes
    place(dx: 13.2cm, dy: ground - 0.35cm, text(9pt)[$x$])
    place(dx: 0.05cm, dy: 0cm, text(9pt)[$z$ (up)])
  }),
  caption: [Geometry of a winch launch (not to scale). The glider starts at the
    left and the winch pulls from the right. The rope sags below the straight
    line of sight, so at the hook it pulls more steeply downwards than the line
    to the winch. $beta$ is the elevation of the glider seen from the winch.],
) <fig-geometry>
