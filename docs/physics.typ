// Physics write-up of the winch-launch simulator.
// Build:  typst compile docs/physics.typ    (-> docs/physics.pdf)
// Content lives in sections/*.typ; shared helpers in template.typ.

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

#align(center)[
  #text(17pt, weight: "bold")[Physics of a glider winch launch]
  #v(0.2em)
  #text(11pt)[Model and numerics of `winch_sim`]
]
#v(1em)

#include "sections/intro.typ"
#include "sections/overview.typ"
#include "sections/model.typ"
#include "sections/operation.typ"

#bibliography("refs.yml", style: "ieee")
