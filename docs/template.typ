// Shared helpers for the physics write-up; import with
//   #import "../template.typ": *

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
