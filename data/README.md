# Data provenance

## File

[`silicon_wafer_thickness.csv`](silicon_wafer_thickness.csv) — $184$ wafer
sections $\times$ $9$ gauge groups.

| Column | Meaning |
| --- | --- |
| `G1` … `G9` | Thickness deviation at gauge group $j$, relative to a manufacturing target thickness |

## Units and semantics

Values are **signed thickness deviations** in the native length unit of the
metrology tool (not absolute wafer thickness). A reading of $0$ means the
gauge sits on target; positive / negative values mean locally thick / thin
relative to that target.

## Spatial layout

Exact $(x,y)$ survey coordinates were not recorded with this table. The
analysis therefore adopts a conventional schematic **$3\times 3$** gauge
grid (row-major `G1`…`G9`) solely to map principal loadings onto a wafer
disk. Conclusions about *which gauges behave alike* follow from the loadings
themselves and do not depend on that schematic; only the wafer-map figures do.

## Provenance

The table originates from a silicon-wafer thickness metrology exercise used
to study correlated multi-sensor thickness fields. No additional metadata
(tool model, wafer diameter, or process recipe) accompanies the numeric
file.
