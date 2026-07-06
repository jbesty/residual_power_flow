# MATPOWER reference cases (`.mat`)

Directly-loadable MATPOWER cases in PowSyBl's binary `.mat` format, each holding
an AC-power-flow-converged operating point. These are the canonical test/reference
grids; load one with the PowSyBl extension:

```julia
using Powsybl, ResidualPowerFlow
sys = build_powersystem(joinpath("data", "matpower", "case9.mat"))
```

`target_variables(sys)` returns the embedded converged voltage state and
`controls_from_solution(sys)` the matching RPF control vector, so the case
doubles as its own cross-validation reference (see
`scripts/compare_matpower_cases.jl`).

## Why `.mat`, not `.m`

PowSyBl's MATPOWER importer reads **only** the binary `.mat` MAT-file, never the
`.m` script
(<https://powsybl.readthedocs.io/projects/powsybl-core/en/latest/grid_exchange_formats/matpower/import.html>).
The `.m` sources are the standard MATPOWER 8.0 library cases and are not vendored
here.

## Cases

`case4gs`, `case5`, `case6ww`, `case9`, `case14`, `case24_ieee_rts`, `case30`,
`case39`, `case57`, `case69`, `case89pegase`, `case118`, `case145`,
`case_ACTIVSg200`, `case300`, `case1354pegase` — 4 to 1354 buses.

`case89pegase.mat` is also the regression fixture for the PowsyblIOExt phase-shift
and shunt-conductance handling (`test/powsybl_io.jl`, system-wide tier).

## Regenerating

Produced from the MATPOWER 8.0 library via `scripts/convert_matpower_to_mat.m`
(`loadcase` → `runpf` → `savecase`); requires MATLAB/Octave + MATPOWER. Edit the
case list there to add more.
