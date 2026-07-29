# Regenerate the serialized case9 test fixture.
#
# Run from the repo root:
#   julia --project test/fixtures/regenerate.jl
#
# This must be re-run whenever the PowerSystem struct changes.

using ResidualPowerFlow
using Powsybl
using JLD2

# Re-solve in single-slack mode (distributed slack and voltage controls off) so the
# stored controls/voltages are an exact RPF fixed point — matching the slow-tier
# _CASE9 build in runtests.jl and benchmark.jl. PowSyBl's default create_ieee9()
# solution uses distributed slack, which RPF's single-slack model cannot reproduce.
net = Powsybl.Network.create_ieee9()
p = Powsybl.LoadFlow.load_flow_parameters()
p.distributed_slack = false
p.use_reactive_limits = false
p.transformer_voltage_control_on = false
p.shunt_compensator_voltage_control_on = false
Powsybl.LoadFlow.run_ac(net, p)
sys = build_powersystem(net; K_DV = [130.0, 21.0, 13.0])

outpath = joinpath(@__DIR__, "case9.jld2")
jldsave(outpath;
    power_system = sys.power_system,
    controls     = controls_from_solution(sys),
    variables    = target_variables(sys),
)
println("Fixture saved to $outpath")
