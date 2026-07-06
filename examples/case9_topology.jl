using Powsybl
using ResidualPowerFlow
using Printf

# Compare RPF solutions before and after removing a transmission line.
#
# Topology changes are represented via the statuses vector in PowerFlowState.
# Setting a branch status to 0 removes it from the current balance.
#
# How to run:
#   julia --project examples/case9_topology.jl

sys = build_powersystem(:case9; K_DV = [130.0, 21.0, 13.0])
ps  = sys.power_system

controls = controls_from_solution(sys)
solver   = GaussNewtonSolver()

# Base case: all branches in service (default all-ones statuses).
state_base = PowerFlowState(ps, controls)
solve!(state_base, solver; solver_options = SolverOptions(max_iterations = 50, tolerance = 1e-8))

# Remove branch 3 (bus 4 → bus 7).
# Statuses layout: first length(single_bus_injectors) entries are injectors,
# then one entry per branch.
n_inj = length(ps.single_bus_injectors)
statuses = ones(Float64, n_inj + ps.n_lines)
statuses[n_inj + 3] = 0.0   # branch 3 out of service

state_mod = PowerFlowState(ps, get_flat_start(ps), controls, statuses)
update_topology!(state_mod)
solve!(state_mod, solver; solver_options = SolverOptions(max_iterations = 50, tolerance = 1e-8))

println("Branch 3 (bus 4 → bus 7) removed.")
println("Bus  |  Base (p.u.)  |  Modified (p.u.)  |  Delta")
v_base = voltage_magnitudes(state_base)
v_mod  = voltage_magnitudes(state_mod)
for i in 1:ps.n_buses
    @printf("  %2d  |   %.6f   |     %.6f     |  %+.6f\n", i, v_base[i], v_mod[i], v_mod[i] - v_base[i])
end
