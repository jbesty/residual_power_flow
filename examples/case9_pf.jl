using Powsybl
using ResidualPowerFlow
using Printf

# Solve the IEEE case 9 network using ResidualPowerFlow (RPF).
#
# RPF minimises a current-balance residual rather than requiring AC feasibility
# as a hard constraint. This script shows the standard feasible case: starting
# from the MATPOWER solution controls, the solver finds the matching AC voltages.
#
# How to run:
#   julia --project examples/case9_pf.jl

sys = build_powersystem(:case9; K_DV = [130.0, 21.0, 13.0])
ps  = sys.power_system

# Initial state: flat-start voltages (1 p.u., 0 rad), controls from MATPOWER solution.
controls = controls_from_solution(sys)
state    = PowerFlowState(ps, controls)

solver = GaussNewtonSolver()
stats  = solve!(state, solver; solver_options = SolverOptions(max_iterations = 50, tolerance = 1e-8))

println("Converged: ", stats["converged"],
        "  ($(stats["iteration_count"]) iterations,",
        "  residual norm = $(round(stats["residual_norm"], sigdigits=4)))")

println("\nBus voltage magnitudes (p.u.):")
v = voltage_magnitudes(state)
for i in 1:ps.n_buses
    @printf("  Bus %2d:  %.6f\n", i, v[i])
end
