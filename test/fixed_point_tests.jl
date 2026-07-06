using LinearAlgebra: norm

# Fixed-point tests: from a converged state, one more iteration of any
# in-scope iterative solver must produce a (near-)zero update.
#
# Flavour A — residual ≈ 0 (true power-flow solution) OR a GN-converged
#             state (stationary point of the iteration), depending on
#             whether the setup admits an exact AC solution.
# Flavour B — residual ≠ 0, J'r ≈ 0 (RPF minimum of an infeasible point).

const _FP_TOL_F64 = 1e-10

# Float32 fixed-point tests are deferred.
# The 2-bus overdetermined system (P_rated=50, controls [0.5,1.0,0.3,0.1]) produces
# a large initial residual that causes Float32 GN to drift past the ‖r‖² minimum and
# eventually overflow when iteration continues beyond the ‖J'r‖ noise floor. A robust
# Float32 fixed-point test requires either a well-determined system or plateau detection
# in the solver. Deferred as an extension.

# Returns max-abs update from a single iteration without mutating `state`.
function _fp_update_norm(state, solver)
    s = deepcopy(state)
    r = step!(s, solver)
    return norm(r.variables_update, Inf)
end

function _fp_build_two_bus(::Type{T}) where {T}
    builder = PowerSystemBuilder{T}(2)
    add_one_bus_injector!(builder, 1, SynchronousMachineStatic(T(50.0)))
    add_one_bus_injector!(builder, 2, ConstantPowerLoad(T(1.0)))
    add_two_bus_injector!(builder, 1, 2, Line(T(0.01), T(0.05), T(0.02)))
    return build!(builder)
end

# ── Flavour A — 2-bus ─────────────────────────────────────────────────────────
#
# The 2-bus test system is structurally overdetermined (4 real residuals from
# two complex bus current balances, 3 variables V1/V2/θ), so GN converges to
# an RPF minimum with nonzero residual rather than to r = 0. We drive GN well
# below the fixed-point tolerance so the resulting state is a clean fixed
# point of the iteration.

@testset "fixed point (A) — 2-bus Float64" begin
    ps = _fp_build_two_bus(Float64)
    controls = Float64[0.5, 1.0, 0.3, 0.1]

    state = PowerFlowState(ps, controls)
    solved, stats = solve(state, GaussNewtonSolver();
        solver_options = SolverOptions(
            max_iterations = 500,
            tolerance      = 1e-12,
        ))
    @test stats["converged"]

    @test _fp_update_norm(solved, GaussNewtonSolver()) < _FP_TOL_F64
end

# ── Flavour A — case9 ─────────────────────────────────────────────────────────

if !isnothing(_CASE9)
    @testset "fixed point (A) — case9 Float64" begin
        ps   = _CASE9.power_system
        ctrl = controls_from_solution(_CASE9)

        state = PowerFlowState(ps, ctrl)
        solved, stats = solve(state, GaussNewtonSolver();
            solver_options = SolverOptions(
                max_iterations = 200,
                tolerance      = 1e-12,
            ))
        @test stats["converged"]
        @test stats["residual_norm"] ≤ 1e-7

        # The Powsybl-built case9 used in the slow tier carries a small slack
        # imbalance in its native solution controls, so r cannot be driven all
        # the way to machine zero — the fixed-point update is floored at
        # ~‖r‖. The fast-tier JLD2 fixture has r ≈ 1e-14 and hits the tight
        # 1e-10 threshold; the slow-tier build sits around r ≈ 1e-8. Scale the
        # threshold so both work.
        fp_tol = max(_FP_TOL_F64, 10 * stats["residual_norm"])

        @test _fp_update_norm(solved, GaussNewtonSolver()) < fp_tol
    end

    # ── Flavour B — case9 RPF minimum ──────────────────────────────────────────
    @testset "fixed point (B) — case9 RPF minimum Float64" begin
        ps   = _CASE9.power_system
        ctrl = copy(controls_from_solution(_CASE9))
        ctrl[1] *= 3.0

        # ‖J'r‖ floor at the RPF minimum scales with ‖J‖·ε·‖r‖; for case9 with
        # ‖r‖ ≈ 0.47 the floor sits ≈ 2.4e-12, so the 1e-12 drive used by
        # Flavour A is unreachable here. 1e-11 keeps a comfortable margin.
        state = PowerFlowState(ps, ctrl)
        solved, stats = solve(state, GaussNewtonSolver();
            solver_options = SolverOptions(
                max_iterations = 200,
                tolerance      = 1e-11,
            ))
        @test stats["converged"]
        @test stats["residual_norm"] > 1e-3

        @test _fp_update_norm(solved, GaussNewtonSolver()) < _FP_TOL_F64
    end
end
