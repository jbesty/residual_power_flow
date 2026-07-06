
export solve, step!, solve!

# ── solve! (simple, non-PtO path) ─────────────────────────────────────────────
#
# The simple shape runs the inner iterate-to-convergence loop directly. PtO
# callers use the three-arg method `solve!(state, solver, pto_config; ...)`
# defined in pto.jl.

function solve!(
    state::PowerFlowState,
    solver::PowerFlowSolver;
    solver_options::SolverOptions = SolverOptions(),
)
    iteration_count, max_update_size = _iterate_to_convergence!(
        state, solver_options,
        () -> step!(state, solver; solver_options))

    residuals = compute_residual!(state)
    residual_norm = sqrt(sum(abs2, residuals))
    converged = max_update_size <= solver_options.tolerance
    return Dict(
        "iteration_count" => iteration_count,
        "residual_norm"   => residual_norm,
        "converged"       => converged,
    )
end

# ── solve (non-mutating, simple) ──────────────────────────────────────────────

function solve(
    state::PowerFlowState,
    solver::PowerFlowSolver;
    solver_options::SolverOptions = SolverOptions(),
)
    new_state = deepcopy(state)
    stats = solve!(new_state, solver; solver_options)
    return new_state, stats
end

# Default entry point: use GaussNewtonSolver with default options.
function solve(
    state::PowerFlowState;
    solver_options::SolverOptions = SolverOptions(),
)
    return solve(state, GaussNewtonSolver(); solver_options)
end
