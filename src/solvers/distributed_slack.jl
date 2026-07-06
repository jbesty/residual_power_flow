
export solve_distributed_slack!, solve_distributed_slack

using SparseArrays: sparse

# ── Distributed-slack feasibility solve ───────────────────────────────────────
#
# Free `free_indices` (the generator-P controls) as a distributed slack and drive
# ‖r‖ → 0 by joint Gauss–Newton on [variables; free controls]. The N free controls
# collapse to a single scalar λ via weights (P-proportional by default), so each
# step augments the voltage Jacobian with one weighted control column and solves
# the combined system. The slack absorbs the power-balance mismatch, so a sampled
# operating condition becomes an AC-feasible solution.
#
# This is the standalone reduction of the old ACFeasiblePF joint-Newton PtO close,
# the load-bearing path for the feasible and balanced/close_slack data samplers.

function _distributed_slack_weights(controls, indices, weights)
    isempty(weights) || return weights
    p_vals = abs.(controls[indices])
    total = sum(p_vals)
    total == 0 && throw(ArgumentError(
        "all free-index P controls are zero — cannot compute distributed-slack weights"))
    return p_vals / total
end

function _distributed_slack_step!(state::PowerFlowState, indices, weights, solver_options)
    ps = state.power_system
    v, u = state.voltages, state.controls
    jm = solver_options.jacobian_method
    st = state.statuses
    cycle_kw = _cycle_kw(state)

    residual = compute_residual!(state)
    jacobian_voltages = compute_jacobian_voltages(ps, v, u, jm; statuses = st, cycle_kw...)
    jacobian_controls = compute_jacobian_controls(ps, v, u, jm; statuses = st, cycle_kw...)

    w = _distributed_slack_weights(u, indices, weights)
    weighted_col = jacobian_controls[:, indices] * w
    jacobian = hcat(jacobian_voltages, sparse(weighted_col))
    combined_update = -(jacobian \ residual)
    variables_update = combined_update[1:end-1]

    controls_update = fill!(state.controls_update_buffer, zero(eltype(v)))
    controls_update[indices] = combined_update[end] .* w

    state.voltages .+= solver_options.step_size_variables .* variables_update
    state.controls .+= solver_options.step_size_controls .* controls_update
    compute_current_balance!(state)

    max_update = maximum(abs, variables_update)
    return (; max_update, residual, variables_update, controls_update)
end

# Mutating entry point. Returns a stats Dict with the iteration count, final
# residual norm, and convergence flag (matching the old PtO close return shape).
function solve_distributed_slack!(
    state::PowerFlowState,
    free_indices::AbstractVector{<:Integer};
    weights::AbstractVector{<:Real} = Float64[],
    tol::Real = 1.0e-8,
    max_iterations::Int = 100,
    jacobian_method::JacobianMethod = ExplicitAnalytical(),
)
    isempty(free_indices) && throw(ArgumentError(
        "solve_distributed_slack!: free_indices is empty"))
    opts = SolverOptions(; max_iterations, tolerance = tol, jacobian_method)
    iteration_count, max_update_size = _iterate_to_convergence!(
        state, opts,
        () -> _distributed_slack_step!(state, free_indices, weights, opts))

    residuals = compute_residual!(state)
    residual_norm = sqrt(sum(abs2, residuals))
    converged = max_update_size <= opts.tolerance && residual_norm <= opts.tolerance
    return Dict(
        "iteration_count" => iteration_count,
        "residual_norm"   => residual_norm,
        "converged"       => converged,
    )
end

# Non-mutating wrapper: returns (new_state, stats).
function solve_distributed_slack(
    state::PowerFlowState,
    free_indices::AbstractVector{<:Integer};
    kwargs...,
)
    new_state = deepcopy(state)
    stats = solve_distributed_slack!(new_state, free_indices; kwargs...)
    return new_state, stats
end
