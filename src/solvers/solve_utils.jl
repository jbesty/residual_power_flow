
# Shared helpers used by step! / pto_step! / solve! across solver types.

function _cycle_kw(state::PowerFlowState)
    topo = state._topology_snapshot
    has_tripped = any(!, topo)
    return (cycle_participation = state.active_cycle_participation,
            cycle_admittances   = state.active_cycle_admittances,
            n_cycles            = state.n_cycles_active[],
            _topology_snapshot  = has_tripped ? topo : nothing,
            _voltage_sparsity_cache = state._voltage_sparsity_cache,
            _control_sparsity_cache = state._control_sparsity_cache)
end

function _iterate_to_convergence!(
    state::PowerFlowState,
    solver_options::SolverOptions,
    step_fn::F,
) where {F}
    max_update_size = one(eltype(state.voltages))
    iteration_count = 0
    while max_update_size > solver_options.tolerance && iteration_count < solver_options.max_iterations
        result = step_fn()
        max_update_size = result.max_update
        iteration_count += 1
    end
    return iteration_count, max_update_size
end
