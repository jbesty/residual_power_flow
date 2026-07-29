import ForwardDiff

export compute_current_injection
export compute_current_balance, compute_current_balance!
export compute_residual, compute_residual!
export compute_energy, compute_energy_hessian
export update_topology!
export controls_from_voltages

# ── Topology-aware cycle recomputation ────────────────────────────────────────

function update_topology!(state::PowerFlowState)
    ps = state.power_system
    n_single = length(ps.single_bus_injectors)
    n_lines  = ps.n_lines

    # Current active mask from statuses
    active_mask = Vector{Bool}(undef, n_lines)
    @inbounds for k in 1:n_lines
        active_mask[k] = state.statuses[n_single + k] > 0
    end

    # No-op if topology unchanged
    if active_mask == state._topology_snapshot
        return
    end
    state._topology_snapshot .= active_mask

    # Build active edge list and index mapping
    active_indices = findall(active_mask)
    active_edges = ps.edge_list[active_indices]

    if isempty(active_edges)
        resize!(state.active_cycle_participation, 0)
        resize!(state.active_cycle_admittances, 0)
        state.n_cycles_active[] = 0
        return
    end

    _, active_participation_reduced = compute_cycles_and_participation(active_edges)
    n_active_cycles = length(active_participation_reduced)

    # Map reduced participation vectors (indexed over active edges) back to
    # full edge indices (length n_lines).
    resize!(state.active_cycle_participation, n_active_cycles)
    resize!(state.active_cycle_admittances, n_active_cycles)
    for k in 1:n_active_cycles
        full_participation = zeros(Int, n_lines)
        for (local_idx, full_idx) in enumerate(active_indices)
            full_participation[full_idx] = active_participation_reduced[k][local_idx]
        end
        state.active_cycle_participation[k] = full_participation
        state.active_cycle_admittances[k] = one(eltype(state.active_cycle_admittances))
    end

    state.n_cycles_active[] = n_active_cycles

    # Invalidate cached sparsity patterns and Jacobian buffers
    state._voltage_sparsity_cache[] = nothing
    state._control_sparsity_cache[] = nothing
    state._voltage_jacobian_cache[] = nothing
    state._control_jacobian_cache[] = nothing
end

# ── System-level current injection (raw, no side effects) ─────────────────────
#
# Returns (single_bus_currents, branch_currents):
#   single_bus_currents: length 2·n_single  ([i_d, i_q] per single-bus injector)
#   branch_currents:     length 4·n_branches ([i_d,i_q] from-end, [i_d,i_q] to-end per branch)

function compute_current_injection(
    power_system::PowerSystem,
    voltages::AbstractVector{<:Real},
    controls::AbstractVector{<:Real};
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    n_single   = length(power_system.single_bus_injectors)
    n_branches = length(power_system.branch_injectors)
    n_buses    = power_system.n_buses
    T = promote_type(eltype(voltages), eltype(controls))

    single_bus_currents = Vector{T}(undef, 2 * n_single)
    branch_currents     = Vector{T}(undef, 4 * n_branches)

    control_cursor = 1
    @inbounds for ii in 1:n_single
        injector = power_system.single_bus_injectors[ii]
        n_component_controls = n_controls(injector.component)
        control_slice = view(controls,
                             control_cursor:control_cursor+n_component_controls-1)
        current = compute_current_injection(injector.component,
                                            voltages[injector.bus_id], control_slice)
        status = statuses === nothing ? one(T) : T(statuses[ii])
        single_bus_currents[2*ii-1] = status * current[1]
        single_bus_currents[2*ii]   = status * current[2]
        control_cursor += n_component_controls
    end

    @inbounds for k in 1:n_branches
        branch = power_system.branch_injectors[k]
        v_from = voltages[branch.from_bus]
        v_to   = voltages[branch.to_bus]
        branch_angle = voltages[n_buses + k]
        i_from, i_to = compute_current_injection(branch.component, v_from, v_to,
                                                 branch_angle)
        status = statuses === nothing ? one(T) : T(statuses[n_single + k])
        branch_currents[4*k-3] = status * i_from[1]
        branch_currents[4*k-2] = status * i_from[2]
        branch_currents[4*k-1] = status * i_to[1]
        branch_currents[4*k]   = status * i_to[2]
    end

    return single_bus_currents, branch_currents
end

# ── Current-balance scatter kernel (single implementation) ───────────────────
#
# All compute_current_balance variants delegate to this kernel.
# `current_balance` must be pre-zeroed by the caller.
# When `single_bus_currents` / `branch_currents` are not nothing, per-component
# currents are stored there (state-aware callers need this).

function _scatter_current_balance!(
    current_balance::AbstractVector,
    power_system::PowerSystem,
    voltages::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    statuses::Union{Nothing, AbstractVector{<:Real}},
    single_bus_currents::Union{Nothing, AbstractVector{<:Real}},
    branch_currents::Union{Nothing, AbstractVector{<:Real}},
)
    n_single = length(power_system.single_bus_injectors)
    n_buses  = power_system.n_buses
    T = eltype(current_balance)

    control_cursor = 1
    @inbounds for ii in 1:n_single
        injector = power_system.single_bus_injectors[ii]
        n_component_controls = n_controls(injector.component)
        control_slice = view(controls,
                             control_cursor:control_cursor+n_component_controls-1)
        current = compute_current_injection(injector.component,
                                            voltages[injector.bus_id], control_slice)
        status = statuses === nothing ? one(T) : T(statuses[ii])
        if single_bus_currents !== nothing
            single_bus_currents[2*ii-1] = status * current[1]
            single_bus_currents[2*ii]   = status * current[2]
        end
        current_balance[2*injector.bus_id-1] += status * current[1]
        current_balance[2*injector.bus_id]   += status * current[2]
        control_cursor += n_component_controls
    end

    @inbounds for k in 1:length(power_system.branch_injectors)
        branch = power_system.branch_injectors[k]
        v_from = voltages[branch.from_bus]
        v_to   = voltages[branch.to_bus]
        branch_angle = voltages[n_buses + k]
        i_from, i_to = compute_current_injection(branch.component, v_from, v_to,
                                                 branch_angle)
        status = statuses === nothing ? one(T) : T(statuses[n_single + k])
        if branch_currents !== nothing
            branch_currents[4*k-3] = status * i_from[1]
            branch_currents[4*k-2] = status * i_from[2]
            branch_currents[4*k-1] = status * i_to[1]
            branch_currents[4*k]   = status * i_to[2]
        end
        current_balance[2*branch.from_bus-1] += status * i_from[1]
        current_balance[2*branch.from_bus]   += status * i_from[2]
        current_balance[2*branch.to_bus-1]   += status * i_to[1]
        current_balance[2*branch.to_bus]     += status * i_to[2]
    end

    return current_balance
end

# ── Current balance (raw, no side effects) ──────────────────────────────────

function compute_current_balance(
    power_system::PowerSystem,
    voltages::AbstractVector{<:Real},
    controls::AbstractVector{<:Real};
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    T = promote_type(eltype(voltages), eltype(controls))
    current_balance = zeros(T, 2 * power_system.n_buses)
    _scatter_current_balance!(current_balance, power_system, voltages, controls,
                              statuses, nothing, nothing)
    return current_balance
end

# ── In-place current balance (writes into state.current_balance_buffer) ──────

function compute_current_balance!(state::PowerFlowState)
    current_balance_result = state.current_balance_buffer
    fill!(current_balance_result, zero(eltype(current_balance_result)))
    _scatter_current_balance!(
        current_balance_result, state.power_system, state.voltages, state.controls,
        state.statuses, state.single_bus_currents, state.branch_currents,
    )
    return current_balance_result
end

# ── Current balance (PowerFlowState dispatch, fills per-component currents) ──

function compute_current_balance(state::PowerFlowState)
    compute_current_balance!(state)
    return copy(state.current_balance_buffer)
end

# ── controls_from_voltages ───────────────────────────────────────────────────
#
# Given a voltage vector and an initial control vector, return a new control
# vector whose generator slots (SynchronousMachineStatic) are rewritten so that
# the RPF current balance is exactly zero at each generator bus. Non-generator
# controls (loads, etc.) are left untouched.
#
# The generator slot inversion uses compute_control_from_current_injection, so
# the returned controls produce a zero generator residual contribution at the
# given voltages under the RPF component equations — making (voltages, new
# controls) an RPF residual-zero point whenever the non-generator + branch
# contributions at each generator bus can be absorbed by a single generator.
#
# Assumes each generator bus hosts at most one SynchronousMachineStatic; if
# multiple gens sit on the same bus, only the first one at that bus absorbs
# the mismatch (arbitrary split). Generators with status 0 are skipped.

function controls_from_voltages(
    power_system::PowerSystem,
    voltages::AbstractVector{<:Real},
    controls::AbstractVector{<:Real};
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    T = promote_type(eltype(voltages), eltype(controls))
    n_single = length(power_system.single_bus_injectors)
    n_buses  = power_system.n_buses
    new_controls = Vector{T}(controls)

    # Zero out generator contributions in a temporary control vector so that
    # compute_current_balance returns the non-generator + branch mismatch at
    # each bus. For SynchronousMachineStatic, T_M=0 and V_ref=V yield i_d=i_q=0.
    temporary_controls = copy(new_controls)
    control_cursor = 1
    @inbounds for ii in 1:n_single
        injector = power_system.single_bus_injectors[ii]
        n_component_controls = n_controls(injector.component)
        if injector.component isa SynchronousMachineStatic
            temporary_controls[control_cursor]     = zero(T)
            temporary_controls[control_cursor + 1] = T(voltages[injector.bus_id])
        end
        control_cursor += n_component_controls
    end

    current_balance = compute_current_balance(
        power_system, voltages, temporary_controls; statuses,
    )

    bus_assigned = falses(n_buses)
    control_cursor = 1
    @inbounds for ii in 1:n_single
        injector = power_system.single_bus_injectors[ii]
        n_component_controls = n_controls(injector.component)
        if injector.component isa SynchronousMachineStatic
            status = statuses === nothing ? one(T) : T(statuses[ii])
            bus = injector.bus_id
            if status != zero(T) && !bus_assigned[bus]
                i_d_needed = -current_balance[2*bus - 1] / status
                i_q_needed = -current_balance[2*bus]     / status
                new_control = compute_control_from_current_injection(
                    injector.component, T(voltages[bus]), [i_d_needed, i_q_needed],
                )
                new_controls[control_cursor]     = new_control[1]
                new_controls[control_cursor + 1] = new_control[2]
                bus_assigned[bus] = true
            end
        end
        control_cursor += n_component_controls
    end

    return new_controls
end

function compute_cycle_balance(
    power_system::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
)
    if isempty(power_system.cycle_participation)
        T = promote_type(eltype(variables), eltype(controls))
        return zeros(T, 0)
    end
    cycle_balance_list = [
        variables[power_system.n_buses+1:end]' * cycle_participation for
        cycle_participation in power_system.cycle_participation
    ]
    return vcat(cycle_balance_list...)
end

function compute_residual(
    power_system::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real};
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    current_balance = compute_current_balance(power_system, variables, controls; statuses)
    cycle_balance =
        compute_cycle_balance(power_system, variables, controls) .*
        power_system.cycle_admittances
    return vcat(current_balance, cycle_balance)
end

function compute_energy(
    power_system::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real};
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    residuals = compute_residual(power_system, variables, controls; statuses)
    return 0.5 * sum(abs2, residuals)
end

# Ordering: z = [variables; controls]; H is (n_v + n_u) × (n_v + n_u).
# ExplicitAnalytical is the default — its method carries the no-method-arg default
# (energy_hessian.jl). DenseAD/SemiAnalytical/GaussNewtonApproximation are opt-in;
# DenseAD remains the ForwardDiff oracle that validates the analytical methods.
function compute_energy_hessian(
    power_system::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    ::DenseAD;
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    n_v = length(variables)
    z = vcat(variables, controls)
    return ForwardDiff.hessian(
        z -> compute_energy(power_system, z[1:n_v], z[n_v+1:end]; statuses),
        z,
    )
end

# --- PowerFlowState dispatch ---

# compute_residual(state) fills per-component current vectors as a side effect.
# Uses active topology (updated by update_topology!).
# Delegates to the in-place variant and returns a copy.
function compute_residual(state::PowerFlowState)
    r = compute_residual!(state)
    return copy(r)
end

# In-place: writes into state.residual_buffer, fills per-component currents.
# Uses active topology (updated by update_topology!).
function compute_residual!(state::PowerFlowState)
    ps = state.power_system
    current_balance_result = compute_current_balance!(state)
    r  = state.residual_buffer
    n_cb = length(current_balance_result)
    n_active = state.n_cycles_active[]
    n_single = length(ps.single_bus_injectors)

    @inbounds for i in 1:n_cb
        r[i] = current_balance_result[i]
    end

    n_buses = ps.n_buses
    branch_angles = view(state.voltages, n_buses+1:n_buses+ps.n_lines)
    @inbounds for k in 1:n_active
        participation = state.active_cycle_participation[k]
        cycle_sum = zero(eltype(r))
        for (j, p) in enumerate(participation)
            if p != 0
                cycle_sum += branch_angles[j] * p
            end
        end
        r[n_cb + k] = cycle_sum * state.active_cycle_admittances[k]
    end

    # Pin tripped branch angles to zero (prevents singular Jacobian).
    n_pins = 0
    @inbounds for k in 1:ps.n_lines
        if !state._topology_snapshot[k]
            n_pins += 1
            r[n_cb + n_active + n_pins] = branch_angles[k]
        end
    end

    n_res = n_cb + n_active + n_pins
    n_res == length(r) ? r : view(r, 1:n_res)
end

function compute_energy(state::PowerFlowState)
    r = compute_residual!(state)
    return 0.5 * sum(abs2, r)
end
