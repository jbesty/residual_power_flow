
using SparseArrays
using StaticArrays: SMatrix

# ── Per-component voltage derivatives ∂I/∂V ──────────────────────────────────
# Returns (∂i_re/∂V, ∂i_im/∂V) at the component's own terminal voltage.

function _dI_dV(
    component::SynchronousMachineStatic,
    voltage::Real,
    control::AbstractVector{<:Real},
)
    T_M = control[1]
    return (-T_M / voltage^2, component.K_DV)
end

function _dI_dV(component::ZIPLoad, voltage::Real, control::AbstractVector{<:Real})
    P = control[1]
    Q = control[2]
    V0 = component.V_0
    ZIP     = component.a + component.b * (voltage / V0) + component.c * (voltage / V0)^2
    dZIP_dV = component.b / V0 + 2 * component.c * voltage / V0^2
    # i = [-P*ZIP/V, Q*ZIP/V]  ⟹  ∂i/∂V = [-P, Q] * ∂(ZIP/V)/∂V
    d_ZIP_over_V = (dZIP_dV * voltage - ZIP) / voltage^2
    return (-P * d_ZIP_over_V, Q * d_ZIP_over_V)
end

function _dI_dV(component::Shunt, voltage::Real, control::AbstractVector{<:Real})
    return (-component.G, -component.B)
end

# ── Per-component control derivatives ∂I/∂u ──────────────────────────────────
# Returns a 2×n_controls matrix of partial derivatives.

function _dI_du(
    component::SynchronousMachineStatic,
    voltage::Real,
    control::AbstractVector{<:Real},
)
    # control = [T_M, V_ref],  i = [T_M/V, -K_DV*(V_ref - V)]
    # ∂i/∂T_M = [1/V, 0],  ∂i/∂V_ref = [0, -K_DV]
    T = promote_type(typeof(voltage), typeof(component.K_DV))
    z = zero(T)
    return SMatrix{2,2,T}(1/voltage, z, z, -component.K_DV)
end

function _dI_du(component::ZIPLoad, voltage::Real, control::AbstractVector{<:Real})
    P  = control[1]
    Q  = control[2]
    V0 = component.V_0
    ZIP = component.a + component.b * (voltage / V0) + component.c * (voltage / V0)^2
    ZV  = ZIP / voltage
    # control = [P, Q],  i = [-P*ZIP/V, Q*ZIP/V]
    # ∂i/∂P = [-ZIP/V, 0],  ∂i/∂Q = [0, ZIP/V]
    z = zero(ZV)
    return SMatrix{2,2,typeof(ZV)}(-ZV, z, z, ZV)
end

function _dI_du(component::Shunt, voltage::Real, control::AbstractVector{<:Real})
    # Shunts have no controls
    return zeros(2, 0)
end

# ── Two-bus branch derivatives ────────────────────────────────────────────────
# Returns (∂I_re/∂V_t, ∂I_im/∂V_t, ∂I_re/∂V_o, ∂I_im/∂V_o, ∂I_re/∂θ, ∂I_im/∂θ)
# for a branch element.
#   direction d ∈ {±1}
#   Y_self = Gs + j·Bs  (Y_11 for from-end d=+1, Y_22 for to-end d=-1)
#   Y_cross = Gc + j·Bc  (Y_12 for from-end, Y_21 for to-end)
#   V_o: other-bus voltage magnitude  (V_to for from-end, V_from for to-end)
#   θ: branch angle variable (θ_to - θ_from)

function _dI_dV_branch(Y_self::Complex, Y_cross::Complex, V_o::Real, θ::Real, d::Int)
    Gs, Bs = real(Y_self), imag(Y_self)
    Gc, Bc = real(Y_cross), imag(Y_cross)
    dθ = d * θ
    cos_dθ, sin_dθ = cos(dθ), sin(dθ)

    dIre_dVt = -Gs
    dIim_dVt = -Bs
    dIre_dVo = -Gc * cos_dθ + Bc * sin_dθ
    dIim_dVo = -Bc * cos_dθ - Gc * sin_dθ
    dIre_dθ  = d * ( Gc * V_o * sin_dθ + Bc * V_o * cos_dθ)
    dIim_dθ  = d * ( Bc * V_o * sin_dθ - Gc * V_o * cos_dθ)

    return dIre_dVt, dIim_dVt, dIre_dVo, dIim_dVo, dIre_dθ, dIim_dθ
end

@inline function _push_nonzero!(rows, cols, entries, row, col, entry)
    if entry != zero(entry)
        push!(rows, row); push!(cols, col); push!(entries, entry)
    end
end

# ── Voltage Jacobian ─────────────────────────────────────────────────────────

function _compute_jacobian_voltages_impl(
    ps::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    ::ExplicitAnalytical;
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
    cycle_participation::Vector{Vector{Int}} = ps.cycle_participation,
    cycle_admittances::AbstractVector{<:Real} = ps.cycle_admittances,
    n_cycles::Int = ps.n_cycles,
    _topology_snapshot::Union{Nothing, Vector{Bool}} = nothing,
    _voltage_jacobian_cache::Union{Nothing, Ref} = nothing,
    kwargs...,  # absorb remaining cache kwargs
)
    # Try in-place fill into cached Jacobian
    if _voltage_jacobian_cache !== nothing && _voltage_jacobian_cache[] !== nothing
        J = _voltage_jacobian_cache[]::SparseMatrixCSC{eltype(variables), Int}
        _fill_voltage_jacobian!(J, ps, variables, controls, statuses,
            cycle_participation, cycle_admittances, n_cycles, _topology_snapshot)
        return J
    end

    # Cold path: build from COO format
    J = _build_voltage_jacobian_coo(ps, variables, controls, statuses,
        cycle_participation, cycle_admittances, n_cycles, _topology_snapshot)

    # Cache the structure for reuse
    if _voltage_jacobian_cache !== nothing
        _voltage_jacobian_cache[] = J
    end

    return J
end

function _build_voltage_jacobian_coo(ps, variables, controls, statuses,
    cycle_participation, cycle_admittances, n_cycles, _topology_snapshot)
    T       = eltype(variables)
    n_buses = ps.n_buses
    n_tripped = _topology_snapshot === nothing ? 0 : count(!, _topology_snapshot)
    n_res   = 2 * n_buses + n_cycles + n_tripped
    n_var   = length(variables)

    n_single   = length(ps.single_bus_injectors)
    n_branches = length(ps.branch_injectors)
    est_nnz    = 2 * n_single + 6 * 2 * n_branches + sum(length, cycle_participation; init=0) + n_tripped
    rows = sizehint!(Int[], est_nnz)
    cols = sizehint!(Int[], est_nnz)
    entries = sizehint!(T[],   est_nnz)

    ctrl_cursor = 1
    for ii in 1:n_single
        inj   = ps.single_bus_injectors[ii]
        nc    = n_controls(inj.component)
        bus   = inj.bus_id
        v_bus = variables[bus]
        u_loc = view(controls, ctrl_cursor:ctrl_cursor+nc-1)

        dIre_dV, dIim_dV = _dI_dV(inj.component, v_bus, u_loc)
        status = statuses === nothing ? one(T) : T(statuses[ii])

        _push_nonzero!(rows, cols, entries, 2*bus-1, bus, status * dIre_dV)
        _push_nonzero!(rows, cols, entries, 2*bus,   bus, status * dIim_dV)
        ctrl_cursor += nc
    end

    for k in 1:n_branches
        br       = ps.branch_injectors[k]
        from_bus = br.from_bus
        to_bus   = br.to_bus
        col_θ    = n_buses + k

        V_from = variables[from_bus]
        V_to   = variables[to_bus]
        θ      = variables[col_θ]

        status = statuses === nothing ? one(T) : T(statuses[n_single + k])

        dIre_dVt, dIim_dVt, dIre_dVo, dIim_dVo, dIre_dθ, dIim_dθ =
            _dI_dV_branch(br.component.Y_11, br.component.Y_12, V_to, θ, 1)
        row_re, row_im = 2*from_bus-1, 2*from_bus
        _push_nonzero!(rows, cols, entries, row_re, from_bus, status * dIre_dVt)
        _push_nonzero!(rows, cols, entries, row_im, from_bus, status * dIim_dVt)
        _push_nonzero!(rows, cols, entries, row_re, to_bus,   status * dIre_dVo)
        _push_nonzero!(rows, cols, entries, row_im, to_bus,   status * dIim_dVo)
        _push_nonzero!(rows, cols, entries, row_re, col_θ,    status * dIre_dθ)
        _push_nonzero!(rows, cols, entries, row_im, col_θ,    status * dIim_dθ)

        dIre_dVt2, dIim_dVt2, dIre_dVo2, dIim_dVo2, dIre_dθ2, dIim_dθ2 =
            _dI_dV_branch(br.component.Y_22, br.component.Y_21, V_from, θ, -1)
        row_re2, row_im2 = 2*to_bus-1, 2*to_bus
        _push_nonzero!(rows, cols, entries, row_re2, to_bus,   status * dIre_dVt2)
        _push_nonzero!(rows, cols, entries, row_im2, to_bus,   status * dIim_dVt2)
        _push_nonzero!(rows, cols, entries, row_re2, from_bus, status * dIre_dVo2)
        _push_nonzero!(rows, cols, entries, row_im2, from_bus, status * dIim_dVo2)
        _push_nonzero!(rows, cols, entries, row_re2, col_θ,    status * dIre_dθ2)
        _push_nonzero!(rows, cols, entries, row_im2, col_θ,    status * dIim_dθ2)
    end

    for (k, participation) in enumerate(cycle_participation)
        adm = cycle_admittances[k]
        row = 2n_buses + k
        for (j, p) in enumerate(participation)
            col = n_buses + j
            entry = p * adm
            if entry != zero(T)
                push!(rows, row); push!(cols, col); push!(entries, entry)
            end
        end
    end

    if _topology_snapshot !== nothing
        pin_idx = 0
        for k in 1:n_branches
            if !_topology_snapshot[k]
                pin_idx += 1
                push!(rows, 2n_buses + n_cycles + pin_idx)
                push!(cols, n_buses + k)
                push!(entries, one(T))
            end
        end
    end

    return sparse(rows, cols, entries, n_res, n_var)
end

# In-place fill: zeros nzval, then writes entries using indexed assignment.
function _fill_voltage_jacobian!(J::SparseMatrixCSC, ps, variables, controls, statuses,
    cycle_participation, cycle_admittances, n_cycles, _topology_snapshot)
    T       = eltype(J)
    n_buses = ps.n_buses
    n_single   = length(ps.single_bus_injectors)
    n_branches = length(ps.branch_injectors)

    fill!(nonzeros(J), zero(T))

    ctrl_cursor = 1
    @inbounds for ii in 1:n_single
        inj   = ps.single_bus_injectors[ii]
        nc    = n_controls(inj.component)
        bus   = inj.bus_id
        v_bus = variables[bus]
        u_loc = view(controls, ctrl_cursor:ctrl_cursor+nc-1)

        dIre_dV, dIim_dV = _dI_dV(inj.component, v_bus, u_loc)
        status = statuses === nothing ? one(T) : T(statuses[ii])

        J[2*bus-1, bus] += status * dIre_dV
        J[2*bus,   bus] += status * dIim_dV
        ctrl_cursor += nc
    end

    @inbounds for k in 1:n_branches
        br       = ps.branch_injectors[k]
        from_bus = br.from_bus
        to_bus   = br.to_bus
        col_θ    = n_buses + k

        V_from = variables[from_bus]
        V_to   = variables[to_bus]
        θ      = variables[col_θ]

        status = statuses === nothing ? one(T) : T(statuses[n_single + k])

        dIre_dVt, dIim_dVt, dIre_dVo, dIim_dVo, dIre_dθ, dIim_dθ =
            _dI_dV_branch(br.component.Y_11, br.component.Y_12, V_to, θ, 1)
        J[2*from_bus-1, from_bus] += status * dIre_dVt
        J[2*from_bus,   from_bus] += status * dIim_dVt
        J[2*from_bus-1, to_bus]   += status * dIre_dVo
        J[2*from_bus,   to_bus]   += status * dIim_dVo
        J[2*from_bus-1, col_θ]    += status * dIre_dθ
        J[2*from_bus,   col_θ]    += status * dIim_dθ

        dIre_dVt2, dIim_dVt2, dIre_dVo2, dIim_dVo2, dIre_dθ2, dIim_dθ2 =
            _dI_dV_branch(br.component.Y_22, br.component.Y_21, V_from, θ, -1)
        J[2*to_bus-1, to_bus]   += status * dIre_dVt2
        J[2*to_bus,   to_bus]   += status * dIim_dVt2
        J[2*to_bus-1, from_bus] += status * dIre_dVo2
        J[2*to_bus,   from_bus] += status * dIim_dVo2
        J[2*to_bus-1, col_θ]    += status * dIre_dθ2
        J[2*to_bus,   col_θ]    += status * dIim_dθ2
    end

    @inbounds for (k, participation) in enumerate(cycle_participation)
        adm = cycle_admittances[k]
        row = 2n_buses + k
        for (j, p) in enumerate(participation)
            if p != 0
                J[row, n_buses + j] = T(p * adm)
            end
        end
    end

    if _topology_snapshot !== nothing
        pin_idx = 0
        for k in 1:n_branches
            if !_topology_snapshot[k]
                pin_idx += 1
                J[2n_buses + n_cycles + pin_idx, n_buses + k] = one(T)
            end
        end
    end

    return J
end

# ── Control Jacobian ──────────────────────────────────────────────────────────

function _compute_jacobian_controls_impl(
    ps::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    ::ExplicitAnalytical;
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
    n_cycles::Int = ps.n_cycles,
    _topology_snapshot::Union{Nothing, Vector{Bool}} = nothing,
    _control_jacobian_cache::Union{Nothing, Ref} = nothing,
    kwargs...,  # absorb remaining cache kwargs
)
    T       = eltype(variables)
    n_buses = ps.n_buses
    n_tripped = _topology_snapshot === nothing ? 0 : count(!, _topology_snapshot)
    n_res   = 2 * n_buses + n_cycles + n_tripped
    n_ctrl  = length(controls)
    n_single = length(ps.single_bus_injectors)

    # Try in-place fill into cached Jacobian
    if _control_jacobian_cache !== nothing && _control_jacobian_cache[] !== nothing
        J = _control_jacobian_cache[]::SparseMatrixCSC{T, Int}
        fill!(nonzeros(J), zero(T))
        ctrl_cursor = 1
        @inbounds for ii in 1:n_single
            inj = ps.single_bus_injectors[ii]
            nc  = n_controls(inj.component)
            nc == 0 && (ctrl_cursor += nc; continue)
            bus   = inj.bus_id
            v_bus = variables[bus]
            u_loc = view(controls, ctrl_cursor:ctrl_cursor+nc-1)
            dI_du = _dI_du(inj.component, v_bus, u_loc)
            s = statuses === nothing ? one(T) : T(statuses[ii])
            for jj in 1:nc
                global_col = ctrl_cursor + jj - 1
                J[2*bus-1, global_col] += status * dI_du[1, jj]
                J[2*bus,   global_col] += status * dI_du[2, jj]
            end
            ctrl_cursor += nc
        end
        return J
    end

    # Cold path: build from COO format
    est_nnz  = 2 * n_single * 2
    rows = sizehint!(Int[], est_nnz)
    cols = sizehint!(Int[], est_nnz)
    entries = sizehint!(T[],   est_nnz)

    ctrl_cursor = 1
    for ii in 1:n_single
        inj  = ps.single_bus_injectors[ii]
        nc   = n_controls(inj.component)
        if nc == 0
            ctrl_cursor += nc
            continue
        end
        bus   = inj.bus_id
        v_bus = variables[bus]
        u_loc = view(controls, ctrl_cursor:ctrl_cursor+nc-1)

        dI_du = _dI_du(inj.component, v_bus, u_loc)
        status = statuses === nothing ? one(T) : T(statuses[ii])

        row_re = 2*bus - 1
        row_im = 2*bus
        for jj in 1:nc
            global_col = ctrl_cursor + jj - 1
            _push_nonzero!(rows, cols, entries, row_re, global_col, status * dI_du[1, jj])
            _push_nonzero!(rows, cols, entries, row_im, global_col, status * dI_du[2, jj])
        end
        ctrl_cursor += nc
    end

    J = sparse(rows, cols, entries, n_res, n_ctrl)
    _control_jacobian_cache !== nothing && (_control_jacobian_cache[] = J)
    return J
end
