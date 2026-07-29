import ForwardDiff

# ── Shared helper: J^T J block matrix from ExplicitAnalytical Jacobians ───────

function _jtj_from_explicit(
    ps::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    statuses,
    ::Type{T},
    n_v::Int,
    n_u::Int,
) where T
    J_v = Matrix{T}(compute_jacobian_voltages(ps, variables, controls, ExplicitAnalytical(); statuses))
    J_u = Matrix{T}(compute_jacobian_controls(ps, variables, controls, ExplicitAnalytical(); statuses))
    H = Matrix{T}(undef, n_v + n_u, n_v + n_u)
    H[1:n_v,     1:n_v]     = J_v' * J_v
    H[1:n_v,     n_v+1:end] = J_v' * J_u
    H[n_v+1:end, 1:n_v]     = J_u' * J_v
    H[n_v+1:end, n_v+1:end] = J_u' * J_u
    return H
end

# ── GaussNewtonApproximation: J^T J, no correction term ───────────────────────

function compute_energy_hessian(
    ps::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    ::GaussNewtonApproximation;
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    T = promote_type(eltype(variables), eltype(controls))
    return _jtj_from_explicit(ps, variables, controls, statuses, T, length(variables), length(controls))
end

# ── SemiAnalytical: J^T J + correction via ForwardDiff per-component ──────────

function compute_energy_hessian(
    ps::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    ::SemiAnalytical;
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    T = promote_type(eltype(variables), eltype(controls))
    n_v = length(variables)
    n_u = length(controls)
    H = _jtj_from_explicit(ps, variables, controls, statuses, T, n_v, n_u)
    r = compute_residual(ps, variables, controls; statuses)
    _add_correction_semianaly!(H, ps, variables, controls, statuses, r, n_v)
    return H
end

# ── ExplicitAnalytical: J^T J + fully analytical correction (default) ─────────

function compute_energy_hessian(
    ps::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    ::ExplicitAnalytical = ExplicitAnalytical();
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    T = promote_type(eltype(variables), eltype(controls))
    n_v = length(variables)
    n_u = length(controls)
    H = _jtj_from_explicit(ps, variables, controls, statuses, T, n_v, n_u)
    r = compute_residual(ps, variables, controls; statuses)
    _add_correction_explicit!(H, ps, variables, controls, statuses, r, n_v)
    return H
end

# ── SemiAnalytical correction: ForwardDiff on individual component functions ──

function _add_correction_semianaly!(
    H::Matrix,
    ps::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    statuses,
    r::AbstractVector{<:Real},
    n_v::Int,
)
    T = eltype(H)
    n_buses  = ps.n_buses
    n_single = length(ps.single_bus_injectors)

    ctrl_cursor = 1
    for ii in 1:n_single
        inj  = ps.single_bus_injectors[ii]
        nc   = n_controls(inj.component)
        bus  = inj.bus_id
        s    = statuses === nothing ? one(T) : T(statuses[ii])
        v_b  = T(variables[bus])
        u_loc = [T(controls[ctrl_cursor + j - 1]) for j in 1:nc]
        z_loc = vcat([v_b], u_loc)
        gcols = vcat([bus], [n_v + ctrl_cursor + j - 1 for j in 1:nc])
        comp  = inj.component
        r_d   = r[2*bus - 1]
        r_q   = r[2*bus]
        H_id = ForwardDiff.hessian(z -> s * compute_current_injection(comp, z[1], z[2:end])[1], z_loc)
        H_iq = ForwardDiff.hessian(z -> s * compute_current_injection(comp, z[1], z[2:end])[2], z_loc)
        for (li, gi) in enumerate(gcols), (lj, gj) in enumerate(gcols)
            H[gi, gj] += r_d * H_id[li, lj] + r_q * H_iq[li, lj]
        end
        ctrl_cursor += nc
    end

    for k in 1:length(ps.branch_injectors)
        br       = ps.branch_injectors[k]
        from_bus = br.from_bus
        to_bus   = br.to_bus
        col_θ    = n_buses + k
        s        = statuses === nothing ? one(T) : T(statuses[n_single + k])
        z_br     = [T(variables[from_bus]), T(variables[to_bus]), T(variables[col_θ])]
        gcols    = [from_bus, to_bus, col_θ]
        r_fre    = r[2*from_bus - 1]
        r_fim    = r[2*from_bus]
        r_tre    = r[2*to_bus - 1]
        r_tim    = r[2*to_bus]
        comp     = br.component
        H_fre = ForwardDiff.hessian(z -> s * compute_current_injection(comp, z[1], z[2], z[3])[1][1], z_br)
        H_fim = ForwardDiff.hessian(z -> s * compute_current_injection(comp, z[1], z[2], z[3])[1][2], z_br)
        H_tre = ForwardDiff.hessian(z -> s * compute_current_injection(comp, z[1], z[2], z[3])[2][1], z_br)
        H_tim = ForwardDiff.hessian(z -> s * compute_current_injection(comp, z[1], z[2], z[3])[2][2], z_br)
        for (li, gi) in enumerate(gcols), (lj, gj) in enumerate(gcols)
            H[gi, gj] += r_fre * H_fre[li, lj] + r_fim * H_fim[li, lj] +
                         r_tre * H_tre[li, lj] + r_tim * H_tim[li, lj]
        end
    end
end

# ── ExplicitAnalytical correction: hand-coded second derivatives ───────────────
#
# Derivation: H = ∑_i r_i ∇²r_i where r_i is residual element i and ∇² is w.r.t.
# z = [variables; controls]. Only nonzero blocks are populated; second derivatives
# of linear terms (Shunt, i_q of SynchronousMachineStatic, cycle balance) are zero.

function _add_correction_explicit!(
    H::Matrix,
    ps::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    statuses,
    r::AbstractVector{<:Real},
    n_v::Int,
)
    T = eltype(H)
    n_buses  = ps.n_buses
    n_single = length(ps.single_bus_injectors)

    ctrl_cursor = 1
    for ii in 1:n_single
        inj = ps.single_bus_injectors[ii]
        nc  = n_controls(inj.component)
        bus = inj.bus_id
        s   = statuses === nothing ? one(T) : T(statuses[ii])
        v   = T(variables[bus])
        u   = view(controls, ctrl_cursor:ctrl_cursor+nc-1)
        r_d = T(r[2*bus - 1])
        r_q = T(r[2*bus])
        _add_singlebus_correction!(H, inj.component, bus, ctrl_cursor, n_v, v, u, s, r_d, r_q)
        ctrl_cursor += nc
    end

    for k in 1:length(ps.branch_injectors)
        _add_branch_correction!(H, ps.branch_injectors[k], n_single, n_buses, k, variables, statuses, r)
    end
end

# SynchronousMachineStatic:
#   i_d = T_M/V  ⟹  ∂²i_d/∂V² = 2T_M/V³,  ∂²i_d/∂V∂T_M = −1/V²
#   i_q = K_DV*(V − V_ref)  ⟹  all second derivatives zero (linear)
function _add_singlebus_correction!(
    H::Matrix{T}, ::SynchronousMachineStatic, bus, cc, n_v, V, u, s, r_d, r_q,
) where T
    T_M   = T(u[1])
    tm_col = n_v + cc
    H[bus,    bus]    += r_d * s * 2 * T_M / V^3
    H[bus,    tm_col] += r_d * s * (-one(T) / V^2)
    H[tm_col, bus]    += r_d * s * (-one(T) / V^2)
end

# ZIPLoad:  f(V) = ZIP(V)/V = a/V + b/V₀ + c·V/V₀²
#   i_d = −s·P·f(V)  ⟹  ∂²i_d/∂V² = −s·P·f″,  ∂²i_d/∂V∂P = −s·f′
#   i_q =  s·Q·f(V)  ⟹  ∂²i_q/∂V² =  s·Q·f″,  ∂²i_q/∂V∂Q =  s·f′
function _add_singlebus_correction!(
    H::Matrix{T}, comp::ZIPLoad, bus, cc, n_v, V, u, s, r_d, r_q,
) where T
    P  = T(u[1]);  Q  = T(u[2])
    a  = T(comp.a);  c_zip = T(comp.c);  V0 = T(comp.V_0)
    f_pp = 2 * a / V^3
    f_p  = -a / V^2 + c_zip / V0^2
    p_col = n_v + cc;  q_col = n_v + cc + 1
    H[bus,   bus]   += r_d * s * (-P * f_pp) + r_q * s * (Q * f_pp)
    H[bus,   p_col] += r_d * s * (-f_p)
    H[p_col, bus]   += r_d * s * (-f_p)
    H[bus,   q_col] += r_q * s * f_p
    H[q_col, bus]   += r_q * s * f_p
end

# Shunt: current injection is linear in V, all second derivatives zero.
function _add_singlebus_correction!(H::Matrix, ::Shunt, bus, cc, n_v, V, u, s, r_d, r_q) end

# Branch / AsymmetricBranch:
# The only nonzero second derivatives are at (θ, θ), (V_to, θ), (V_from, θ) entries.
# From-end (i_from depends on V_from linearly, and on V_to and θ nonlinearly):
#   ∂²i_from_re/∂θ²      = s·V_to·(Y12r·cos θ − Y12i·sin θ)
#   ∂²i_from_re/∂V_to∂θ  = s·(Y12r·sin θ + Y12i·cos θ)
#   ∂²i_from_im/∂θ²      = s·V_to·(Y12i·cos θ + Y12r·sin θ)
#   ∂²i_from_im/∂V_to∂θ  = s·(Y12i·sin θ − Y12r·cos θ)
# To-end (i_to depends on V_to linearly, and on V_from and θ nonlinearly):
#   ∂²i_to_re/∂θ²        = s·V_from·(Y21r·cos θ + Y21i·sin θ)
#   ∂²i_to_re/∂V_from∂θ  = s·(Y21r·sin θ − Y21i·cos θ)
#   ∂²i_to_im/∂θ²        = s·V_from·(Y21i·cos θ − Y21r·sin θ)
#   ∂²i_to_im/∂V_from∂θ  = s·(Y21i·sin θ + Y21r·cos θ)
function _add_branch_correction!(
    H::Matrix{T}, br::BranchRecord, n_single, n_buses, k, variables, statuses, r,
) where T
    from_bus = br.from_bus;  to_bus = br.to_bus;  col_θ = n_buses + k
    s        = statuses === nothing ? one(T) : T(statuses[n_single + k])
    V_from   = T(variables[from_bus])
    V_to     = T(variables[to_bus])
    θ        = T(variables[col_θ])
    cos_θ, sin_θ = cos(θ), sin(θ)
    Y12r = T(real(br.component.Y_12));  Y12i = T(imag(br.component.Y_12))
    Y21r = T(real(br.component.Y_21));  Y21i = T(imag(br.component.Y_21))
    r_fre = T(r[2*from_bus - 1]);  r_fim = T(r[2*from_bus])
    r_tre = T(r[2*to_bus   - 1]);  r_tim = T(r[2*to_bus])

    d2_fre_θθ    = s * V_to   * (Y12r * cos_θ - Y12i * sin_θ)
    d2_fre_Vto_θ = s          * (Y12r * sin_θ + Y12i * cos_θ)
    d2_fim_θθ    = s * V_to   * (Y12i * cos_θ + Y12r * sin_θ)
    d2_fim_Vto_θ = s          * (Y12i * sin_θ - Y12r * cos_θ)

    d2_tre_θθ     = s * V_from * (Y21r * cos_θ + Y21i * sin_θ)
    d2_tre_Vfr_θ  = s          * (Y21r * sin_θ - Y21i * cos_θ)
    d2_tim_θθ     = s * V_from * (Y21i * cos_θ - Y21r * sin_θ)
    d2_tim_Vfr_θ  = s          * (Y21i * sin_θ + Y21r * cos_θ)

    H[col_θ,   col_θ]   += r_fre * d2_fre_θθ    + r_fim * d2_fim_θθ +
                            r_tre * d2_tre_θθ    + r_tim * d2_tim_θθ
    H[to_bus,  col_θ]   += r_fre * d2_fre_Vto_θ + r_fim * d2_fim_Vto_θ
    H[col_θ,   to_bus]  += r_fre * d2_fre_Vto_θ + r_fim * d2_fim_Vto_θ
    H[from_bus, col_θ]  += r_tre * d2_tre_Vfr_θ + r_tim * d2_tim_Vfr_θ
    H[col_θ,   from_bus]+= r_tre * d2_tre_Vfr_θ + r_tim * d2_tim_Vfr_θ
end
