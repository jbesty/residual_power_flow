
using ForwardDiff

# ── Dense AD (default) ────────────────────────────────────────────────────────

function compute_jacobian_voltages(
    power_system::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    ::DenseAD = DenseAD();
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    return ForwardDiff.jacobian(v -> compute_residual(power_system, v, controls; statuses), variables)
end

function compute_jacobian_controls(
    power_system::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    ::DenseAD = DenseAD();
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    return ForwardDiff.jacobian(u -> compute_residual(power_system, variables, u; statuses), controls)
end

# ── Finite differences (central, reference) ───────────────────────────────────

function _central_diff_jacobian(f, x::AbstractVector{T}) where {T<:Real}
    h  = sqrt(eps(T))
    r0 = f(x)
    J  = zeros(T, length(r0), length(x))
    xp = copy(x)
    for j in eachindex(xp)
        xp[j] += h
        rp = f(xp)
        xp[j] -= 2h
        rm = f(xp)
        xp[j] += h
        J[:, j] = (rp .- rm) ./ (2h)
    end
    return J
end

function compute_jacobian_voltages(
    power_system::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    ::CentralDifferenceApproximation;
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    return _central_diff_jacobian(v -> compute_residual(power_system, v, controls; statuses), variables)
end

function compute_jacobian_controls(
    power_system::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    ::CentralDifferenceApproximation;
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    return _central_diff_jacobian(u -> compute_residual(power_system, variables, u; statuses), controls)
end

# ── ExplicitAnalytical — implemented in a separate file ───────────────────────

function compute_jacobian_voltages(
    power_system::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    method::ExplicitAnalytical;
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
    kwargs...,
)
    return _compute_jacobian_voltages_impl(power_system, variables, controls, method; statuses, kwargs...)
end

function compute_jacobian_controls(
    power_system::PowerSystem,
    variables::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    method::ExplicitAnalytical;
    statuses::Union{Nothing, AbstractVector{<:Real}} = nothing,
    kwargs...,
)
    return _compute_jacobian_controls_impl(power_system, variables, controls, method; statuses, kwargs...)
end
