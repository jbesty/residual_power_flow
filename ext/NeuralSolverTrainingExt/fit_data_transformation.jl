# --- DataTransformation factory ---
#
# Construction-side fit for DataTransformation (A, b for controls and voltages).
# The struct and inference-side methods live in src/learning/normalisation.jl;
# this file defines only the fitting procedure and the scheme builders.
#
# Physics-informed schemes use compute_energy_hessian from src/core, so the
# Hessian is a core function of the power system (not a training artefact).

"""
    fit_data_transformation(dataset, power_system; controls_scheme, voltages_scheme, ...)

Compute affine normalization maps from training data.

Schemes: `:identity`, `:standardised`, `:physics_informed`.
Default S-H (standardised controls, physics-informed voltages) follows the paper's
recommendation for robustness across dataset sizes.
"""
function fit_data_transformation(
    dataset::PowerFlowDataset,
    power_system::PowerSystem;
    controls_scheme::Symbol = :standardised,
    voltages_scheme::Symbol = :physics_informed,
    n_hessian_samples::Int = 100,
    rng::AbstractRNG = default_rng(),
)
    controls = controls_matrix(dataset)   # N × n_controls
    voltages = voltages_matrix(dataset)   # N × n_variables
    T = eltype(controls)

    b_controls = vec(mean(controls, dims=1))
    b_voltages = vec(mean(voltages, dims=1))

    needs_hessian = controls_scheme == :physics_informed || voltages_scheme == :physics_informed

    H_vv = nothing
    W_controls = nothing
    if needs_hessian
        H_vv_raw, W_controls_raw = _compute_hessian_blocks(dataset, power_system, n_hessian_samples, rng)
        H_vv = T.(H_vv_raw)
        W_controls = T.(W_controls_raw)
    end

    A_voltages = if voltages_scheme == :physics_informed
        # A_v = H_vv^{-1/2}: paper Eq. 24a A_v^T A_v = W_v^{-1} with W_v = H_vv.
        _symmetric_inv_sqrt(H_vv)
    else
        _build_diagonal_transform(voltages_scheme, voltages, T)
    end

    A_controls = if controls_scheme == :physics_informed
        # A_controls = W_controls^{-1/2}: matches voltage-side convention so that
        # ||normalize_controls(u)||^2 = (u - b_controls)' W_controls (u - b_controls),
        # the backward-error magnitude. See docs/formulation/main.tex eq:Au-construction.
        _symmetric_inv_sqrt(W_controls)
    else
        _build_diagonal_transform(controls_scheme, controls, T)
    end

    t = DataTransformation(
        Matrix{T}(A_controls), T.(b_controls),
        Matrix{T}(A_voltages), T.(b_voltages),
    )

    return t
end

# --- Scheme builders ---

function _build_diagonal_transform(scheme::Symbol, data::AbstractMatrix{T}, ::Type{T}) where {T}
    if scheme == :identity
        return Matrix{T}(I, size(data, 2), size(data, 2))
    elseif scheme == :standardised
        # A = diag(σ) maps non-dim → physical; the non-dim variable has unit
        # variance, the physical variable has standard deviation σ.
        σ = vec(std(data, dims=1))
        σ[σ .< eps(T)] .= one(T)   # guard against zero-variance columns
        return Matrix{T}(Diagonal(σ))
    else
        error("Unknown scheme :$scheme. Valid choices: :identity, :standardised, :physics_informed.")
    end
end

# Symmetric positive-definite inverse square root via eigendecomposition.
# Small or negative eigenvalues are clamped to avoid blowup on ill-conditioned Hessians.
function _symmetric_inv_sqrt(M::AbstractMatrix{T}) where {T}
    F = eigen(Symmetric(M))
    clamped = max.(F.values, eps(T))
    inv_sqrtvals = one(T) ./ sqrt.(clamped)
    return Matrix{T}(F.vectors * Diagonal(inv_sqrtvals) * F.vectors')
end

