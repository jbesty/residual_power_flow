
export DataTransformation, refactor!
export normalize_controls, denormalize_controls
export normalize_voltages, denormalize_voltages
export nondimensional_loss
export TransformationCharacterisation

using LinearAlgebra
using LinearAlgebra: eigvals, svdvals, Symmetric, Cholesky, cholesky
using Statistics: mean, std
import Printf

# --- DataTransformation ---
#
# Stores affine maps between dimensionless and dimensionful (physical) space:
#
#   dimensionful  = A * dimensionless + b       (A: non-dim → physical)
#   dimensionless = A⁻¹ * (dimensionful - b)
#
# This matches the paper convention (Stiasny et al., Eqs. 23a–23b): y = A_y ỹ + b_y.
# The paper's Eq. 24a gives A_v^T A_v = W_v^{-1}; the corresponding controls relation
# is A_u^T A_u = W_x^{-1},
# which is the inverse of the literal Eq. 24b in the published paper. Training a
# neural solver with unweighted MSE in dimensionless space is equivalent to
# minimising a physically-informed weighted MSE in the original space, with
# voltage weight W_v = (A_v A_v^T)⁻¹. Three schemes determine how A is
# constructed:
#
#   :identity          — A = I (no transformation)
#   :standardised      — A = diag(σ), giving W = diag(1/σ²) when symmetric
#   :physics_informed  — voltages: A = H_vv^{-1/2},  giving W_v = H_vv
#                        controls: A = W_x^{-1/2},   giving ||nondimensional_controls||² = (u-b)^T W_x (u-b)
#                                                    (backward-error magnitude)
#
# The default (controls: :standardised, voltages: :physics_informed) is the S-H
# scheme from the paper, which performs best across dataset sizes and widths.
#
# Cholesky factorisations of A_controls and A_voltages are cached so that
# normalize_* can avoid a dense inverse; the factorisation is numerically
# preferred over storing A⁻¹ explicitly, especially when A is ill-conditioned.
# Mutable so fit!(::NeuralSolver, dataset, ...) can refresh fields in place.

mutable struct DataTransformation{T<:AbstractFloat}
    A_controls      :: Matrix{T}              # n_controls × n_controls
    A_controls_fact :: Cholesky{T, Matrix{T}}
    b_controls      :: Vector{T}              # n_controls
    A_voltages      :: Matrix{T}              # n_variables × n_variables
    A_voltages_fact :: Cholesky{T, Matrix{T}}
    b_voltages      :: Vector{T}              # n_variables
end

# Public constructor: takes A and b for each side, builds Cholesky factorisations.
function DataTransformation(
    A_controls::AbstractMatrix{T},
    b_controls::AbstractVector,
    A_voltages::AbstractMatrix{T},
    b_voltages::AbstractVector,
) where {T<:AbstractFloat}
    A_u = Matrix{T}(A_controls)
    A_v = Matrix{T}(A_voltages)
    return DataTransformation{T}(
        A_u,
        cholesky(Symmetric((A_u + A_u') / 2)),
        Vector{T}(b_controls),
        A_v,
        cholesky(Symmetric((A_v + A_v') / 2)),
        Vector{T}(b_voltages),
    )
end

# Rebuild the cached Cholesky factorisations from the current A_controls and
# A_voltages. Call after in-place mutation of the stored matrices.
function refactor!(t::DataTransformation{T}) where {T}
    t.A_controls_fact = cholesky(Symmetric((t.A_controls + t.A_controls') / 2))
    t.A_voltages_fact = cholesky(Symmetric((t.A_voltages + t.A_voltages') / 2))
    return t
end

# --- Transforms: physical ↔ non-dimensional ---
#
# normalize_controls(t, u) = A_u⁻¹(u − b_u)  →  nondimensional_controls
# normalize_voltages(t, v) = A_v⁻¹(v − b_v)  →  nondimensional_voltages
#
# Vector (single sample) and matrix (N × m, rows are samples) dispatch.

normalize_controls(t::DataTransformation, u::AbstractVector) =
    t.A_controls_fact \ (u .- t.b_controls)

normalize_voltages(t::DataTransformation, v::AbstractVector) =
    t.A_voltages_fact \ (v .- t.b_voltages)

denormalize_controls(t::DataTransformation, nondimensional_controls::AbstractVector) =
    t.A_controls * nondimensional_controls .+ t.b_controls

denormalize_voltages(t::DataTransformation, nondimensional_voltages::AbstractVector) =
    t.A_voltages * nondimensional_voltages .+ t.b_voltages

normalize_controls(t::DataTransformation, U::AbstractMatrix) =
    (U .- t.b_controls') / t.A_controls_fact'

normalize_voltages(t::DataTransformation, V::AbstractMatrix) =
    (V .- t.b_voltages') / t.A_voltages_fact'

denormalize_controls(t::DataTransformation, nondimensional_controls::AbstractMatrix) =
    nondimensional_controls * t.A_controls' .+ t.b_controls'

denormalize_voltages(t::DataTransformation, nondimensional_voltages::AbstractMatrix) =
    nondimensional_voltages * t.A_voltages' .+ t.b_voltages'

# fit_data_transformation is defined in
# ext/NeuralSolverTrainingExt/fit_data_transformation.jl — construction-side
# code moves to the training extension; the struct and inference-side methods
# (normalize, denormalize, nondimensional_loss, characterise) stay here.

# --- Per-sample weighted loss in physical space ---
#
# nondimensional_loss(t, y_pred, y_true) computes per-sample ||normalize_voltages(ŷ) - normalize_voltages(y)||²/2,
# i.e. the non-dimensional squared error — the same quantity the model minimises during training.
# Inputs may be vectors (single sample) or N × n_variables matrices (rows are samples).

function nondimensional_loss(
    t::DataTransformation,
    y_pred::AbstractVector,
    y_true::AbstractVector,
)
    return sum(abs2, normalize_voltages(t, y_pred) .- normalize_voltages(t, y_true)) / 2
end

function nondimensional_loss(
    t::DataTransformation,
    y_pred::AbstractMatrix,
    y_true::AbstractMatrix,
)
    return vec(sum(abs2, normalize_voltages(t, y_pred) .- normalize_voltages(t, y_true); dims=2)) ./ 2
end

# --- Transformation characterisation ---

struct TransformationCharacterisation{T<:AbstractFloat}
    eigenvalues_controls::Vector{T}
    eigenvalues_voltages::Vector{T}
    condition_number_controls::T
    condition_number_voltages::T
    diagonal_scaling_controls::Vector{T}   # diag(A_controls)
    diagonal_scaling_voltages::Vector{T}   # diag(A_voltages)
    physical_controls_stats::Union{Nothing, Matrix{T}}     # n_controls × 3 (mean, min, max)
    physical_voltages_stats::Union{Nothing, Matrix{T}}     # n_variables × 3 (mean, min, max)
    normalised_controls_stats::Union{Nothing, Matrix{T}}   # n_controls × 4 (mean, std, min, max)
    normalised_voltages_stats::Union{Nothing, Matrix{T}}   # n_variables × 4 (mean, std, min, max)
end

function characterise(t::DataTransformation{T}) where {T}
    eig_c = eigvals(Symmetric(t.A_controls' * t.A_controls))
    eig_v = eigvals(Symmetric(t.A_voltages' * t.A_voltages))

    cond_c = _safe_condition_number(t.A_controls)
    cond_v = _safe_condition_number(t.A_voltages)

    diag_c = T[t.A_controls[i, i] for i in 1:size(t.A_controls, 1)]
    diag_v = T[t.A_voltages[i, i] for i in 1:size(t.A_voltages, 1)]

    return TransformationCharacterisation{T}(
        eig_c, eig_v, cond_c, cond_v, diag_c, diag_v,
        nothing, nothing, nothing, nothing,
    )
end

function characterise(t::DataTransformation{T}, dataset::PowerFlowDataset) where {T}
    base = characterise(t)

    controls_raw = controls_matrix(dataset)
    voltages_raw = voltages_matrix(dataset)
    controls_norm = normalize_controls(t, controls_raw)
    voltages_norm = normalize_voltages(t, voltages_raw)

    c_phys = _column_summary(controls_raw)
    v_phys = _column_summary(voltages_raw)
    c_stats = _column_stats(controls_norm)
    v_stats = _column_stats(voltages_norm)

    return TransformationCharacterisation{T}(
        base.eigenvalues_controls, base.eigenvalues_voltages,
        base.condition_number_controls, base.condition_number_voltages,
        base.diagonal_scaling_controls, base.diagonal_scaling_voltages,
        T.(c_phys), T.(v_phys), T.(c_stats), T.(v_stats),
    )
end

function _safe_condition_number(A::AbstractMatrix{T}) where {T}
    sv = svdvals(A)
    smin = minimum(sv)
    smin < eps(T) && return T(Inf)
    return T(maximum(sv) / smin)
end

function _column_stats(X::AbstractMatrix)
    n = size(X, 2)
    stats = Matrix{Float64}(undef, n, 4)
    for j in 1:n
        col = @view X[:, j]
        stats[j, 1] = mean(col)
        stats[j, 2] = std(col)
        stats[j, 3] = minimum(col)
        stats[j, 4] = maximum(col)
    end
    return stats
end

function _column_summary(X::AbstractMatrix)
    n = size(X, 2)
    stats = Matrix{Float64}(undef, n, 3)
    for j in 1:n
        col = @view X[:, j]
        stats[j, 1] = mean(col)
        stats[j, 2] = minimum(col)
        stats[j, 3] = maximum(col)
    end
    return stats
end

function Base.show(io::IO, ::MIME"text/plain", c::TransformationCharacterisation)
    println(io, "TransformationCharacterisation")
    Printf.@printf(io, "  Controls:  cond=%.2g  eig=[%.2g, %.2g]\n",
        c.condition_number_controls,
        minimum(c.eigenvalues_controls), maximum(c.eigenvalues_controls))
    Printf.@printf(io, "  Voltages:  cond=%.2g  eig=[%.2g, %.2g]\n",
        c.condition_number_voltages,
        minimum(c.eigenvalues_voltages), maximum(c.eigenvalues_voltages))
    if c.physical_controls_stats !== nothing
        println(io, "  Physical controls (mean [min, max]):")
        for j in axes(c.physical_controls_stats, 1)
            row = c.physical_controls_stats[j, :]
            Printf.@printf(io, "    [%2d] %+.3f [%.3f, %.3f]\n",
                j, row[1], row[2], row[3])
        end
        println(io, "  Physical voltages (mean [min, max]):")
        for j in axes(c.physical_voltages_stats, 1)
            row = c.physical_voltages_stats[j, :]
            Printf.@printf(io, "    [%2d] %+.3f [%.3f, %.3f]\n",
                j, row[1], row[2], row[3])
        end
    end
    if c.normalised_controls_stats !== nothing
        println(io, "  Normalised controls (mean, std [min, max]):")
        for j in axes(c.normalised_controls_stats, 1)
            row = c.normalised_controls_stats[j, :]
            Printf.@printf(io, "    [%2d] %+.2f  std=%.2f [%.2f, %.2f]\n",
                j, row[1], row[2], row[3], row[4])
        end
        println(io, "  Normalised voltages (mean, std [min, max]):")
        for j in axes(c.normalised_voltages_stats, 1)
            row = c.normalised_voltages_stats[j, :]
            Printf.@printf(io, "    [%2d] %+.2f  std=%.2f [%.2f, %.2f]\n",
                j, row[1], row[2], row[3], row[4])
        end
    end
end

