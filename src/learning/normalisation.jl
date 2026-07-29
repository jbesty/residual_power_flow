
export DataTransformation
export normalize_controls, denormalize_controls
export normalize_voltages, denormalize_voltages
export nondimensional_loss

using LinearAlgebra
using LinearAlgebra: Symmetric, Cholesky, cholesky

# --- DataTransformation ---
#
# Stores affine maps between dimensionless and dimensionful (physical) space:
#
#   dimensionful  = A * dimensionless + b       (A: non-dim → physical)
#   dimensionless = A⁻¹ * (dimensionful - b)
#
# This matches the paper's affine transforms (Stiasny et al.): y = A_y ỹ + b_y and
# x = A_x x̃ + b_x, with the linear maps chosen from A_x^T A_x = W_x^{-1} and
# A_y^T A_y = W_y^{-1} (the biases b are the empirical training-data means). Here
# A_u is A_x (controls) and A_v is A_y (voltages), so A_u^T A_u = W_x^{-1} and
# A_v^T A_v = W_v^{-1} = H_vv^{-1}, exactly the paper's relations. Training a
# neural solver with unweighted MSE in dimensionless space is then equivalent to
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
# The default (controls: :standardised, voltages: :physics_informed) is a library
# convenience, NOT one of the paper's schemes. The paper compares :identity,
# :standardised, and the proposed :physics_informed applied to BOTH sides (see the
# experiment configs and the fit_data_transformation docstring); the physics-informed
# non-dimensionalisation is the paper's contribution.
#
# Cholesky factorisations of A_controls and A_voltages are cached so that
# normalize_* can avoid a dense inverse; the factorisation is numerically
# preferred over storing A⁻¹ explicitly, especially when A is ill-conditioned.

struct DataTransformation{T<:AbstractFloat}
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

# --- Transforms: physical ↔ non-dimensional ---
#
# normalize_controls(transformation, u) = A_u⁻¹(u − b_u)  →  nondimensional_controls
# normalize_voltages(transformation, v) = A_v⁻¹(v − b_v)  →  nondimensional_voltages
#
# Vector (single sample) and matrix (N × m, rows are samples) dispatch.

normalize_controls(transformation::DataTransformation, u::AbstractVector) =
    transformation.A_controls_fact \ (u .- transformation.b_controls)

normalize_voltages(transformation::DataTransformation, v::AbstractVector) =
    transformation.A_voltages_fact \ (v .- transformation.b_voltages)

denormalize_controls(transformation::DataTransformation, nondimensional_controls::AbstractVector) =
    transformation.A_controls * nondimensional_controls .+ transformation.b_controls

denormalize_voltages(transformation::DataTransformation, nondimensional_voltages::AbstractVector) =
    transformation.A_voltages * nondimensional_voltages .+ transformation.b_voltages

normalize_controls(transformation::DataTransformation, U::AbstractMatrix) =
    (U .- transformation.b_controls') / transformation.A_controls_fact'

normalize_voltages(transformation::DataTransformation, V::AbstractMatrix) =
    (V .- transformation.b_voltages') / transformation.A_voltages_fact'

denormalize_controls(transformation::DataTransformation, nondimensional_controls::AbstractMatrix) =
    nondimensional_controls * transformation.A_controls' .+ transformation.b_controls'

denormalize_voltages(transformation::DataTransformation, nondimensional_voltages::AbstractMatrix) =
    nondimensional_voltages * transformation.A_voltages' .+ transformation.b_voltages'

# fit_data_transformation is defined in
# ext/NeuralSolverTrainingExt/fit_data_transformation.jl — construction-side
# code moves to the training extension; the struct and inference-side methods
# (normalize, denormalize, nondimensional_loss) stay here.

# --- Per-sample weighted loss in physical space ---
#
# nondimensional_loss(transformation, y_pred, y_true) computes per-sample ||normalize_voltages(ŷ) - normalize_voltages(y)||²/2,
# i.e. the non-dimensional squared error — the same quantity the model minimises during training.
# Inputs may be vectors (single sample) or N × n_variables matrices (rows are samples).

function nondimensional_loss(
    transformation::DataTransformation,
    y_pred::AbstractVector,
    y_true::AbstractVector,
)
    return sum(abs2, normalize_voltages(transformation, y_pred) .- normalize_voltages(transformation, y_true)) / 2
end

function nondimensional_loss(
    transformation::DataTransformation,
    y_pred::AbstractMatrix,
    y_true::AbstractMatrix,
)
    return vec(sum(abs2, normalize_voltages(transformation, y_pred) .- normalize_voltages(transformation, y_true); dims=2)) ./ 2
end


