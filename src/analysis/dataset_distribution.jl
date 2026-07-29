
export dataset_distribution, DatasetDistribution

import LinearAlgebra
import LinearAlgebra: logdet, Symmetric, eigen, Diagonal
import Random: AbstractRNG, default_rng, shuffle
import Printf
import Statistics: mean

# ── Hessian helpers ───────────────────────────────────────────────────────────
#
# Used by dataset_distribution (below) and by NeuralSolverTrainingExt for
# physics-informed normalisation.

function _symmetric_sqrt(M::AbstractMatrix{T}) where {T}
    F = eigen(Symmetric(M))
    sqrtvals = sqrt.(max.(F.values, zero(T)))
    return Matrix{T}(F.vectors * Diagonal(sqrtvals) * F.vectors')
end

# Returns (H_vv_mean, W_controls_mean):
#   H_vv_mean — mean ∂²ρ/∂v²  (n_variables × n_variables)
#   W_controls_mean  — mean Schur complement H_uv H_vv⁻¹ H_vu  (n_controls × n_controls)
# Averaged over a random subset of n_hessian_samples points from dataset.
function _compute_hessian_blocks(
    dataset::PowerFlowDataset,
    power_system::PowerSystem,
    n_hessian_samples::Int,
    rng::AbstractRNG,
)
    N = length(dataset)
    n_u = power_system.n_controls
    n_v = power_system.n_variables

    sample_indices = shuffle(rng, 1:N)[1:min(n_hessian_samples, N)]

    H_vv_sum = zeros(n_v, n_v)
    W_controls_sum  = zeros(n_u, n_u)

    for ii in sample_indices
        controls = dataset[ii].controls
        voltages = dataset[ii].voltages

        H     = compute_energy_hessian(power_system, voltages, controls, ExplicitAnalytical())
        H_vv  = H[1:n_v, 1:n_v]
        H_uv  = H[n_v+1:end, 1:n_v]
        H_vv  = 0.5 * (H_vv + H_vv')

        H_vv_sum += H_vv
        W_controls_sum  += H_uv * (H_vv \ H_uv')
    end

    n         = length(sample_indices)
    H_vv_mean = H_vv_sum / n
    W_controls_mean  = 0.5 * (W_controls_sum / n + (W_controls_sum / n)')

    return H_vv_mean, W_controls_mean
end

# ── DatasetDistribution ───────────────────────────────────────────────────────

struct DatasetDistribution
    ū::Vector{Float64}
    W_controls_point::Matrix{Float64}
    W_controls_mean::Matrix{Float64}
    covariance_point::Matrix{Float64}
    covariance_mean::Matrix{Float64}
    effective_variance_point::Float64
    effective_variance_mean::Float64
    log_volume_point::Float64
    log_volume_mean::Float64
    jensen_gap::Float64
end

function dataset_distribution(
    dataset::PowerFlowDataset,
    power_system::PowerSystem;
    n_hessian_samples::Int = 100,
    rng::AbstractRNG = default_rng(),
)
    controls_mat = controls_matrix(dataset)
    N, n_u = size(controls_mat)
    n_v    = power_system.n_variables

    ū = vec(mean(Float64.(controls_mat); dims=1))

    # ── Single-point W_controls at (v*, ū) ──────────────────────────────────────────
    state_mean = PowerFlowState(power_system, ū)
    solve!(state_mean, GaussNewtonSolver())
    v_star = copy(state_mean.voltages)

    H      = compute_energy_hessian(power_system, v_star, ū)
    H_vv_pt = 0.5 .* (H[1:n_v, 1:n_v] .+ H[1:n_v, 1:n_v]')
    H_uv_pt = H[n_v+1:end, 1:n_v]

    W_controls_point = try
        schur = H_uv_pt * (H_vv_pt \ H_uv_pt')
        0.5 .* (schur .+ schur')
    catch e
        e isa LinearAlgebra.SingularException || rethrow(e)
        throw(ArgumentError(
            "H_vv is singular at the mean operating point; cannot compute the point-estimate " *
            "Schur complement. Verify that the mean control ū yields a valid, non-degenerate " *
            "operating point."
        ))
    end

    # ── Sample-averaged W_controls ───────────────────────────────────────────────────
    _, W_controls_mean = _compute_hessian_blocks(dataset, power_system, n_hessian_samples, rng)

    # ── Covariance of Hessian-normalised controls ─────────────────────────────
    function _normalised_covariance(W_controls)
        sqrtWx    = _symmetric_sqrt(Float64.(W_controls))
        U_centred = Float64.(controls_mat) .- ū'   # N × n_u
        U_tilde   = U_centred * sqrtWx             # N × n_u
        return (U_tilde' * U_tilde) / N
    end

    cov_point = _normalised_covariance(W_controls_point)
    cov_mean  = _normalised_covariance(W_controls_mean)

    log_vol_point = logdet(cov_point)
    log_vol_mean  = logdet(cov_mean)

    return DatasetDistribution(
        ū,
        Float64.(W_controls_point), Float64.(W_controls_mean),
        cov_point, cov_mean,
        exp(log_vol_point / n_u), exp(log_vol_mean / n_u),
        log_vol_point, log_vol_mean,
        log_vol_point - log_vol_mean,
    )
end

function Base.show(io::IO, ::MIME"text/plain", d::DatasetDistribution)
    n_u = length(d.ū)
    println(io, "DatasetDistribution  (n_controls = ", n_u, ")")
    Printf.@printf(io, "  effective_variance:  point = %.4e   mean = %.4e\n",
        d.effective_variance_point, d.effective_variance_mean)
    Printf.@printf(io, "  log_volume:         point = %.4f   mean = %.4f\n",
        d.log_volume_point, d.log_volume_mean)
    Printf.@printf(io, "  jensen_gap:         %.4f\n", d.jensen_gap)
end
