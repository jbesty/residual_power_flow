
export generate_dataset, subsample
export PowerFlowDataset
export converged, feasible, stationary, voltages_matrix, controls_matrix

using Random: AbstractRNG, default_rng
using LinearAlgebra: norm

# --- PowerFlowDataset ---
#
# A typed dataset: one PowerSystem shared across N operating points.
# Each element is a full PowerFlowState. Indexing with an integer returns
# that state directly; indexing with a vector or BitVector returns a
# sub-dataset.

struct PowerFlowDataset{T<:ALLOWED_TYPES}
    power_system :: PowerSystem{T}
    states       :: Vector{PowerFlowState{T}}

    function PowerFlowDataset(power_system::PowerSystem{T}, states::Vector{PowerFlowState{T}}) where {T<:ALLOWED_TYPES}
        for state in states
            state.power_system === power_system ||
                throw(ArgumentError("All states must share the same PowerSystem instance (===)"))
        end
        return new{T}(power_system, states)
    end
end

Base.length(dataset::PowerFlowDataset) = length(dataset.states)

Base.getindex(dataset::PowerFlowDataset, i::Int) = dataset.states[i]

Base.getindex(dataset::PowerFlowDataset, mask::AbstractVector) =
    PowerFlowDataset(dataset.power_system, dataset.states[mask])

Base.iterate(dataset::PowerFlowDataset, args...) = iterate(dataset.states, args...)
Base.eltype(::Type{PowerFlowDataset{T}}) where {T} = PowerFlowState{T}

function converged(dataset::PowerFlowDataset{T}; tolerance::Real = eps(T)^(3/4)) where {T}
    return [sqrt(2 * compute_energy(state)) < tolerance for state in dataset.states]
end

# feasibility: ‖r‖ ≈ 0 (the AC-feasible / energy criterion). `converged` is kept
# as a feasibility alias for back-compatibility with existing consumers.
feasible(dataset::PowerFlowDataset; kwargs...) = converged(dataset; kwargs...)

# stationarity: ‖J'r‖ ≈ 0 — the Gauss-Newton convergence criterion the balanced
# sampler accepts on (see gauss_newton_solver.jl). Recomputed per state at access
# time, mirroring `converged`. Distinct from feasibility: a deliberately
# light-infeasible OC is stationary (the solver settled) but not feasible.
# The default tolerance sits above the sampler's solve tolerance (1e-10) so
# legitimately-accepted states recompute as stationary, and scales with T.
function stationary(dataset::PowerFlowDataset{T};
                    tolerance::Real = sqrt(eps(T)),
                    jacobian_method = ExplicitAnalytical()) where {T}
    flags = Vector{Bool}(undef, length(dataset))
    for (i, state) in enumerate(dataset.states)
        r = compute_residual!(state)
        J = compute_jacobian_voltages(state.power_system, state.voltages, state.controls,
                                      jacobian_method; statuses = state.statuses, _cycle_kw(state)...)
        flags[i] = norm(J' * r) < tolerance
    end
    return flags
end

function voltages_matrix(dataset::PowerFlowDataset{T}) where {T}
    N = length(dataset)
    n_vars = dataset.power_system.n_variables
    N == 0 && return Matrix{T}(undef, 0, n_vars)
    mat = Matrix{T}(undef, N, n_vars)
    for (i, state) in enumerate(dataset.states)
        mat[i, :] .= state.voltages
    end
    return mat
end

function controls_matrix(dataset::PowerFlowDataset{T}) where {T}
    N = length(dataset)
    n_ctrl = dataset.power_system.n_controls
    N == 0 && return Matrix{T}(undef, 0, n_ctrl)
    mat = Matrix{T}(undef, N, n_ctrl)
    for (i, state) in enumerate(dataset.states)
        mat[i, :] .= state.controls
    end
    return mat
end

# --- Dataset-level solve ---

function solve!(
    dataset::PowerFlowDataset,
    solver::PowerFlowSolver;
    solver_options::SolverOptions = SolverOptions(),
)
    for state in dataset.states
        solve!(state, solver; solver_options)
    end
    return dataset
end

function solve(
    dataset::PowerFlowDataset,
    solver::PowerFlowSolver;
    solver_options::SolverOptions = SolverOptions(),
)
    new_dataset = deepcopy(dataset)
    solve!(new_dataset, solver; solver_options)
    return new_dataset
end

# --- Subsampling ---

function subsample(dataset::PowerFlowDataset, n::Int; rng::AbstractRNG = default_rng())
    n <= length(dataset) || throw(ArgumentError(
        "subsample: requested $n samples but dataset has only $(length(dataset))"
    ))
    indices = shuffle(rng, 1:length(dataset))[1:n]
    return dataset[indices]
end

# --- Dataset generation ---
#
# generate_dataset(ps, config::AbstractSamplerConfig, n; rng) is the strategy-agnostic
# entry; it dispatches to the selected sampler strategy's _generate (the seam in
# src/sampling/sampler_strategy.jl). The strategies live in src/sampling/.
