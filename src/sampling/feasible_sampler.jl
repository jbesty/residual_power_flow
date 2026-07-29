export FeasibleSamplerConfig

using LinearAlgebra: SingularException

# ── Feasible strategy (paper slack-feasible sampler) ──────────────────────────
#
# Recreates the old paper operating-condition sampler that produced the committed
# AC-feasible case9 datasets (PowerUp narrow/wide, RPF case9_paper). Two pieces:
#
#   1. OC construction — a uniform sweep over operating conditions (total apparent
#      power S over a range, per-load power factor, per-generator V setpoint,
#      participation shares). Self-contained below (`_feasible_sweep_controls!`).
#   2. slack-feasible solve — designate the generator P controls as a distributed
#      slack and solve to AC-feasibility (‖r‖ → 0) via `solve_distributed_slack!`.
#      The slack absorbs the power-balance mismatch, so a sampled OC becomes a
#      feasible solution.
#
# SCOPE: this is the deliberately naive uniform-sweep + slack recipe. It is for
# SMALL, HOMOGENEOUS systems (case9 and similar) only — random participation
# shares and a single slack pool ignore real capacity distribution and throw OCs
# past the nose on large heterogeneous grids. It is a paper-reproduction tool, not
# a scalable sampler; `balanced` is the auto-calibrating one.

struct FeasibleSamplerConfig <: AbstractSamplerConfig
    # uniform-sweep OC ranges (loss scaling is fixed at 1.0 — the slack absorbs the
    # loss mismatch, so a pre-solve loss-anticipation factor is redundant here)
    s_total_min :: Float64
    s_total_max :: Float64
    pf_min      :: Float64
    pf_max      :: Float64
    vset_min    :: Float64
    vset_max    :: Float64
    # feasibility accept gate: accept iff the slack solve converged and ‖r‖ ≤ tau
    # (no voltage band — collapse is already excluded by non-convergence)
    tau         :: Float64
    max_iters   :: Int          # GN / slack-solve iterations per OC
    max_attempts_mult :: Int    # rejection-loop budget = mult · n
end

function FeasibleSamplerConfig(;
    s_total_min = 1.0,
    s_total_max = 4.0,
    pf_min      = 0.9,
    pf_max      = 1.0,
    vset_min    = 1.0,
    vset_max    = 1.05,
    tau         = 1.0e-8,
    max_iters   = 100,
    max_attempts_mult = 60,
)
    0.0 ≤ pf_min ≤ 1.0 && 0.0 ≤ pf_max ≤ 1.0 ||
        throw(ArgumentError("power factors must lie in [0, 1]"))
    return FeasibleSamplerConfig(
        Float64(s_total_min), Float64(s_total_max),
        Float64(pf_min), Float64(pf_max),
        Float64(vset_min), Float64(vset_max),
        Float64(tau), Int(max_iters), Int(max_attempts_mult),
    )
end

_strategy_for(::FeasibleSamplerConfig) = FeasibleStrategy()

# ── Operating-condition draw (uniform sweep) ──────────────────────────────────
#
# One uniform OC control vector: total load split over loads, per-load power
# factor, generation split, per-generator V set point. Generation loss scaling is
# fixed at 1.0 — the distributed slack carries the losses.

struct _FeasibleControlIndices
    generator_P_indices    :: Vector{Int}
    generator_Vref_indices :: Vector{Int}
    load_P_indices         :: Vector{Int}
    load_Q_indices         :: Vector{Int}
end

_feasible_control_indices(ps::PowerSystem) = _FeasibleControlIndices(
    control_indices(ps, SynchronousMachineStatic, :P),
    control_indices(ps, SynchronousMachineStatic, :Vref),
    control_indices(ps, ZIPLoad, :P),
    control_indices(ps, ZIPLoad, :Q),
)

function _feasible_sweep_controls!(
    controls::AbstractVector{T}, indices::_FeasibleControlIndices,
    config::FeasibleSamplerConfig, rng::AbstractRNG,
) where {T}
    draw_uniform(lower, upper) = lower + rand(rng) * (upper - lower)
    n_loads      = length(indices.load_P_indices)
    n_generators = length(indices.generator_P_indices)

    total_load  = draw_uniform(config.s_total_min, config.s_total_max)
    load_shares = (raw = rand(rng, n_loads); raw ./ sum(raw))
    S_load = total_load .* load_shares
    P_load = similar(S_load)
    Q_load = similar(S_load)
    for load in 1:n_loads
        power_factor = draw_uniform(config.pf_min, config.pf_max)
        P_load[load] = S_load[load] * power_factor
        Q_load[load] = sqrt(max(S_load[load]^2 - P_load[load]^2, 0.0))
    end
    total_P = sum(P_load)

    generator_shares = (raw = rand(rng, n_generators); raw ./ sum(raw))
    P_gen = total_P .* generator_shares       # loss scaling fixed at 1.0

    for (generator, control) in enumerate(indices.generator_P_indices)
        controls[control] = T(P_gen[generator])
    end
    for control in indices.generator_Vref_indices
        controls[control] = T(draw_uniform(config.vset_min, config.vset_max))
    end
    for (load, control) in enumerate(indices.load_P_indices); controls[control] = T(P_load[load]); end
    for (load, control) in enumerate(indices.load_Q_indices); controls[control] = T(Q_load[load]); end
    return nothing
end

function _generate(
    ::FeasibleStrategy,
    power_system::PowerSystem{T},
    config::FeasibleSamplerConfig,
    n_samples::Int;
    rng::AbstractRNG = default_rng(),
) where {T}
    indices = _feasible_control_indices(power_system)
    # Distributed slack = all generator P controls (matches the paper configs'
    # slack_indices = [1, 3, 5]).
    generator_P_indices = control_indices(power_system, SynchronousMachineStatic, :P)
    solver_options = SolverOptions(; max_iterations = config.max_iters, tolerance = config.tau,
                                   jacobian_method = ExplicitAnalytical())

    states = Vector{PowerFlowState{T}}(undef, 0)
    controls = zeros(T, power_system.n_controls)
    attempts = 0
    # Rejection tally: a non-accepted OC either could not be driven feasible
    # (past-nose / stationary residual above tau) or the solve diverged.
    n_infeasible = 0; n_diverged = 0
    max_attempts = config.max_attempts_mult * n_samples
    while length(states) < n_samples && attempts < max_attempts
        attempts += 1
        _feasible_sweep_controls!(controls, indices, config, rng)
        state = PowerFlowState(power_system, Vector{T}(controls))
        accepted = false
        try
            # Warm start at the stationary point, then free the slack and drive ‖r‖→0.
            solve!(state, GaussNewtonSolver(); solver_options = solver_options)
            stats = solve_distributed_slack!(state, generator_P_indices;
                tol = config.tau, max_iterations = config.max_iters)
            residual_norm = get(stats, "residual_norm", Inf)
            converged     = get(stats, "converged", false)
            if converged && isfinite(residual_norm) && residual_norm <= config.tau
                accepted = true
            elseif isfinite(residual_norm)
                n_infeasible += 1
            else
                n_diverged += 1
            end
        catch exception
            exception isa Union{DomainError, SingularException} || rethrow()
            n_diverged += 1
        end
        accepted && push!(states, state)
    end

    length(states) >= n_samples || error(
        "feasible sampler kept only $(length(states)) / $n_samples after $attempts attempts; " *
        "widen the sweep ranges, raise tau, or raise max_attempts_mult")

    dataset = PowerFlowDataset(power_system, states)
    diagnostics = SamplerDiagnostics(; attempts = attempts, accepted = length(states),
        extra = Dict{String,Any}(
            "yield"      => attempts > 0 ? length(states) / attempts : 0.0,
            "rejections" => Dict{String,Any}(
                "infeasible" => n_infeasible,
                "diverged"   => n_diverged)))
    return dataset, diagnostics
end
