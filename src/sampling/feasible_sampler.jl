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

struct _FeasibleBox
    gen_P_idx    :: Vector{Int}
    gen_Vref_idx :: Vector{Int}
    load_P_idx   :: Vector{Int}
    load_Q_idx   :: Vector{Int}
end

_feasible_box(ps::PowerSystem) = _FeasibleBox(
    control_indices(ps, SynchronousMachineStatic, :P),
    control_indices(ps, SynchronousMachineStatic, :Vref),
    control_indices(ps, ZIPLoad, :P),
    control_indices(ps, ZIPLoad, :Q),
)

function _feasible_sweep_controls!(
    u::AbstractVector{T}, box::_FeasibleBox, cfg::FeasibleSamplerConfig,
    rng::AbstractRNG,
) where {T}
    band(lo, hi) = lo + rand(rng) * (hi - lo)
    n_load = length(box.load_P_idx)
    n_gen  = length(box.gen_P_idx)

    total_load = band(cfg.s_total_min, cfg.s_total_max)
    split = (s = rand(rng, n_load); s ./ sum(s))
    S_load = total_load .* split
    P_load = similar(S_load)
    Q_load = similar(S_load)
    for k in 1:n_load
        pf = band(cfg.pf_min, cfg.pf_max)
        P_load[k] = S_load[k] * pf
        Q_load[k] = sqrt(max(S_load[k]^2 - P_load[k]^2, 0.0))
    end
    total_P = sum(P_load)

    gsplit = (g = rand(rng, n_gen); g ./ sum(g))
    P_gen = total_P .* gsplit       # loss scaling fixed at 1.0

    for (k, i) in enumerate(box.gen_P_idx);    u[i] = T(P_gen[k]);  end
    for (k, i) in enumerate(box.gen_Vref_idx); u[i] = T(band(cfg.vset_min, cfg.vset_max)); end
    for (k, i) in enumerate(box.load_P_idx);   u[i] = T(P_load[k]); end
    for (k, i) in enumerate(box.load_Q_idx);   u[i] = T(Q_load[k]); end
    return nothing
end

function _generate(
    ::FeasibleStrategy,
    power_system::PowerSystem{T},
    cfg::FeasibleSamplerConfig,
    n::Int;
    rng::AbstractRNG = default_rng(),
) where {T}
    box = _feasible_box(power_system)
    # Distributed slack = all generator P controls (matches the paper configs'
    # slack_indices = [1, 3, 5]).
    gen_P_idx = control_indices(power_system, SynchronousMachineStatic, :P)
    opts = SolverOptions(; max_iterations = cfg.max_iters, tolerance = cfg.tau,
                         jacobian_method = ExplicitAnalytical())

    states = Vector{PowerFlowState{T}}(undef, 0)
    u = zeros(T, power_system.n_controls)
    attempts = 0
    # Rejection tally: a non-accepted OC either could not be driven feasible
    # (past-nose / stationary residual above tau) or the solve diverged.
    n_infeasible = 0; n_diverged = 0
    max_attempts = cfg.max_attempts_mult * n
    while length(states) < n && attempts < max_attempts
        attempts += 1
        _feasible_sweep_controls!(u, box, cfg, rng)
        state = PowerFlowState(power_system, Vector{T}(u))
        accepted = false
        try
            # Warm start at the stationary point, then free the slack and drive ‖r‖→0.
            solve!(state, GaussNewtonSolver(); solver_options = opts)
            stats = solve_distributed_slack!(state, gen_P_idx;
                tol = cfg.tau, max_iterations = cfg.max_iters)
            rn   = get(stats, "residual_norm", Inf)
            conv = get(stats, "converged", false)
            if conv && isfinite(rn) && rn <= cfg.tau
                accepted = true
            elseif isfinite(rn)
                n_infeasible += 1
            else
                n_diverged += 1
            end
        catch err
            err isa Union{DomainError, SingularException} || rethrow()
            n_diverged += 1
        end
        accepted && push!(states, state)
    end

    length(states) >= n || error(
        "feasible sampler kept only $(length(states)) / $n after $attempts attempts; " *
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
