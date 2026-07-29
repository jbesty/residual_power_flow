export SamplerConfig, calibrate_capacity

using Random: AbstractRNG, default_rng, randn!
using LinearAlgebra: norm, lu

# ── SamplerConfig ────────────────────────────────────────────────────────────
#
# Universal knob block for the prior-centred balanced operating-condition
# sampler. Defaults reproduce the paper's tuned case9 recipe. The same block
# runs any system size — only the load range auto-calibrates to the system's
# carrying capacity. Seeds live in DatasetConfig / the rng argument, not here.

struct SamplerConfig <: AbstractSamplerConfig
    # load reactive Q/P ratio band (negative = capacitive, positive = inductive)
    qp_min        :: Float64
    qp_max        :: Float64
    # per-generator target-voltage band (V_ref = VT_g + common-K_DV offset);
    # distinct VT_g per generator avoids an all-equal-Vref degenerate control set
    vtarget_min   :: Float64
    vtarget_max   :: Float64
    # infeasibility offset half-widths (fractions of total load)
    offset_p      :: Float64
    offset_q      :: Float64
    # asymmetric rejection band
    tau_p_rel     :: Float64   # active residual tolerance, fraction of total load
    v_min         :: Float64   # voltage band (reactive deviation surrogate)
    v_max         :: Float64
    # voltage-collapse nose pre-screen (staged GN solve)
    k_prescreen   :: Int
    v_prescreen   :: Float64
    max_iters     :: Int
    # auto-calibration of the load range to carrying capacity
    load_max_frac :: Float64   # L_MAX = load_max_frac · capacity
    load_min_frac :: Float64   # L_MIN = load_min_frac · L_MAX
    cap_target    :: Float64   # accept-fraction defining capacity
    cap_samples   :: Int       # probes per capacity evaluation
    # condenser handling: false = synchronous condenser (active P fixed at 0,
    # excluded from the active sphere). Empty = all generators dispatchable.
    dispatchable  :: Vector{Bool}
    # rejection-loop budget
    max_attempts_mult :: Int
    # optional slack-close_to_feasibility to AC-feasibility (see _generate). When true, each
    # in-band stationary draw has its generator P controls freed as a distributed
    # slack and is driven to ‖r‖ → 0 (PtO / ACFeasiblePF), so targets are FEASIBLE
    # (ρ ≈ 0) instead of relative-residual in-band. The balanced draw already lands
    # away from the nose, so the close_to_feasibility converges — this is why balanced + close_to_feasibility
    # scales to large heterogeneous grids where the feasible sampler's naive sweep
    # draw cannot. Off = the standard (deliberately infeasible) balanced draw.
    close_slack   :: Bool
    tau_feasible  :: Float64   # absolute ‖r‖ accept tolerance when close_slack is on
end

function SamplerConfig(;
    qp_min        = -0.3,
    qp_max        = 0.4,
    vtarget_min   = 1.03,
    vtarget_max   = 1.05,
    offset_p      = 0.04,
    offset_q      = 0.02,
    tau_p_rel     = 0.03,
    v_min         = 0.85,
    v_max         = 1.15,
    k_prescreen   = 8,
    v_prescreen   = 0.60,
    max_iters     = 100,
    load_max_frac = 1.0,
    load_min_frac = 0.4,
    cap_target    = 0.6,
    cap_samples   = 120,
    dispatchable  = Bool[],
    max_attempts_mult = 60,
    close_slack   = false,
    tau_feasible  = 1.0e-8,
)
    return SamplerConfig(
        Float64(qp_min), Float64(qp_max),
        Float64(vtarget_min), Float64(vtarget_max),
        Float64(offset_p), Float64(offset_q),
        Float64(tau_p_rel), Float64(v_min), Float64(v_max),
        Int(k_prescreen), Float64(v_prescreen), Int(max_iters),
        Float64(load_max_frac), Float64(load_min_frac),
        Float64(cap_target), Int(cap_samples),
        Vector{Bool}(dispatchable),
        Int(max_attempts_mult),
        Bool(close_slack), Float64(tau_feasible),
    )
end

# ── System-derived sampler workspace (computed once per generate call) ────────
#
# Bundles the index vectors, line parameters, the reduced DC-susceptance
# factorisation and the dispatch weights so the rejection loop and the capacity
# scan reuse them without recomputation.

struct _SamplerSystem{T}
    ps                         :: PowerSystem{T}
    n_buses                    :: Int
    n_branches                 :: Int
    n_generators               :: Int
    n_loads                    :: Int
    generator_buses            :: Vector{Int}
    load_buses                 :: Vector{Int}
    generator_P_indices        :: Vector{Int}
    generator_Vref_indices     :: Vector{Int}
    load_P_indices             :: Vector{Int}
    load_Q_indices             :: Vector{Int}
    branch_R                   :: Vector{Float64}
    branch_X                   :: Vector{Float64}
    branch_B                   :: Vector{Float64}
    branch_from_bus            :: Vector{Int}
    branch_to_bus              :: Vector{Int}
    total_charging_susceptance :: Float64
    generator_K_DV             :: Vector{Float64}
    total_K_DV                 :: Float64
    reduced_B_factorisation    :: Any        # lu factorisation of the reference-bus-reduced Bbus
    dispatch_weights           :: Vector{Float64}
end

# Per-branch line parameters; handles Branch and AsymmetricBranch (the latter
# stores only the admittance entries — recover R/X/B as in the paper reference).
function _branch_params(record::BranchRecord)
    component = record.component
    if component isa Branch
        return (Float64(component.R), Float64(component.X), Float64(component.B))
    else  # AsymmetricBranch
        impedance = 1 / -component.Y_12
        return (real(impedance), imag(impedance),
                2 * imag(component.Y_11 + component.Y_12))
    end
end

function _build_sampler_system(ps::PowerSystem{T}, dispatchable::Vector{Bool}) where {T}
    n_buses         = ps.n_buses
    generator_buses = Int[]
    load_buses      = Int[]
    generator_K_DV  = Float64[]
    for injector in ps.single_bus_injectors
        if injector.component isa SynchronousMachineStatic
            push!(generator_buses, injector.bus_id)
            push!(generator_K_DV, Float64(injector.component.K_DV))
        elseif injector.component isa ZIPLoad
            push!(load_buses, injector.bus_id)
        end
    end
    total_K_DV   = sum(generator_K_DV; init = 0.0)
    n_generators = length(generator_buses)
    n_loads      = length(load_buses)

    generator_P_indices    = control_indices(ps, SynchronousMachineStatic, :P)
    generator_Vref_indices = control_indices(ps, SynchronousMachineStatic, :Vref)
    load_P_indices         = control_indices(ps, ZIPLoad, :P)
    load_Q_indices         = control_indices(ps, ZIPLoad, :Q)

    n_branches      = length(ps.branch_injectors)
    branch_R        = Vector{Float64}(undef, n_branches)
    branch_X        = Vector{Float64}(undef, n_branches)
    branch_B        = Vector{Float64}(undef, n_branches)
    branch_from_bus = Vector{Int}(undef, n_branches)
    branch_to_bus   = Vector{Int}(undef, n_branches)
    for (branch, record) in enumerate(ps.branch_injectors)
        R, X, B = _branch_params(record)
        branch_R[branch] = R; branch_X[branch] = X; branch_B[branch] = B
        branch_from_bus[branch] = record.from_bus
        branch_to_bus[branch]   = record.to_bus
    end
    total_charging_susceptance = sum(branch_B; init = 0.0)

    # DC susceptance matrix, reference bus 1 dropped, factorised once.
    Bbus = zeros(Float64, n_buses, n_buses)
    for branch in 1:n_branches
        susceptance = 1 / branch_X[branch]
        from_bus, to_bus = branch_from_bus[branch], branch_to_bus[branch]
        Bbus[from_bus, from_bus] += susceptance; Bbus[to_bus, to_bus] += susceptance
        Bbus[from_bus, to_bus]   -= susceptance; Bbus[to_bus, from_bus] -= susceptance
    end
    reduced_B_factorisation = lu(@view Bbus[2:n_buses, 2:n_buses])

    if isempty(dispatchable)
        dispatch_weights = ones(Float64, n_generators)
    else
        length(dispatchable) == n_generators || throw(ArgumentError(
            "SamplerConfig.dispatchable has length $(length(dispatchable)) but the " *
            "system has $n_generators generators"))
        dispatch_weights = Float64.(dispatchable)
    end

    return _SamplerSystem{T}(
        ps, n_buses, n_branches, n_generators, n_loads, generator_buses, load_buses,
        generator_P_indices, generator_Vref_indices, load_P_indices, load_Q_indices,
        branch_R, branch_X, branch_B, branch_from_bus, branch_to_bus,
        total_charging_susceptance, generator_K_DV, total_K_DV,
        reduced_B_factorisation, dispatch_weights,
    )
end

# DC line flows from bus injections: θ ≈ B⁻¹ p (reference bus 1 pinned to 0),
# P_line = (θ_from − θ_to) / X.
function _dc_lineflows(sampler_system::_SamplerSystem, P_inj::AbstractVector)
    n_buses = sampler_system.n_buses
    θ = zeros(Float64, n_buses)
    θ[2:n_buses] = sampler_system.reduced_B_factorisation \ @view P_inj[2:n_buses]
    return [
        (θ[sampler_system.branch_from_bus[branch]] - θ[sampler_system.branch_to_bus[branch]]) /
        sampler_system.branch_X[branch]
        for branch in 1:sampler_system.n_branches
    ]
end

# ── Construct one OC at the balance manifold + offsets ────────────────────────
#
# Writes the control vector into `controls`. `load_directions` / `generator_directions`
# are scratch buffers (load / generation sphere directions). Places the OC on the
# power-balance manifold via a DC-flow exact match, then applies the reactive and
# voltage-target offsets.
function _construct_controls!(
    controls::AbstractVector, sampler_system::_SamplerSystem, config::SamplerConfig,
    rng::AbstractRNG,
    load_directions::Vector{Float64}, generator_directions::Vector{Float64},
    voltage_targets::Vector{Float64},
    load_level::Float64, offset_P::Float64, offset_Q::Float64,
)
    n_loads      = sampler_system.n_loads
    n_generators = sampler_system.n_generators

    randn!(rng, load_directions); load_directions ./= norm(load_directions)
    load_weights = abs.(load_directions)
    p_load = load_level .* (load_weights ./ sum(load_weights))
    q_load = similar(p_load)
    for load in 1:n_loads
        q_over_p_ratio = config.qp_min + rand(rng) * (config.qp_max - config.qp_min)
        q_load[load] = p_load[load] * q_over_p_ratio
    end

    randn!(rng, generator_directions); generator_directions ./= norm(generator_directions)
    # condensers carry no active power
    generator_weights = abs.(generator_directions) .* sampler_system.dispatch_weights
    total_generator_weight = sum(generator_weights)

    # Per-generator target voltage, drawn from the V-target band. Distinct VT_g
    # breaks the all-equal-Vref degeneracy; the common reactive offset is applied
    # about the K_DV-weighted mean so the per-gen deviations cancel and aggregate
    # reactive balance is preserved.
    for generator in 1:n_generators
        voltage_targets[generator] =
            config.vtarget_min + rand(rng) * (config.vtarget_max - config.vtarget_min)
    end

    P_inj = zeros(Float64, sampler_system.n_buses)
    for (load, bus) in enumerate(sampler_system.load_buses)
        P_inj[bus] -= p_load[load]
    end
    for (generator, bus) in enumerate(sampler_system.generator_buses)
        P_inj[bus] += (load_level / total_generator_weight) * generator_weights[generator]
    end
    P_line = _dc_lineflows(sampler_system, P_inj)
    P_loss = sum(sampler_system.branch_R[branch] * P_line[branch]^2
                 for branch in 1:sampler_system.n_branches; init = 0.0)
    Q_gap  = sum(sampler_system.branch_X[branch] * P_line[branch]^2
                 for branch in 1:sampler_system.n_branches; init = 0.0) -
             sampler_system.total_charging_susceptance

    # 0 for condensers
    p_gen = ((load_level + P_loss + offset_P * load_level) / total_generator_weight) .*
            generator_weights
    Q_target = sum(q_load) + Q_gap + offset_Q * load_level
    weighted_mean_target_voltage =
        sum(sampler_system.generator_K_DV[generator] * voltage_targets[generator]
            for generator in 1:n_generators) / sampler_system.total_K_DV
    δ = Q_target / (weighted_mean_target_voltage * sampler_system.total_K_DV)

    T = eltype(controls)
    for (generator, control) in enumerate(sampler_system.generator_P_indices)
        controls[control] = T(p_gen[generator])
    end
    for (generator, control) in enumerate(sampler_system.generator_Vref_indices)
        controls[control] = T(voltage_targets[generator] + δ)
    end
    for (load, control) in enumerate(sampler_system.load_P_indices)
        controls[control] = T(p_load[load])
    end
    for (load, control) in enumerate(sampler_system.load_Q_indices)
        controls[control] = T(q_load[load])
    end
    return nothing
end

# Draw a full random OC (load level + offsets + target voltage) into `controls`.
function _draw_controls!(
    controls::AbstractVector, sampler_system::_SamplerSystem, config::SamplerConfig,
    rng::AbstractRNG,
    load_directions::Vector{Float64}, generator_directions::Vector{Float64},
    voltage_targets::Vector{Float64},
    load_level_min::Float64, load_level_max::Float64,
)
    load_level = load_level_min + rand(rng) * (load_level_max - load_level_min)
    offset_P = -config.offset_p + rand(rng) * 2 * config.offset_p
    offset_Q = -config.offset_q + rand(rng) * 2 * config.offset_q
    _construct_controls!(controls, sampler_system, config, rng,
                         load_directions, generator_directions, voltage_targets,
                         load_level, offset_P, offset_Q)
    return load_level
end

# ── Staged GN solve with voltage-collapse nose pre-screen ─────────────────────
_sampler_solver_options(max_iterations::Int) = SolverOptions(;
    max_iterations = max_iterations, tolerance = 1.0e-10,
    jacobian_method = ExplicitAnalytical())

function _staged_solve(sampler_system::_SamplerSystem{T}, controls::AbstractVector,
                       config::SamplerConfig) where {T}
    state = PowerFlowState(sampler_system.ps, Vector{T}(controls))
    solve!(state, GaussNewtonSolver();
           solver_options = _sampler_solver_options(config.k_prescreen))
    minimum(@view state.voltages[1:sampler_system.n_buses]) < config.v_prescreen &&
        return state, nothing
    stats = solve!(state, GaussNewtonSolver();
                   solver_options = _sampler_solver_options(config.max_iters))
    return state, stats
end

# Evaluate one control vector. Returns
# (tag, state, kcl_real, kcl_imag, voltage_min, voltage_max, total_active_load)
# with tag ∈ {:prescreen, :diverged, :rejected, :ok}. `total_active_load` is the
# residual-tolerance denominator, surfaced so the caller can attribute a :rejected
# outcome to residual vs voltage without recomputing. Validity is stationarity (the
# GN convergence flag), never feasibility (‖r‖≈0) — the offsets make these OCs
# deliberately light-infeasible.
function _evaluate(sampler_system::_SamplerSystem, controls::AbstractVector,
                   config::SamplerConfig; slack_indices = nothing)
    state, stats = _staged_solve(sampler_system, controls, config)
    stats === nothing && return (:prescreen, state, 0.0, 0.0, 0.0, 0.0, 0.0)
    stats["converged"] || return (:diverged, state, 0.0, 0.0, 0.0, 0.0, 0.0)
    # The close-to-feasibility path is driven by whether `slack_indices` was supplied,
    # not by config.close_slack: capacity calibration calls this WITHOUT slack_indices
    # so it probes the stationary nose, independent of the close; the generate loop
    # supplies slack_indices to close to feasibility.
    close_to_feasibility = slack_indices !== nothing
    if close_to_feasibility
        # In-band stationary draw → free the generator P slack and drive ‖r‖ → 0.
        # Starting away from the nose (the staged solve converged in-band), so the
        # close converges; failure to reach tolerance is a genuine rejection.
        slack_stats = solve_distributed_slack!(state, slack_indices;
            tol = config.tau_feasible, max_iterations = config.max_iters)
        residual_norm = get(slack_stats, "residual_norm", Inf)
        converged     = get(slack_stats, "converged", false)
        (converged && isfinite(residual_norm)) ||
            return (:diverged, state, 0.0, 0.0, 0.0, 0.0, 0.0)
        kcl_real = residual_norm; kcl_imag = 0.0
    else
        r = compute_residual(sampler_system.ps, state.voltages, state.controls)
        n_current_balance = 2 * sampler_system.n_buses
        kcl_real = sqrt(sum(abs2, @view r[1:2:n_current_balance]))
        kcl_imag = sqrt(sum(abs2, @view r[2:2:n_current_balance]))
    end
    bus_voltages = @view state.voltages[1:sampler_system.n_buses]
    voltage_min, voltage_max = extrema(bus_voltages)
    total_active_load =
        sum(state.controls[control] for control in sampler_system.load_P_indices; init = 0.0)
    residual_in_band = close_to_feasibility ? (kcl_real <= config.tau_feasible) :
                       (kcl_real <= config.tau_p_rel * total_active_load)
    accepted = residual_in_band && voltage_min >= config.v_min && voltage_max <= config.v_max
    return (accepted ? :ok : :rejected, state, kcl_real, kcl_imag,
            voltage_min, voltage_max, total_active_load)
end

# ── Auto-calibration of the carrying capacity (bracket then bisect) ───────────
#
# Resolves the load level at which the nominal (zero-offset) accept fraction
# falls to `cap_target` — the collapse-side (upper) edge of the feasible load
# window. Scale-agnostic (relative bisection tolerance). Exposed so callers can
# reuse it; the load range derives as
# L_MAX = load_max_frac · capacity, L_MIN = load_min_frac · L_MAX.
#
# The accept fraction is NOT monotone in load: at very light load OCs are
# rejected on high voltage (line-charging / Ferranti effect), so the floor can
# sit *below* the feasible window rather than above the collapse edge. The
# search therefore first marches up to enter the window (accept_fraction ≥ target) before
# bracketing the upper edge. Systems whose floor is already inside the window
# (accept_fraction(floor_start) ≥ target) skip the march and behave exactly as the plain
# bracket-then-bisect did.
function calibrate_capacity(
    ps::PowerSystem, config::SamplerConfig;
    rng::AbstractRNG = default_rng(),
)
    sampler_system = _build_sampler_system(ps, config.dispatchable)
    return _calibrate_capacity(sampler_system, config, rng)
end

function _calibrate_capacity(sampler_system::_SamplerSystem{T}, config::SamplerConfig, rng::AbstractRNG;
    floor_start::Float64 = 1.0, tolerance_fraction::Float64 = 0.03) where {T}
    controls_buffer      = zeros(T, sampler_system.ps.n_controls)
    load_directions      = zeros(Float64, sampler_system.n_loads)
    generator_directions = zeros(Float64, sampler_system.n_generators)
    voltage_targets      = zeros(Float64, sampler_system.n_generators)
    accept_fraction(load_level) = begin
        n_accepted = 0
        for _ in 1:config.cap_samples
            _construct_controls!(controls_buffer, sampler_system, config, rng,
                                 load_directions, generator_directions, voltage_targets,
                                 load_level, 0.0, 0.0)
            _evaluate(sampler_system, controls_buffer, config)[1] === :ok && (n_accepted += 1)
        end
        n_accepted / config.cap_samples
    end
    lower = floor_start
    if accept_fraction(lower) < config.cap_target
        # Floor sits below the feasible window (light-load high-voltage
        # rejection). March up until the accept fraction first reaches target.
        # Doubling matches the bracket granularity below; a window narrower than
        # one doubling step is missed and falls through to the floor fallback.
        probe = lower; entered = false
        while probe < 1.0e4
            probe *= 2
            if accept_fraction(probe) >= config.cap_target
                lower = probe; entered = true; break
            end
        end
        entered || return floor_start   # no feasible window found; degenerate bands
    end
    upper = 2 * lower
    while accept_fraction(upper) >= config.cap_target && upper < 1.0e4
        lower = upper; upper *= 2
    end
    while (upper - lower) / upper > tolerance_fraction
        midpoint = 0.5 * (lower + upper)
        accept_fraction(midpoint) >= config.cap_target ? (lower = midpoint) : (upper = midpoint)
    end
    return round(lower; digits = 2)
end

# ── Balanced strategy: generate a balanced dataset ────────────────────────────
#
# Auto-calibrates the load range, then rejection-samples balanced OCs until
# `n_samples` stationarity-converged, in-band states are collected. Returns a
# (PowerFlowDataset, SamplerDiagnostics) pair. Throws if the budget
# (max_attempts_mult · n_samples) is exhausted before the target is met.
#
# This is the BPC sampler moved behind the strategy seam unchanged — the RNG draw
# sequence is identical to the pre-seam `generate_dataset`, so generated values are
# byte-identical at a fixed seed. `generate_dataset` (dataset.jl / sampler_strategy.jl)
# dispatches here for SamplerConfig via `_strategy_for`.
_strategy_for(::SamplerConfig) = BalancedStrategy()

function _generate(
    ::BalancedStrategy,
    power_system::PowerSystem,
    config::SamplerConfig,
    n_samples::Int;
    rng::AbstractRNG = default_rng(),
)
    T = eltype(get_flat_start(power_system))
    sampler_system = _build_sampler_system(power_system, config.dispatchable)

    capacity = _calibrate_capacity(sampler_system, config, rng)
    load_level_max = config.load_max_frac * capacity
    load_level_min = config.load_min_frac * load_level_max

    # Optional slack-close: free the generator P controls (distributed slack) and
    # drive each in-band stationary draw to AC-feasibility via solve_distributed_slack!.
    slack_indices = config.close_slack ? sampler_system.generator_P_indices : nothing

    states               = Vector{PowerFlowState{T}}(undef, 0)
    controls             = zeros(T, power_system.n_controls)
    load_directions      = zeros(Float64, sampler_system.n_loads)
    generator_directions = zeros(Float64, sampler_system.n_generators)
    voltage_targets      = zeros(Float64, sampler_system.n_generators)

    attempts = 0
    # Rejection-reason tally. The first three partition every non-accepted
    # attempt (prescreen + diverged + rejected + accepted == attempts); the
    # residual/v_low/v_high sub-counts attribute each :rejected outcome and may
    # overlap (a state can fail more than one in-band test at once).
    n_prescreen = 0; n_diverged = 0; n_rejected = 0
    n_rejected_residual = 0; n_rejected_voltage_low = 0; n_rejected_voltage_high = 0
    max_attempts = config.max_attempts_mult * n_samples
    while length(states) < n_samples && attempts < max_attempts
        attempts += 1
        _draw_controls!(controls, sampler_system, config, rng,
                        load_directions, generator_directions, voltage_targets,
                        load_level_min, load_level_max)
        tag, state, kcl_real, _, voltage_min, voltage_max, total_active_load =
            _evaluate(sampler_system, controls, config; slack_indices)
        if tag === :ok
            push!(states, state)
        elseif tag === :prescreen
            n_prescreen += 1
        elseif tag === :diverged
            n_diverged += 1
        else  # :rejected — in-band test failed; attribute which
            n_rejected += 1
            residual_tolerance = config.close_slack ? config.tau_feasible :
                                 config.tau_p_rel * total_active_load
            kcl_real > residual_tolerance && (n_rejected_residual += 1)
            voltage_min < config.v_min && (n_rejected_voltage_low += 1)
            voltage_max > config.v_max && (n_rejected_voltage_high += 1)
        end
    end

    length(states) >= n_samples || error(
        "balanced sampler kept only $(length(states)) / $n_samples after " *
        "$attempts attempts (capacity $capacity, " *
        "load range [$load_level_min, $load_level_max]); " *
        "loosen the bands or raise max_attempts_mult")

    dataset = PowerFlowDataset(power_system, states)
    diagnostics = SamplerDiagnostics(; capacity = capacity, attempts = attempts,
        accepted = length(states),
        extra = Dict{String,Any}(
            "yield"      => attempts > 0 ? length(states) / attempts : 0.0,
            "load_range" => [load_level_min, load_level_max],
            "rejections" => Dict{String,Any}(
                "prescreen" => n_prescreen,
                "diverged"  => n_diverged,
                "rejected"  => n_rejected),
            "reject_detail" => Dict{String,Any}(
                "residual" => n_rejected_residual,
                "v_low"    => n_rejected_voltage_low,
                "v_high"   => n_rejected_voltage_high)))
    return dataset, diagnostics
end
