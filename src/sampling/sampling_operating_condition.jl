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
    # optional slack-close to AC-feasibility (see _generate). When true, each
    # in-band stationary draw has its generator P controls freed as a distributed
    # slack and is driven to ‖r‖ → 0 (PtO / ACFeasiblePF), so targets are FEASIBLE
    # (ρ ≈ 0) instead of relative-residual in-band. The balanced draw already lands
    # away from the nose, so the close converges — this is why balanced + close
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
    ps           :: PowerSystem{T}
    n_buses      :: Int
    n_branches   :: Int
    n_gen        :: Int
    n_load       :: Int
    gen_buses    :: Vector{Int}
    load_buses   :: Vector{Int}
    gen_P_idx    :: Vector{Int}
    gen_Vref_idx :: Vector{Int}
    load_P_idx   :: Vector{Int}
    load_Q_idx   :: Vector{Int}
    br_R         :: Vector{Float64}
    br_X         :: Vector{Float64}
    br_B         :: Vector{Float64}
    br_from      :: Vector{Int}
    br_to        :: Vector{Int}
    sumB         :: Float64
    gen_KDV      :: Vector{Float64}
    sumKDV       :: Float64
    Bred_factor  :: Any        # lu factorisation of the reference-bus-reduced Bbus
    dispatch_w   :: Vector{Float64}
end

# Per-branch line parameters; handles Branch and AsymmetricBranch (the latter
# stores only the admittance entries — recover R/X/B as in the paper reference).
function _branch_params(rec::BranchRecord)
    c = rec.component
    if c isa Branch
        return (Float64(c.R), Float64(c.X), Float64(c.B))
    else  # AsymmetricBranch
        z = 1 / -c.Y_12
        return (real(z), imag(z), 2 * imag(c.Y_11 + c.Y_12))
    end
end

function _build_sampler_system(ps::PowerSystem{T}, dispatchable::Vector{Bool}) where {T}
    nb = ps.n_buses
    gen_buses  = Int[]
    load_buses = Int[]
    gen_KDV    = Float64[]
    for inj in ps.single_bus_injectors
        if inj.component isa SynchronousMachineStatic
            push!(gen_buses, inj.bus_id)
            push!(gen_KDV, Float64(inj.component.K_DV))
        elseif inj.component isa ZIPLoad
            push!(load_buses, inj.bus_id)
        end
    end
    sumKDV = sum(gen_KDV; init = 0.0)
    n_gen  = length(gen_buses)
    n_load = length(load_buses)

    gen_P_idx    = control_indices(ps, SynchronousMachineStatic, :P)
    gen_Vref_idx = control_indices(ps, SynchronousMachineStatic, :Vref)
    load_P_idx   = control_indices(ps, ZIPLoad, :P)
    load_Q_idx   = control_indices(ps, ZIPLoad, :Q)

    nl = length(ps.branch_injectors)
    br_R = Vector{Float64}(undef, nl)
    br_X = Vector{Float64}(undef, nl)
    br_B = Vector{Float64}(undef, nl)
    br_from = Vector{Int}(undef, nl)
    br_to   = Vector{Int}(undef, nl)
    for (l, rec) in enumerate(ps.branch_injectors)
        R, X, B = _branch_params(rec)
        br_R[l] = R; br_X[l] = X; br_B[l] = B
        br_from[l] = rec.from_bus; br_to[l] = rec.to_bus
    end
    sumB = sum(br_B; init = 0.0)

    # DC susceptance matrix, reference bus 1 dropped, factorised once.
    Bbus = zeros(Float64, nb, nb)
    for l in 1:nl
        bl = 1 / br_X[l]
        f, t = br_from[l], br_to[l]
        Bbus[f, f] += bl; Bbus[t, t] += bl
        Bbus[f, t] -= bl; Bbus[t, f] -= bl
    end
    Bred_factor = lu(@view Bbus[2:nb, 2:nb])

    if isempty(dispatchable)
        dispatch_w = ones(Float64, n_gen)
    else
        length(dispatchable) == n_gen || throw(ArgumentError(
            "SamplerConfig.dispatchable has length $(length(dispatchable)) but the " *
            "system has $n_gen generators"))
        dispatch_w = Float64.(dispatchable)
    end

    return _SamplerSystem{T}(
        ps, nb, nl, n_gen, n_load, gen_buses, load_buses,
        gen_P_idx, gen_Vref_idx, load_P_idx, load_Q_idx,
        br_R, br_X, br_B, br_from, br_to, sumB, gen_KDV, sumKDV, Bred_factor, dispatch_w,
    )
end

# DC line flows from bus injections: θ ≈ B⁻¹ p (reference bus 1 pinned to 0),
# P_line = (θ_from − θ_to) / X.
function _dc_lineflows(sys::_SamplerSystem, P_inj::AbstractVector)
    nb = sys.n_buses
    θ = zeros(Float64, nb)
    θ[2:nb] = sys.Bred_factor \ @view P_inj[2:nb]
    return [(θ[sys.br_from[l]] - θ[sys.br_to[l]]) / sys.br_X[l] for l in 1:sys.n_branches]
end

# ── Construct one OC at the balance manifold + offsets ────────────────────────
#
# Writes the control vector into `u`. `dl`/`dg` are scratch buffers (load /
# generation sphere directions). Implements sampling.tex eqs (ploss), (qgap),
# (dcflow), (exactmatch), (vref), (offsets).
function _construct_controls!(
    u::AbstractVector, sys::_SamplerSystem, cfg::SamplerConfig, rng::AbstractRNG,
    dl::Vector{Float64}, dg::Vector{Float64}, vt::Vector{Float64},
    L::Float64, epP::Float64, epQ::Float64,
)
    n_load = sys.n_load
    n_gen  = sys.n_gen

    randn!(rng, dl); dl ./= norm(dl)
    w = abs.(dl); p_load = L .* (w ./ sum(w))
    q_load = similar(p_load)
    for k in 1:n_load
        qp = cfg.qp_min + rand(rng) * (cfg.qp_max - cfg.qp_min)
        q_load[k] = p_load[k] * qp
    end

    randn!(rng, dg); dg ./= norm(dg)
    wg = abs.(dg) .* sys.dispatch_w           # condensers carry no active power
    swg = sum(wg)

    # Per-generator target voltage, drawn from the V-target band. Distinct VT_g
    # breaks the all-equal-Vref degeneracy; the common reactive offset is applied
    # about the K_DV-weighted mean so the per-gen deviations cancel and aggregate
    # reactive balance is preserved (sampling.tex eq vref, generalised).
    for g in 1:n_gen
        vt[g] = cfg.vtarget_min + rand(rng) * (cfg.vtarget_max - cfg.vtarget_min)
    end

    P_inj = zeros(Float64, sys.n_buses)
    for (k, b) in enumerate(sys.load_buses); P_inj[b] -= p_load[k]; end
    for (k, b) in enumerate(sys.gen_buses);  P_inj[b] += (L / swg) * wg[k]; end
    Pline = _dc_lineflows(sys, P_inj)
    P_loss = sum(sys.br_R[l] * Pline[l]^2 for l in 1:sys.n_branches; init = 0.0)
    Q_gap  = sum(sys.br_X[l] * Pline[l]^2 for l in 1:sys.n_branches; init = 0.0) - sys.sumB

    p_gen = ((L + P_loss + epP * L) / swg) .* wg          # 0 for condensers
    Q_target = sum(q_load) + Q_gap + epQ * L
    VbarT = sum(sys.gen_KDV[g] * vt[g] for g in 1:n_gen) / sys.sumKDV
    δ = Q_target / (VbarT * sys.sumKDV)

    T = eltype(u)
    for (k, i) in enumerate(sys.gen_P_idx);    u[i] = T(p_gen[k]);     end
    for (k, i) in enumerate(sys.gen_Vref_idx); u[i] = T(vt[k] + δ);    end
    for (k, i) in enumerate(sys.load_P_idx);   u[i] = T(p_load[k]);    end
    for (k, i) in enumerate(sys.load_Q_idx);   u[i] = T(q_load[k]);    end
    return nothing
end

# Draw a full random OC (load level + offsets + target voltage) into `u`.
function _draw_controls!(
    u::AbstractVector, sys::_SamplerSystem, cfg::SamplerConfig, rng::AbstractRNG,
    dl::Vector{Float64}, dg::Vector{Float64}, vt::Vector{Float64},
    Lmin::Float64, Lmax::Float64,
)
    L = Lmin + rand(rng) * (Lmax - Lmin)
    epP = -cfg.offset_p + rand(rng) * 2 * cfg.offset_p
    epQ = -cfg.offset_q + rand(rng) * 2 * cfg.offset_q
    _construct_controls!(u, sys, cfg, rng, dl, dg, vt, L, epP, epQ)
    return L
end

# ── Staged GN solve with voltage-collapse nose pre-screen ─────────────────────
_sampler_opts(it::Int) = SolverOptions(;
    max_iterations = it, tolerance = 1.0e-10, jacobian_method = ExplicitAnalytical())

function _staged_solve(sys::_SamplerSystem{T}, u::AbstractVector, cfg::SamplerConfig) where {T}
    state = PowerFlowState(sys.ps, Vector{T}(u))
    solve!(state, GaussNewtonSolver(); solver_options = _sampler_opts(cfg.k_prescreen))
    minimum(@view state.voltages[1:sys.n_buses]) < cfg.v_prescreen && return state, nothing
    stats = solve!(state, GaussNewtonSolver(); solver_options = _sampler_opts(cfg.max_iters))
    return state, stats
end

# Evaluate one control vector. Returns (tag, state, kcl_real, kcl_imag, vlo, vhi, lp)
# with tag ∈ {:prescreen, :diverged, :rejected, :ok} and lp the total active load
# (the residual-tolerance denominator, surfaced so the caller can attribute a
# :rejected outcome to residual vs voltage without recomputing). Validity is
# stationarity (the GN convergence flag), never feasibility (‖r‖≈0) — the offsets
# make these OCs deliberately light-infeasible. See sampling.tex Remark.
function _evaluate(sys::_SamplerSystem, u::AbstractVector, cfg::SamplerConfig;
                  slack_idx = nothing)
    state, stats = _staged_solve(sys, u, cfg)
    stats === nothing && return (:prescreen, state, 0.0, 0.0, 0.0, 0.0, 0.0)
    stats["converged"] || return (:diverged, state, 0.0, 0.0, 0.0, 0.0, 0.0)
    # The close path is driven by whether `slack_idx` was supplied, not by cfg.close_slack:
    # capacity calibration calls this WITHOUT slack_idx so it probes the stationary nose,
    # independent of the close; the generate loop supplies slack_idx to close to feasibility.
    close = slack_idx !== nothing
    if close
        # In-band stationary draw → free the generator P slack and drive ‖r‖ → 0.
        # Starting away from the nose (the staged solve converged in-band), so the
        # close converges; failure to reach tolerance is a genuine rejection.
        fstats = solve_distributed_slack!(state, slack_idx;
            tol = cfg.tau_feasible, max_iterations = cfg.max_iters)
        rn   = get(fstats, "residual_norm", Inf)
        conv = get(fstats, "converged", false)
        (conv && isfinite(rn)) || return (:diverged, state, 0.0, 0.0, 0.0, 0.0, 0.0)
        kr = rn; ki = 0.0
    else
        r = compute_residual(sys.ps, state.voltages, state.controls)
        ncb = 2 * sys.n_buses
        kr = sqrt(sum(abs2, @view r[1:2:ncb]))
        ki = sqrt(sum(abs2, @view r[2:2:ncb]))
    end
    vmag = @view state.voltages[1:sys.n_buses]
    vlo, vhi = extrema(vmag)
    lp = sum(state.controls[i] for i in sys.load_P_idx; init = 0.0)
    resid_ok = close ? (kr <= cfg.tau_feasible) : (kr <= cfg.tau_p_rel * lp)
    ok = resid_ok && vlo >= cfg.v_min && vhi <= cfg.v_max
    return (ok ? :ok : :rejected, state, kr, ki, vlo, vhi, lp)
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
# search therefore first marches up to enter the window (frac ≥ target) before
# bracketing the upper edge. Systems whose floor is already inside the window
# (frac(floor0) ≥ target) skip the march and behave exactly as the plain
# bracket-then-bisect did.
function calibrate_capacity(
    ps::PowerSystem, cfg::SamplerConfig;
    rng::AbstractRNG = default_rng(),
)
    sys = _build_sampler_system(ps, cfg.dispatchable)
    return _calibrate_capacity(sys, cfg, rng)
end

function _calibrate_capacity(sys::_SamplerSystem{T}, cfg::SamplerConfig, rng::AbstractRNG;
    floor0::Float64 = 1.0, tol_frac::Float64 = 0.03) where {T}
    ub = zeros(T, sys.ps.n_controls)
    dl = zeros(Float64, sys.n_load); dg = zeros(Float64, sys.n_gen)
    vt = zeros(Float64, sys.n_gen)
    frac(L) = begin
        acc = 0
        for _ in 1:cfg.cap_samples
            _construct_controls!(ub, sys, cfg, rng, dl, dg, vt, L, 0.0, 0.0)
            _evaluate(sys, ub, cfg)[1] === :ok && (acc += 1)
        end
        acc / cfg.cap_samples
    end
    lo = floor0
    if frac(lo) < cfg.cap_target
        # Floor sits below the feasible window (light-load high-voltage
        # rejection). March up until the accept fraction first reaches target.
        # Doubling matches the bracket granularity below; a window narrower than
        # one doubling step is missed and falls through to the floor fallback.
        probe = lo; entered = false
        while probe < 1.0e4
            probe *= 2
            if frac(probe) >= cfg.cap_target
                lo = probe; entered = true; break
            end
        end
        entered || return floor0   # no feasible window found; degenerate bands
    end
    hi = 2 * lo
    while frac(hi) >= cfg.cap_target && hi < 1.0e4
        lo = hi; hi *= 2
    end
    while (hi - lo) / hi > tol_frac
        mid = 0.5 * (lo + hi)
        frac(mid) >= cfg.cap_target ? (lo = mid) : (hi = mid)
    end
    return round(lo; digits = 2)
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
    cfg::SamplerConfig,
    n_samples::Int;
    rng::AbstractRNG = default_rng(),
)
    T = eltype(get_flat_start(power_system))
    sys = _build_sampler_system(power_system, cfg.dispatchable)

    cap = _calibrate_capacity(sys, cfg, rng)
    L_MAX = cfg.load_max_frac * cap
    L_MIN = cfg.load_min_frac * L_MAX

    # Optional slack-close: free the generator P controls (distributed slack) and
    # drive each in-band stationary draw to AC-feasibility via solve_distributed_slack!.
    slack_idx = cfg.close_slack ? sys.gen_P_idx : nothing

    states = Vector{PowerFlowState{T}}(undef, 0)
    u = zeros(T, power_system.n_controls)
    dl = zeros(Float64, sys.n_load); dg = zeros(Float64, sys.n_gen)
    vt = zeros(Float64, sys.n_gen)

    attempts = 0
    # Rejection-reason tally. The first three partition every non-accepted
    # attempt (prescreen + diverged + rejected + accepted == attempts); the
    # residual/v_low/v_high sub-counts attribute each :rejected outcome and may
    # overlap (a state can fail more than one in-band test at once).
    n_prescreen = 0; n_diverged = 0; n_rejected = 0
    n_rej_residual = 0; n_rej_vlow = 0; n_rej_vhigh = 0
    max_attempts = cfg.max_attempts_mult * n_samples
    while length(states) < n_samples && attempts < max_attempts
        attempts += 1
        _draw_controls!(u, sys, cfg, rng, dl, dg, vt, L_MIN, L_MAX)
        tag, state, kr, _, vlo, vhi, lp = _evaluate(sys, u, cfg; slack_idx)
        if tag === :ok
            push!(states, state)
        elseif tag === :prescreen
            n_prescreen += 1
        elseif tag === :diverged
            n_diverged += 1
        else  # :rejected — in-band test failed; attribute which
            n_rejected += 1
            kr > (cfg.close_slack ? cfg.tau_feasible : cfg.tau_p_rel * lp) && (n_rej_residual += 1)
            vlo < cfg.v_min && (n_rej_vlow += 1)
            vhi > cfg.v_max && (n_rej_vhigh += 1)
        end
    end

    length(states) >= n_samples || error(
        "balanced sampler kept only $(length(states)) / $n_samples after " *
        "$attempts attempts (capacity $cap, load range [$L_MIN, $L_MAX]); " *
        "loosen the bands or raise max_attempts_mult")

    dataset = PowerFlowDataset(power_system, states)
    diagnostics = SamplerDiagnostics(; capacity = cap, attempts = attempts,
        accepted = length(states),
        extra = Dict{String,Any}(
            "yield"      => attempts > 0 ? length(states) / attempts : 0.0,
            "load_range" => [L_MIN, L_MAX],
            "rejections" => Dict{String,Any}(
                "prescreen" => n_prescreen,
                "diverged"  => n_diverged,
                "rejected"  => n_rejected),
            "reject_detail" => Dict{String,Any}(
                "residual" => n_rej_residual,
                "v_low"    => n_rej_vlow,
                "v_high"   => n_rej_vhigh)))
    return dataset, diagnostics
end
