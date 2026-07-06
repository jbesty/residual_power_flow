# --- Unified fit! for NeuralSolver ---
#
# cfg.optimiser is passed directly to Optimization.solve:
#   Optimisers.AbstractRule (Adam, …) → OptimizationOptimisers backend
#   Optim.AbstractOptimizer (LBFGS, …) → OptimizationOptimJL backend
#
# `datasets` is a (train, val) tuple. The caller is responsible for splitting;
# the validation dataset must be at least 10 % of the training dataset.
# Always returns (solver, TrainingDiagnostics).
#
# Hot path  — called on every gradient evaluation:
#   model(X, ps, initial_states)  pre-normalised matrices, no allocation
#
# Cold path — called once per optimiser step via step_cb!:
#   predict + nondimensional_loss  (validation loss, energy diagnostics)

# Log-spaced iteration indices in [1, maxit] for energy recording: dense at the
# start (the first integers all appear) and thinning geometrically — captures the
# fast early L-BFGS descent while bounding the recorded points to ~npts. Far cheaper
# than every-step on a log-axis convergence plot, where linear sampling wastes points.
function _energy_log_iters(maxit::Int, npts::Int)
    (maxit <= 1 || npts <= 1) && return Set(1:max(maxit, 1))
    return Set(round.(Int, exp.(range(0.0, log(maxit), length = npts))))
end

function fit!(
    solver::NeuralSolver,
    datasets::Tuple{PowerFlowDataset,PowerFlowDataset},
    cfg::NeuralTraining;
    energy_every::Int = 1,
    energy_logpoints::Int = 0,
)
    energy_every >= 1 || throw(ArgumentError("energy_every must be ≥ 1, got $energy_every"))
    # energy_logpoints > 0 ⇒ record ρ at ~that many log-spaced iterations (early
    # iterations captured); otherwise fall back to linear every-`energy_every` steps.
    energy_iters = energy_logpoints > 0 ?
        _energy_log_iters(cfg.max_iterations, energy_logpoints) : nothing
    # ── Unpack and validate train / validation datasets ──────────────────────
    train_dataset, val_dataset = datasets
    length(val_dataset) >= 0.1 * length(train_dataset) || throw(
        ArgumentError(
            "validation dataset ($(length(val_dataset)) samples) is less than " *
            "10 % of the training dataset ($(length(train_dataset)) samples)",
        ),
    )

    # ── Pre-normalise training matrices for the gradient objective ───────────
    # X and Y are reused on every gradient evaluation; computing them here
    # avoids redundant normalisation inside the hot path.
    X, Y, n_train_samples = _prepare_gradient_matrices(solver, train_dataset)

    # ── Training state ───────────────────────────────────────────────────────
    diag = TrainingDiagnostics()
    best_val_loss = Ref(Inf)
    best_step = Ref(0)
    initial_flat_params = copy(solver.parameters)
    best_flat_params = copy(initial_flat_params)
    best_states = Ref(solver.states)
    iter_count = Ref(0)
    initial_states = solver.states

    # ── Per-step callback ────────────────────────────────────────────────────
    # Called by both the Optimisers and Optim backends after every gradient step.
    step_cb! = function (flat_params, training_loss_value)
        iter_count[] += 1

        # Validation loss (cold path — not on the gradient tape). Recorded every
        # step because it drives best-parameter selection below.
        val_loss = mean(predict_nondimensional_loss(val_dataset, solver, flat_params))
        push!(diag.training_loss, Float64(training_loss_value))
        push!(diag.validation_loss, val_loss)

        # Energy diagnostics are the cold-path bottleneck (per-sample predict_energies
        # on train AND val). Record on a schedule: log-spaced iterations when
        # energy_logpoints > 0 (dense early, thinning), else every `energy_every`-th
        # step. The energy vectors are generally shorter than the loss vectors, so
        # `energy_steps` stores the matching iteration for each recorded ρ point.
        record_energy = energy_iters === nothing ?
            (iter_count[] % energy_every == 0) : (iter_count[] in energy_iters)
        if record_energy
            energies_val   = predict_energies(val_dataset, solver, flat_params)
            energies_train = predict_energies(train_dataset, solver, flat_params)
            push!(diag.validation_energy_1norm, norm(energies_val, 1))
            push!(diag.validation_energy_2norm, norm(energies_val, 2))
            push!(diag.validation_energy_infnorm, norm(energies_val, Inf))
            push!(diag.training_energy_median, median(energies_train))
            push!(diag.validation_energy_median, median(energies_val))
            push!(diag.energy_steps, iter_count[])
        end

        if isfinite(val_loss) && val_loss < best_val_loss[]
            best_val_loss[] = val_loss
            best_step[] = iter_count[]
            copyto!(best_flat_params, flat_params)
            _, best_states[] = solver.model(X, flat_params, initial_states)
        end

        if cfg.print_every > 0 && iter_count[] % cfg.print_every == 0
            @printf("  iter %5d  loss = %.3e\n", iter_count[], training_loss_value)
        end
    end

    # ── Optimization loop ────────────────────────────────────────────────────
    function training_loss(flat_params, _)
        ŷ, _ = solver.model(X, flat_params, initial_states)
        return sum(abs2, ŷ .- Y) / (2 * n_train_samples)
    end
    # OptimizationOptimJL wraps every Optim.jl method in a TwiceDifferentiable,
    # which preallocates a dense n×n Hessian (alloc_H) — O(n²) memory a first-order
    # method never touches (≈4.6 GB at case57's ~24k params, enough to OOM). Handing
    # it a sparse `hess_prototype` makes that buffer allocate via `similar(prototype)`
    # at zero cost. L-BFGS never reads H, so results are unchanged; only first-order
    # Optim methods get this — Adam (Optimisers) and any second-order method are
    # untouched, the latter still receiving a real Hessian.
    objective = if cfg.optimiser isa Optim.FirstOrderOptimizer
        n_params = length(initial_flat_params)
        Optimization.OptimizationFunction(training_loss, Optimization.AutoZygote();
            hess = (H, θ) -> nothing,
            hess_prototype = spzeros(eltype(initial_flat_params), n_params, n_params))
    else
        Optimization.OptimizationFunction(training_loss, Optimization.AutoZygote())
    end
    problem = Optimization.OptimizationProblem(objective, initial_flat_params)
    # save_best=false: OptimizationOptimisers fires an extra callback on the last
    # iteration when save_best=true; disable to get exactly max_iterations calls.
    backend_kwargs = if cfg.optimiser isa Optimisers.AbstractRule
        (; save_best = false)
    elseif isfinite(cfg.abstol)
        (; abstol = cfg.abstol)
    else
        (;)
    end
    elapsed = @elapsed result = Optimization.solve(
        problem,
        cfg.optimiser;
        maxiters = cfg.max_iterations,
        callback = (state, lv) -> (step_cb!(state.u, lv); false),
        backend_kwargs...,
    )

    # ── Apply best parameters and capture final states ───────────────────────
    solver.parameters = isfinite(best_val_loss[]) ? best_flat_params : result.u
    solver.states = best_states[]
    diag.best_step = best_step[]
    push!(diag.phase_seconds, elapsed)

    return solver, diag
end

# --- Multi-phase training ---
#
# Run `phases` in order, each continuing from the solver's current weights, and
# return a single TrainingDiagnostics whose curves are the concatenation across
# phases. Equivalent to calling the single-phase fit! once per phase in sequence —
# which is exactly what it does. `phase_steps` records the cumulative recorded-step
# count at each phase boundary so a plot can mark optimiser transitions, and
# `best_step` indexes the applied params within the concatenated curve (the solver
# is left holding the LAST phase's best-validation iterate, as in a sequential run).
function fit!(
    solver::NeuralSolver,
    datasets::Tuple{PowerFlowDataset,PowerFlowDataset},
    phases::AbstractVector{<:NeuralTraining};
    energy_every::Int = 1,
    energy_logpoints::Int = 0,
)
    isempty(phases) && throw(ArgumentError("phases must be non-empty"))
    combined = TrainingDiagnostics()
    step_offset = 0
    for cfg in phases
        _, d = fit!(solver, datasets, cfg; energy_every, energy_logpoints)
        append!(combined.training_loss,             d.training_loss)
        append!(combined.validation_loss,           d.validation_loss)
        append!(combined.validation_energy_1norm,   d.validation_energy_1norm)
        append!(combined.validation_energy_2norm,   d.validation_energy_2norm)
        append!(combined.validation_energy_infnorm, d.validation_energy_infnorm)
        append!(combined.training_energy_median,    d.training_energy_median)
        append!(combined.validation_energy_median,  d.validation_energy_median)
        append!(combined.energy_steps,              d.energy_steps .+ step_offset)
        # The solver now holds THIS phase's best-validation params, so the globally
        # applied best_step is this phase's best_step shifted past the prior phases.
        combined.best_step = d.best_step == 0 ? 0 : step_offset + d.best_step
        step_offset += length(d.training_loss)
        push!(combined.phase_steps, step_offset)
        append!(combined.phase_seconds, d.phase_seconds)
    end
    return solver, combined
end

# --- Best-of / multi-start training ---
#
# Run every candidate schedule from the SAME initial weights (cloned once, reset
# before each candidate) and keep the candidate with the best validation
# `select_by`. Selection uses validation only — test is never an argument to fit!.
# Leaves `solver` at the winner's (best-validation) params and returns a
# BestOfResult with the winner's diagnostics, every candidate's diagnostics, the
# winner index, and the per-candidate validation select values.
function fit!(
    solver::NeuralSolver,
    datasets::Tuple{PowerFlowDataset,PowerFlowDataset},
    cfg::BestOf;
    energy_every::Int = 1,
    energy_logpoints::Int = 0,
)
    isempty(cfg.candidates) && throw(ArgumentError("BestOf requires at least one candidate"))
    initial_params = deepcopy(solver.parameters)
    initial_states = solver.states

    n = length(cfg.candidates)
    cand_diags    = Vector{TrainingDiagnostics}(undef, n)
    cand_params   = Vector{Any}(undef, n)
    cand_states   = Vector{Any}(undef, n)
    select_values = Vector{Float64}(undef, n)

    for (i, phases) in enumerate(cfg.candidates)
        # Reset to the identical initial weights before each candidate.
        solver.parameters = deepcopy(initial_params)
        solver.states     = initial_states
        _, d = fit!(solver, datasets, phases; energy_every, energy_logpoints)
        cand_diags[i]    = d
        cand_params[i]   = deepcopy(solver.parameters)
        cand_states[i]   = solver.states
        select_values[i] = _select_value(cfg.select_by, d)
    end

    winner = argmin(select_values)
    solver.parameters = cand_params[winner]
    solver.states     = cand_states[winner]

    return solver, BestOfResult(winner, cand_diags[winner], cand_diags, select_values)
end

# Validation metric used to rank BestOf candidates. :val_loss is the validation
# loss of the APPLIED model — the best-validation iterate the solver is left at
# (validation_loss[best_step]); for a single-phase run that is the run minimum.
function _select_value(select_by::Symbol, d::TrainingDiagnostics)
    select_by === :val_loss || throw(ArgumentError("unsupported BestOf select_by $(repr(select_by))"))
    isempty(d.validation_loss) && return Inf
    return d.best_step == 0 ? minimum(d.validation_loss) : d.validation_loss[d.best_step]
end
