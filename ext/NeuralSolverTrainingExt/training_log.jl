# --- Tidy training-log export (Part D) ---
#
# training_log turns a TrainingDiagnostics (or a BestOfResult) into a tidy,
# serialization-ready table of rows
#
#   (candidate, phase, step, set ∈ {:train,:val}, metric ∈ {:loss,:rho}, value)
#
# so a plot or CSV gets loss and energy (ρ = median ½‖r‖²) side by side per step.
# `:loss` rows appear every step; `:rho` rows only at the recorded energy steps
# (the energy schedule is subsampled — see fit! energy_every / energy_logpoints).
# training_summary returns a one-row NamedTuple (best step, final/best val loss
# and ρ, total/per-phase wall-clock). write_training_log_csv is a tiny
# dependency-free writer. No plotting lives in RPF — this is just the data.

# Which 1-based phase a step belongs to, from the cumulative phase_steps boundaries.
# Empty phase_steps (single-phase fit!) ⇒ everything is phase 1.
function _phase_of_step(step::Int, phase_steps::AbstractVector{Int})
    isempty(phase_steps) && return 1
    p = findfirst(boundary -> step <= boundary, phase_steps)
    return p === nothing ? length(phase_steps) : p
end

const _LOG_ROW = NamedTuple{(:candidate, :phase, :step, :set, :metric, :value),
                            Tuple{Int,Int,Int,Symbol,Symbol,Float64}}

function training_log(diag::TrainingDiagnostics; candidate::Int = 1)
    rows = _LOG_ROW[]
    ps = diag.phase_steps
    for s in 1:length(diag.training_loss)
        ph = _phase_of_step(s, ps)
        push!(rows, (candidate = candidate, phase = ph, step = s, set = :train, metric = :loss,
                     value = diag.training_loss[s]))
        push!(rows, (candidate = candidate, phase = ph, step = s, set = :val, metric = :loss,
                     value = diag.validation_loss[s]))
    end
    for (i, s) in enumerate(diag.energy_steps)
        ph = _phase_of_step(s, ps)
        push!(rows, (candidate = candidate, phase = ph, step = s, set = :train, metric = :rho,
                     value = diag.training_energy_median[i]))
        push!(rows, (candidate = candidate, phase = ph, step = s, set = :val, metric = :rho,
                     value = diag.validation_energy_median[i]))
    end
    return rows
end

# Tidy log across all BestOf candidates (candidate column = 1-based candidate index).
function training_log(result::BestOfResult)
    rows = _LOG_ROW[]
    for (i, d) in enumerate(result.candidate_diagnostics)
        append!(rows, training_log(d; candidate = i))
    end
    return rows
end

# One-row summary of a single run. best_val_* are taken at the applied
# (best-validation) iterate; final_val_* at the last recorded step.
function training_summary(diag::TrainingDiagnostics; candidate::Int = 1)
    n = length(diag.validation_loss)
    best_val_loss = n == 0 ? NaN :
        (diag.best_step == 0 ? minimum(diag.validation_loss) : diag.validation_loss[diag.best_step])
    final_val_loss = n == 0 ? NaN : diag.validation_loss[end]
    has_rho = !isempty(diag.validation_energy_median)
    return (candidate = candidate,
            best_step = diag.best_step,
            n_steps = n,
            n_phases = max(length(diag.phase_steps), 1),
            best_val_loss = best_val_loss,
            final_val_loss = final_val_loss,
            best_val_rho = has_rho ? minimum(diag.validation_energy_median) : NaN,
            final_val_rho = has_rho ? diag.validation_energy_median[end] : NaN,
            total_seconds = sum(diag.phase_seconds; init = 0.0),
            phase_seconds = copy(diag.phase_seconds))
end

# Summary for a BestOf run: the winner's summary, plus who won and the per-candidate
# validation select values so the caller can log "lbfgs vs adam_lbfgs" and the winner.
function training_summary(result::BestOfResult)
    base = training_summary(result.diagnostics; candidate = result.winner)
    return (; winner = result.winner, select_values = copy(result.select_values), base...)
end

# Dependency-free CSV writer for a training_log row vector. Symbols are written as
# their bare names (train, val, loss, rho); Float64 values at full precision.
function write_training_log_csv(path::AbstractString, rows::AbstractVector)
    open(path, "w") do io
        println(io, "candidate,phase,step,set,metric,value")
        for r in rows
            println(io, string(r.candidate), ",", string(r.phase), ",", string(r.step), ",",
                    string(r.set), ",", string(r.metric), ",", string(r.value))
        end
    end
    return path
end
