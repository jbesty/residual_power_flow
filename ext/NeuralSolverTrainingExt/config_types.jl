# --- Training config types ---
#
# NeuralTraining is the unified training config backed by Optimization.jl.
# AdamTraining and LBFGSTraining are convenience constructors that return
# NeuralTraining; call-site syntax is unchanged from the old struct forms.

struct NeuralTraining
    optimiser      # Optimization.jl-compatible solver
    max_iterations::Int
    print_every::Int
    abstol::Float64    # NaN = use backend default; otherwise passed to Optimization.solve
end

function AdamTraining(;
    learning_rate::Real = 1e-3,
    n_epochs::Int       = 1000,
    print_every::Int    = 0,
)
    NeuralTraining(Optimisers.Adam(learning_rate), n_epochs, print_every, NaN)
end

function LBFGSTraining(;
    m::Int              = 10,
    max_iterations::Int = 1000,
    g_tol::Real         = NaN,
    print_every::Int    = 0,
)
    NeuralTraining(Optim.LBFGS(m = m), max_iterations, print_every, Float64(g_tol))
end

# Per-step recording of loss and energy returned by every fit! call.
# "Step" means one epoch for Adam, one accepted outer iteration for L-BFGS.
# Mutable so fit! can stamp `best_step` and (for multi-phase) `phase_steps` after
# the optimisation loop; the vector fields are still grown in place during it.
mutable struct TrainingDiagnostics
    training_loss::Vector{Float64}
    validation_loss::Vector{Float64}
    validation_energy_1norm::Vector{Float64}
    validation_energy_2norm::Vector{Float64}
    validation_energy_infnorm::Vector{Float64}
    # Median energy ρ = ½‖r‖² on the train/val sets — the physical metric, tracked
    # alongside the (transformed-space) loss so plots can show loss vs ρ. Recorded
    # on a subsampled schedule (see fit! energy_every / energy_logpoints), so these
    # are generally shorter than the loss vectors; `energy_steps` holds the matching
    # iteration index for each recorded ρ point.
    training_energy_median::Vector{Float64}
    validation_energy_median::Vector{Float64}
    energy_steps::Vector{Int}
    # Iteration whose parameters were applied to the solver — the lowest-validation
    # -loss iterate (validation_loss[best_step] == minimum validation loss), not the
    # final epoch. 0 means no finite validation loss was seen (final params kept).
    best_step::Int
    # Cumulative recorded-step count at the end of each training phase (multi-phase
    # fit!): phase_steps[i] = steps through phase i, so optimiser transitions are at
    # phase_steps[1:end-1]. Empty for a single-phase fit!.
    phase_steps::Vector{Int}
    # Wall-clock seconds spent in each phase's optimisation loop. A single-phase
    # fit! records one entry; multi-phase concatenates one per phase.
    phase_seconds::Vector{Float64}
end

TrainingDiagnostics() = TrainingDiagnostics(Float64[], Float64[], Float64[], Float64[], Float64[],
                                            Float64[], Float64[], Int[], 0, Int[], Float64[])

# --- Best-of / multi-start training config ---
#
# Each candidate is a training schedule — a single NeuralTraining phase or a phase
# list (Vector{NeuralTraining}). `fit!(solver, (train, val), ::BestOf)` runs every
# candidate from the SAME initial weights and keeps the one with the best `select_by`
# metric on the VALIDATION set. Test is never an argument to fit!, so it cannot leak
# into selection. Only :val_loss is supported (the transformed-space objective fit!
# already minimises); :val_energy (median ρ) is a documented future extension.
struct BestOf
    candidates::Vector{Vector{NeuralTraining}}
    select_by::Symbol
end

_as_phase_list(c::NeuralTraining) = NeuralTraining[c]
_as_phase_list(c::AbstractVector{<:NeuralTraining}) = collect(NeuralTraining, c)

function BestOf(candidates::AbstractVector; select_by::Symbol = :val_loss)
    select_by === :val_loss || throw(ArgumentError(
        "BestOf select_by must be :val_loss (got $(repr(select_by))); " *
        ":val_energy selection is a future extension"))
    isempty(candidates) && throw(ArgumentError("BestOf requires at least one candidate"))
    return BestOf([_as_phase_list(c) for c in candidates], select_by)
end

# Result of a BestOf run: the winner's diagnostics plus every candidate's
# diagnostics and the per-candidate validation `select_by` values, so the caller
# can log both legs and who won. `winner` indexes `candidate_diagnostics`.
struct BestOfResult
    winner::Int
    diagnostics::TrainingDiagnostics
    candidate_diagnostics::Vector{TrainingDiagnostics}
    select_values::Vector{Float64}
end
