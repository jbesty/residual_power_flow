
export NeuralSolver, predict, predict_nondimensional_loss, predict_energies

# Training-only config types (AdamTraining, LBFGSTraining, TrainingDiagnostics)
# live in ext/NeuralSolverTrainingExt/config_types.jl. MLPArchitecture and
# _identity_transformation live in ext/LuxExt.jl alongside the Lux-backed
# NeuralSolver constructor.

# --- NeuralSolver ---
#
# Neural network surrogate for the power flow solution map u → v.
# The inner Lux model operates entirely in dimensionless space; step!() handles
# the dimensionful ↔ dimensionless conversion transparently via DataTransformation,
# so from the solver stack's point of view the NeuralSolver is a physical → physical
# map.
#
# Mutable struct so fit! can update parameters and states via direct field
# assignment. P is always ComponentArray (set by LuxExt constructors) so the
# flat parameter vector can be passed to model() and to Optimization.solve
# without any restructuring step.
#
# Construction: NeuralSolver(power_system; architecture, transformation) or
#               NeuralSolver(transformation; ...) — requires Lux (ext/LuxExt.jl).
# Training: fit!(solver, datasets, ::NeuralTraining) — requires
#           Lux + Zygote + Optimisers (ext/NeuralSolverTrainingExt.jl).

mutable struct NeuralSolver{M, P, S} <: PowerFlowSolver
    model::M
    parameters::P
    states::S
    transformation::DataTransformation
    opt_state::Base.RefValue{Any}
end

# Traverse a Lux parameter NamedTuple tree to find the eltype of the first array leaf.
# Used at training and inference boundaries to detect model precision (Float32 vs Float64).
_model_eltype(ps::NamedTuple) = begin
    for v in values(ps)
        T = _model_eltype(v)
        T !== nothing && return T
    end
    return nothing
end
_model_eltype(a::AbstractArray) = eltype(a)
_model_eltype(::Any) = nothing

# Physical controls → physical voltages via a single forward pass.
# Vector dispatch: single sample; matrix dispatch: N × n_controls → N × n_variables.
# Optional third argument overrides the parameter tree (used during training to
# evaluate at the current optimiser iterate without mutating solver.parameters).
function predict(u::AbstractVector, solver::NeuralSolver, parameters=solver.parameters)
    T_model = something(_model_eltype(parameters), eltype(u))
    nondimensional_controls = normalize_controls(solver.transformation, u)
    nondimensional_voltages, _ = solver.model(
        reshape(T_model.(nondimensional_controls), :, 1), parameters, solver.states)
    return denormalize_voltages(solver.transformation, vec(nondimensional_voltages))
end

function predict(U::AbstractMatrix, solver::NeuralSolver, parameters=solver.parameters)
    T_model = something(_model_eltype(parameters), eltype(U))
    nondimensional_controls = T_model.(Matrix(normalize_controls(solver.transformation, U)'))
    nondimensional_voltages, _ = solver.model(nondimensional_controls, parameters, solver.states)
    return denormalize_voltages(solver.transformation, Matrix(nondimensional_voltages'))
end

# Per-sample nondimensional loss (same quantity minimised during training) from a
# NeuralSolver prediction on a dataset.
# Optional parameters override lets training diagnostics evaluate at the
# current optimiser iterate without mutating solver.parameters.
function predict_nondimensional_loss(dataset::PowerFlowDataset, solver::NeuralSolver,
                                     parameters=solver.parameters)
    U = controls_matrix(dataset)
    V = predict(U, solver, parameters)
    return nondimensional_loss(solver.transformation, V, voltages_matrix(dataset))
end

# Per-sample RPF energies from a NeuralSolver prediction on a dataset.
# Optional parameters override lets training diagnostics evaluate at the
# current optimiser iterate without mutating solver.parameters.
function predict_energies(dataset::PowerFlowDataset, solver::NeuralSolver,
                          parameters=solver.parameters)
    U = controls_matrix(dataset)
    V = predict(U, solver, parameters)
    ps = dataset.power_system
    return [Float64(compute_energy(ps, view(V, i, :), view(U, i, :)))
            for i in 1:size(V, 1)]
end

function step!(
    state::PowerFlowState,
    solver::NeuralSolver;
    solver_options::SolverOptions = SolverOptions(),
)
    T = eltype(state.voltages)
    state.voltages .= predict(state.controls, solver)
    residual = compute_residual!(state)
    return (; max_update = zero(T), residual, variables_update = nothing,
              controls_update = nothing, jacobian_voltages = nothing)
end

