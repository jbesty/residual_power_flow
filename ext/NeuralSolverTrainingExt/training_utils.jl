# --- Training-only utilities ---

# Pre-normalise controls and voltages from `dataset` into the matrix layout
# expected by the Lux model — n_features × n_samples, model float type.
# Called once before training so the gradient objective never repeats the work.
function _prepare_gradient_matrices(solver::NeuralSolver, dataset::PowerFlowDataset)
    t = solver.transformation
    T = something(ResidualPowerFlow._model_eltype(solver.parameters), Float64)
    X = T.(Matrix(normalize_controls(t, controls_matrix(dataset))'))   # n_controls × n_samples
    Y = T.(Matrix(normalize_voltages(t, voltages_matrix(dataset)))')    # n_variables × n_samples
    return X, Y, size(X, 2)
end


