
export evaluate_predictions

import LinearAlgebra: norm

function evaluate_predictions(model::PowerFlowSolver, dataset::PowerFlowDataset)
    energies = [compute_energy(state) for state in solve(dataset, model).states]
    return (; energy = energies, residual_norm = sqrt.(2 .* energies))
end

function evaluate_predictions(model::PowerFlowSolver, states::AbstractVector{<:PowerFlowState})
    isempty(states) && return (; energy = Float64[], residual_norm = Float64[])
    return evaluate_predictions(model, PowerFlowDataset(first(states).power_system, states))
end

