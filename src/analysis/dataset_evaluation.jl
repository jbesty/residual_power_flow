
export evaluate_predictions
export evaluate_solvers
export solve_and_compare

import LinearAlgebra: norm

function evaluate_predictions(model::PowerFlowSolver, dataset::PowerFlowDataset)
    energies = [compute_energy(s) for s in solve(dataset, model).states]
    return (; energy = energies, residual_norm = sqrt.(2 .* energies))
end

function evaluate_predictions(model::PowerFlowSolver, states::AbstractVector{<:PowerFlowState})
    isempty(states) && return (; energy = Float64[], residual_norm = Float64[])
    return evaluate_predictions(model, PowerFlowDataset(first(states).power_system, states))
end

# Solve a state exactly with the iterative solver and compare against the provided state.
# Useful for quantifying how far an approximate state (e.g. from a surrogate) is from
# the true AC solution.
#
# Returns a NamedTuple:
#   energy_approx  — RPF energy of the provided state
#   energy_exact   — RPF energy after solving to convergence from the provided voltages
#   voltage_error  — ||v_exact - v_approx|| (Euclidean distance in voltage space)
#   converged      — whether the exact solve converged
function solve_and_compare(state::PowerFlowState; kwargs...)
    exact, stats = solve(state; kwargs...)
    return (;
        energy_approx = compute_energy(state),
        energy_exact  = compute_energy(exact),
        voltage_error = norm(exact.voltages .- state.voltages),
        converged     = stats["converged"],
    )
end

function evaluate_solvers(
    solvers::AbstractVector{<:Pair{String, <:PowerFlowSolver}},
    dataset::PowerFlowDataset,
)
    v_true = voltages_matrix(dataset)
    N = length(dataset)
    results = NamedTuple[]
    for (label, solver) in solvers
        ep = evaluate_predictions(solver, dataset)
        v_pred = voltages_matrix(solve(dataset, solver))
        voltage_error = Vector{Float64}(undef, N)
        for i in 1:N
            voltage_error[i] = norm(view(v_pred, i, :) .- view(v_true, i, :))
        end
        push!(results, (;
            label         = label,
            energy        = ep.energy,
            residual_norm = ep.residual_norm,
            voltage_error = voltage_error,
        ))
    end
    return results
end

