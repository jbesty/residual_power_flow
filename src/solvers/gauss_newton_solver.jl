
import LinearAlgebra: norm

export GaussNewtonSolver

# ── GaussNewtonSolver ──────────────────────────────────────────────────────────
#
# Fieldless algorithm tag. The state owns the PowerSystem and all mutable data.

struct GaussNewtonSolver <: PowerFlowSolver end

# ── step! ──────────────────────────────────────────────────────────────────────
#
# Pure voltage update: residual → Jacobian → linear solve → update voltages.
# No control logic, no branching. PtO-related logic lives in pto.jl.
#
# Convergence metric: ‖J'r‖ (gradient of ½‖r‖²). The old max‖Δv‖ criterion
# fired at Float32 roundoff floor on overdetermined systems before reaching a
# genuine stationary point; J'r = 0 is the correct stationarity condition.

function step!(
    state::PowerFlowState,
    solver::GaussNewtonSolver;
    solver_options::SolverOptions = SolverOptions(),
)
    ps = state.power_system
    v, u = state.voltages, state.controls
    jm = solver_options.jacobian_method
    st = state.statuses
    cycle_kw = _cycle_kw(state)

    residual = compute_residual!(state)
    jacobian_voltages = compute_jacobian_voltages(ps, v, u, jm; statuses = st, cycle_kw...)

    variables_update = -(jacobian_voltages \ residual)
    controls_update = fill!(state.controls_update_buffer, zero(eltype(v)))

    state.voltages .+= solver_options.step_size_variables .* variables_update
    compute_current_balance!(state)

    max_update = norm(jacobian_voltages' * residual)
    return (; max_update, residual, variables_update, controls_update, jacobian_voltages)
end
