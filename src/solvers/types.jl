export JacobianMethod, DenseAD, CentralDifferenceApproximation, ExplicitAnalytical
export SemiAnalytical, GaussNewtonApproximation

# --- Jacobian / Hessian method dispatch types ---

abstract type JacobianMethod end
struct DenseAD <: JacobianMethod end  # ForwardDiff dense
struct CentralDifferenceApproximation <: JacobianMethod end  # central finite differences — reference
struct ExplicitAnalytical <: JacobianMethod end  # hand-coded per-component derivatives
struct SemiAnalytical <: JacobianMethod end       # analytical J + ForwardDiff per-component correction
struct GaussNewtonApproximation <: JacobianMethod end  # J^T J only (no correction term)

# --- Solver options ---

export SolverOptions

struct SolverOptions
    max_iterations::Int
    tolerance::Float64
    verbose::Bool
    step_size_variables::Float64
    step_size_controls::Float64
    jacobian_method::JacobianMethod
end

function SolverOptions(;
    max_iterations::Int = 20,
    tolerance::Real = 1e-6,
    verbose::Bool = false,
    step_size_variables::Real = 1.0,
    step_size_controls::Real = 1.0,
    jacobian_method::JacobianMethod = ExplicitAnalytical(),
)
    return SolverOptions(
        max_iterations,
        Float64(tolerance),
        verbose,
        Float64(step_size_variables),
        Float64(step_size_controls),
        jacobian_method,
    )
end

export PowerFlowSolver

# Neural / surrogate solvers implement step!(state, solver) as their primary interface.
abstract type PowerFlowSolver end
