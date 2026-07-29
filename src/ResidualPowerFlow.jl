module ResidualPowerFlow

include("core/types.jl")
include("solvers/types.jl")

include("core/component_functions.jl")
include("core/balance.jl")
include("core/network_matrices.jl")
include("core/accessors.jl")
include("core/power_system_construction.jl")

include("core/jacobians.jl")
include("core/jacobians_explicit.jl")
include("core/energy_hessian.jl")

include("solvers/solve_utils.jl")
include("solvers/gauss_newton_solver.jl")
include("solvers/solve.jl")
include("solvers/distributed_slack.jl")

# Dataset container + artifact contract (src/dataset/), the OC sampler subsystem
# (src/sampling/), dataset analysis tools (src/analysis/), and the neural/ML-specific
# pieces (src/learning/). All live in this one module, so the folders are
# organisational only — include order still respects definition deps.
include("dataset/dataset.jl")
include("learning/normalisation.jl")
include("analysis/dataset_evaluation.jl")
include("analysis/dataset_distribution.jl")
include("sampling/sampler_strategy.jl")
include("sampling/sampling_operating_condition.jl")
include("sampling/feasible_sampler.jl")
include("dataset/config.jl")
include("learning/neural_solver.jl")

# ── Training extension stubs ───────────────────────────────────────────────────
# fit!, fit_data_transformation, LBFGSTraining, and AdamTraining are
# implemented in ext/NeuralSolverTrainingExt.jl when Lux + Zygote + Optimisers
# are loaded.
export fit!, fit_data_transformation
export LBFGSTraining, AdamTraining, parse_training
export training_log, training_summary, write_training_log_csv
function fit! end
function fit_data_transformation end
function LBFGSTraining end
function AdamTraining end
function parse_training end
function training_log end
function training_summary end
function write_training_log_csv end

# ── PowSyBl IO extension stubs ─────────────────────────────────────────────────
# Implemented in ext/PowsyblIOExt.jl when Powsybl is loaded.
export control_indices
export build_powersystem, controls_from_solution, target_variables

function build_powersystem end
function controls_from_solution end
function target_variables end

# ── JLD2 IO extension stubs ───────────────────────────────────────────────────
# Implemented in ext/JLD2Ext.jl when JLD2 is loaded.
export save_dataset, load_dataset
export save_solver, load_solver
export dataset_is_current, save_config_dataset, load_train_dataset, load_test_dataset
export load_or_generate_datasets

function save_dataset end
function load_dataset end
function save_solver end
function load_solver end

# Portable NeuralSolver serialization bridge. LuxExt builds/rebuilds the plain
# portable form (architecture + flat params + DataTransformation); JLD2Ext writes
# and reads it. Internal — not exported. The from-portable fallback errors clearly
# when Lux is absent, since rebuilding the Lux model requires it.
function _solver_to_portable end
_solver_from_portable(::Any) = error(
    "reconstructing a saved NeuralSolver requires Lux — add `using Lux` and retry")
function dataset_is_current end
function save_config_dataset end
function load_train_dataset end
function load_test_dataset end
function load_or_generate_datasets end

end # module
