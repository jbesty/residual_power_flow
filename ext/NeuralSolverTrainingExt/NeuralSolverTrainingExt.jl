module NeuralSolverTrainingExt

import Zygote
import Optimisers
import Optim
import Optimization
import OptimizationOptimJL
import OptimizationOptimisers
using Statistics: mean, std, median
using LinearAlgebra: norm, Diagonal, I, Symmetric, eigen
using SparseArrays: spzeros
using Random: AbstractRNG, default_rng
using Printf: @printf

using ResidualPowerFlow
import ResidualPowerFlow: fit!, diagnose, fit_data_transformation,
    LBFGSTraining, AdamTraining, parse_training,
    training_log, training_summary, write_training_log_csv
using ResidualPowerFlow: voltages_matrix, controls_matrix,
    PowerSystem, PowerFlowDataset,
    DataTransformation,
    normalize_controls, normalize_voltages, denormalize_voltages,
    compute_energy, compute_energy_hessian, evaluate_predictions,
    _compute_hessian_blocks, _symmetric_sqrt

include("config_types.jl")
include("training_utils.jl")
include("fit_data_transformation.jl")
include("fit.jl")
include("parse_training.jl")
include("training_log.jl")
include("diagnose.jl")

end # module NeuralSolverTrainingExt
