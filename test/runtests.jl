using ResidualPowerFlow
using Test
using Random: MersenneTwister
using LinearAlgebra
using ForwardDiff
import ResidualPowerFlow: PowerSystemBuilder, add_one_bus_injector!, add_two_bus_injector!, build!,
    compute_cycles_and_participation, compute_Ybus, check_bus_reference

# Heavy deps loaded unconditionally (the suite is a single tier): JLD2 for the
# case9 fixture + dataset IO, and the Lux + training stack for NeuralSolver
# fit!/evaluate and the save_solver/load_solver round-trip.
using JLD2
using Lux
using Zygote, Optimisers
using Optimization, OptimizationOptimisers, OptimizationOptimJL
# CommonSolve (pulled in by SciMLBase) exports solve/solve!/step! — pin the RPF
# bindings explicitly to avoid the ambiguity error in tests.
import ResidualPowerFlow: solve, solve!, step!

# Bring extension-defined config types into scope for bare-name use in tests.
# MLPArchitecture lives in LuxExt; the training configs in NeuralSolverTrainingExt.
# Julia's package-extension mechanism doesn't expose ext names through
# `using ResidualPowerFlow.LuxExt: X`; go through Base.get_extension.
const _LuxExt = Base.get_extension(ResidualPowerFlow, :LuxExt)
const _NSTE   = Base.get_extension(ResidualPowerFlow, :NeuralSolverTrainingExt)
const MLPArchitecture     = _LuxExt.MLPArchitecture
const AdamTraining        = _NSTE.AdamTraining
const LBFGSTraining       = _NSTE.LBFGSTraining
const TrainingDiagnostics = _NSTE.TrainingDiagnostics
const BestOf              = _NSTE.BestOf
const BestOfResult        = _NSTE.BestOfResult

# Balanced-sampler config for tests. cap_samples is reduced from the default
# (120) to keep the per-call capacity scan fast. Deterministic given a fixed rng.
const CASE9_OC_SETUP = SamplerConfig(cap_samples = 20)

# The suite is a single tier (the SLOW/FAST split and the sysimage were removed
# for the public release). A handful of legacy slow-only blocks remain gated on
# this flag; they stay off.
const SLOW_TESTS = false

# case9 fixture: a near-exact RPF fixed point (r ≈ 5e-7 at the stored solution),
# deserialized from JLD2 so
# the suite needs no Powsybl. Powsybl-dependent testsets are gated on
# `_powsybl_available`, which is false on this path.
const _powsybl_available = false
const _fixture_data = JLD2.load(joinpath(@__DIR__, "fixtures", "case9.jld2"))
const _CASE9 = (power_system = _fixture_data["power_system"],)

# Lightweight accessors so tests can call controls_from_solution / target_variables
# without PowsyblIOExt loaded. Return a fresh copy each call (PowsyblIOExt's
# controls_from_solution allocates) — otherwise tests that mutate the returned
# vector would silently corrupt the shared fixture for later tests.
ResidualPowerFlow.controls_from_solution(::Any) = copy(_fixture_data["controls"])
ResidualPowerFlow.target_variables(::Any) = copy(_fixture_data["variables"])

@testset "ResidualPowerFlow.jl" begin
    include("core_types.jl")
    include("current_injections.jl")
    include("power_system_construction.jl")
    include("jacobians.jl")
    include("test_distributed_slack_verification.jl")
    include("test_topology_switching.jl")

    include("fixed_point_tests.jl")
    include("float32_type_preservation.jl")
    include("compute_energy_hessian.jl")
    include("learning_distribution.jl")
    include("learning_sampler.jl")
    include("learning_loss_energy.jl")
    include("learning_dataset_contract.jl")
    include("learning_normalisation.jl")
    include("learning_io.jl")
    include("smoke.jl")
end
