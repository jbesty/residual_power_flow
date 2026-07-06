isnothing(_CASE9) && return

# End-to-end functional smoke for the trimmed public release. Exercises the
# load-bearing path: build a system → distributed-slack feasibility close →
# feasible-sample with both kept samplers → fit! → evaluate → save/load.
# The energy-Hessian ExplicitAnalytical ≈ DenseAD agreement is covered separately
# by compute_energy_hessian.jl.

@testset "smoke: distributed-slack drives a sampled OC to feasibility" begin
    ps = _CASE9.power_system
    gen_P_idx = ResidualPowerFlow.control_indices(ps, SynchronousMachineStatic, :P)

    # A stationary (deliberately light-infeasible) balanced OC, then free the
    # generator-P distributed slack and drive ‖r‖ → 0.
    ds = generate_dataset(ps, CASE9_OC_SETUP, 1; rng = MersenneTwister(11))
    state = deepcopy(ds.states[1])
    stats = solve_distributed_slack!(state, gen_P_idx; tol = 1e-8, max_iterations = 100)
    @test stats["converged"]
    @test stats["residual_norm"] ≤ 1e-6
end

@testset "smoke: feasible sampler produces AC-feasible data" begin
    ps  = _CASE9.power_system
    cfg = FeasibleSamplerConfig(max_attempts_mult = 200)
    ds  = generate_dataset(ps, cfg, 5; rng = MersenneTwister(7))
    @test length(ds) == 5
    @test all(feasible(ds))
end

@testset "smoke: balanced close_slack produces AC-feasible data" begin
    ps  = _CASE9.power_system
    cfg = SamplerConfig(cap_samples = 20, close_slack = true)
    ds  = generate_dataset(ps, cfg, 5; rng = MersenneTwister(3))
    @test length(ds) == 5
    @test all(feasible(ds))
end

@testset "smoke: fit! → evaluate → save_solver/load_solver round-trip" begin
    ps    = _CASE9.power_system
    train = generate_dataset(ps, CASE9_OC_SETUP, 40; rng = MersenneTwister(42))
    val   = generate_dataset(ps, CASE9_OC_SETUP, 10; rng = MersenneTwister(43))

    transformation = fit_data_transformation(train, ps;
        controls_scheme = :standardised, voltages_scheme = :standardised)
    solver = NeuralSolver(ps;
        architecture   = MLPArchitecture(n_hidden_layers = 2, n_neurons = 16),
        transformation = transformation,
        rng            = MersenneTwister(42))
    fit!(solver, (train, val), LBFGSTraining(max_iterations = 10))

    ev = evaluate_predictions(solver, val)
    @test all(isfinite, ev.energy)
    @test all(isfinite, ev.residual_norm)

    U = controls_matrix(train)
    pred_before = ResidualPowerFlow.predict(U, solver)

    path = joinpath(mktempdir(), "solver.jld2")
    @test save_solver(path, solver) == path
    loaded = load_solver(path)
    @test loaded isa NeuralSolver
    pred_after = ResidualPowerFlow.predict(U, loaded)
    @test pred_after ≈ pred_before atol = 1e-10 rtol = 1e-10
end
