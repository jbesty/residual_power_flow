isnothing(_CASE9) && return

import ResidualPowerFlow: control_indices, calibrate_capacity, SamplerConfig

@testset "balanced sampler — case9" begin
    ps = _CASE9.power_system

    @testset "reproducible under a fixed seed" begin
        d1 = generate_dataset(ps, CASE9_OC_SETUP, 6; rng = MersenneTwister(7))
        d2 = generate_dataset(ps, CASE9_OC_SETUP, 6; rng = MersenneTwister(7))
        @test controls_matrix(d1) == controls_matrix(d2)
        @test voltages_matrix(d1) == voltages_matrix(d2)
    end

    @testset "character: voltage band + stationarity + residual asymmetry" begin
        cfg = CASE9_OC_SETUP
        dataset = generate_dataset(ps, cfg, 20; rng = MersenneTwister(42))
        @test length(dataset) == 20

        V = voltages_matrix(dataset)[:, 1:ps.n_buses]
        @test all(cfg.v_min .<= V .<= cfg.v_max)

        ctrl = controls_matrix(dataset)
        load_P_idx = control_indices(ps, ZIPLoad, :P)
        for s in dataset.states
            # stationarity (GN step below tol), NOT feasibility (‖r‖≈0)
            stats = solve!(deepcopy(s), GaussNewtonSolver();
                solver_options = SolverOptions(max_iterations = 100, tolerance = 1e-10))
            @test stats["converged"]
            r = compute_residual(ps, s.voltages, s.controls)
            ncb = 2 * ps.n_buses
            kr = sqrt(sum(abs2, @view r[1:2:ncb]))
            ki = sqrt(sum(abs2, @view r[2:2:ncb]))
            lp = sum(s.controls[i] for i in load_P_idx)
            @test kr <= cfg.tau_p_rel * lp
            @test ki <= kr            # reactive infeasibility manifests as voltage, not residual
        end
    end
end

@testset "balanced sampler — synchronous condenser" begin
    ps = _CASE9.power_system
    gen_P_idx = control_indices(ps, SynchronousMachineStatic, :P)
    @test length(gen_P_idx) == 3

    # Mark the second generator as a condenser (no prime mover → active P fixed 0).
    cfg = SamplerConfig(cap_samples = 20, dispatchable = [true, false, true])
    dataset = generate_dataset(ps, cfg, 5; rng = MersenneTwister(0))
    @test length(dataset) == 5

    ctrl = controls_matrix(dataset)
    @test all(iszero, ctrl[:, gen_P_idx[2]])              # condenser carries no active power
    @test all(ctrl[:, gen_P_idx[1]] .> 0)                 # dispatchable fleet carries it
    @test all(ctrl[:, gen_P_idx[3]] .> 0)
end

@testset "calibrate_capacity — finite and deterministic" begin
    ps = _CASE9.power_system
    cap1 = calibrate_capacity(ps, CASE9_OC_SETUP; rng = MersenneTwister(1))
    cap2 = calibrate_capacity(ps, CASE9_OC_SETUP; rng = MersenneTwister(1))
    @test isfinite(cap1) && cap1 > 0
    @test cap1 == cap2
end
