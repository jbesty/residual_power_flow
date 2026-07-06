@testset "PowSyBl IO" begin

    _iidm_dir = joinpath(Base.pkgdir(ResidualPowerFlow), "test", "data", "iidm")

    # K_DV values matched to the RPF generator parameterisation for each built-in case.
    _K_DV = Dict(:case9 => [130.0, 21.0, 13.0], :case14 => fill(100.0, 5))

    # ── Shared case9 fixture ──────────────────────────────────────────────────
    #
    # Run Powsybl AC LF once; derive controls from the solved state via
    # controls_from_solution.  Unbalanced controls scale each load's active
    # power by 1.1, creating a ~31 MW generation deficit.

    _lf_params = Powsybl.LoadFlow.load_flow_parameters()

    _net9 = Powsybl.Network.create_ieee9()
    Powsybl.LoadFlow.run_ac(_net9, _lf_params)
    _sys9 = build_powersystem(_net9; K_DV = _K_DV[:case9])
    _ctrl_balanced = controls_from_solution(_sys9)

    _net14 = Powsybl.Network.create_ieee14()
    Powsybl.LoadFlow.run_ac(_net14, _lf_params)
    _sys14 = build_powersystem(_net14; K_DV = _K_DV[:case14])
    _ctrl_unbalanced = let c = copy(_ctrl_balanced)
        n_gen = size(_sys9.gens_df, 1)
        n_load = size(_sys9.loads_df, 1)
        for k = 1:n_load
            c[2*n_gen+2*(k-1)+1] *= 1.1   # scale P0 only, leave Q0 unchanged
        end
        c
    end

    # ── Structure ─────────────────────────────────────────────────────────────

    @testset "Structure (case9)" begin
        ps = _sys9.power_system
        @test ps isa PowerSystem{Float64}
        @test ps.n_buses == 9
        @test ps.n_lines == 9       # 6 lines + 3 transformers
        @test ps.n_controls == 12      # 3 generators × 2 + 3 loads × 2
    end

    @testset "Structure (case14)" begin
        ps = _sys14.power_system
        @test ps isa PowerSystem{Float64}
        @test ps.n_buses == 14
        @test ps.n_lines == 20
        @test ps.n_controls > 0
    end

    # ── K_DV propagation ─────────────────────────────────────────────────────

    @testset "K_DV propagation (case9)" begin
        ps = _sys9.power_system
        @test ps.single_bus_injectors[1].component.K_DV == 130.0
        @test ps.single_bus_injectors[2].component.K_DV == 21.0
        @test ps.single_bus_injectors[3].component.K_DV == 13.0
    end

    # ── Solver modes ──────────────────────────────────────────────────────────

    @testset "FixedControls — balanced solution converges (case9)" begin
        state = PowerFlowState(_sys9.power_system, _ctrl_balanced)
        stats = solve!(state, GaussNewtonSolver();
            solver_options = SolverOptions(max_iterations = 50, tolerance = 1e-8))
        @test stats["converged"] == true
        @test stats["residual_norm"] ≤ 1e-6
    end

    @testset "FixedControls — nonzero residual under power imbalance (case9)" begin
        state = PowerFlowState(_sys9.power_system, _ctrl_unbalanced)
        stats = solve!(state, GaussNewtonSolver();
            solver_options = SolverOptions(max_iterations = 50, tolerance = 1e-8))
        @test stats["converged"] == true   # step criterion met
        @test stats["residual_norm"] > 1e-2   # but residual stays non-zero
    end

    @testset "SingleSlack — zero residual under same imbalance (case9)" begin
        state = PowerFlowState(_sys9.power_system, _ctrl_unbalanced)
        stats = solve!(state, GaussNewtonSolver(), PtOConfig(free_indices = [1]);
            solver_options = SolverOptions(max_iterations = 50, tolerance = 1e-8))
        @test stats["converged"] == true
        @test stats["residual_norm"] ≤ 1e-6
    end

    @testset "SingleSlack convergence (case14)" begin
        controls = controls_from_solution(_sys14)
        state = PowerFlowState(_sys14.power_system, controls)
        stats = solve!(state, GaussNewtonSolver(), PtOConfig(free_indices = [1]);
            solver_options = SolverOptions(max_iterations = 50, tolerance = 1e-8))
        @test stats["converged"] == true
        @test stats["residual_norm"] ≤ 1e-6
        n = _sys14.power_system.n_buses
        @test maximum(abs.(state.voltages[1:n] .- target_variables(_sys14)[1:n])) ≤ 1e-3
    end

    if SLOW_TESTS
    @testset "SingleSlack convergence (case57)" begin
        net = Powsybl.Network.create_ieee57()
        Powsybl.LoadFlow.run_ac(net, _lf_params)
        sys = build_powersystem(net; K_DV = 100.0)
        controls = controls_from_solution(sys)
        state = PowerFlowState(sys.power_system, controls)
        stats = solve!(state, GaussNewtonSolver(), PtOConfig(free_indices = [1]);
            solver_options = SolverOptions(max_iterations = 50, tolerance = 1e-8))
        @test stats["converged"] == true
        @test stats["residual_norm"] ≤ 1e-6
        n = sys.power_system.n_buses
        @test maximum(abs.(state.voltages[1:n] .- target_variables(sys)[1:n])) ≤ 1e-3
    end

    @testset "SingleSlack convergence (case118)" begin
        net = Powsybl.Network.create_ieee118()
        Powsybl.LoadFlow.run_ac(net, _lf_params)
        sys = build_powersystem(net; K_DV = 100.0)
        controls = controls_from_solution(sys)
        state = PowerFlowState(sys.power_system, controls)
        stats = solve!(state, GaussNewtonSolver(), PtOConfig(free_indices = [69]);
            solver_options = SolverOptions(max_iterations = 50, tolerance = 1e-8))
        @test stats["converged"] == true
        @test stats["residual_norm"] ≤ 1e-6
        n = sys.power_system.n_buses
        @test maximum(abs.(state.voltages[1:n] .- target_variables(sys)[1:n])) ≤ 1e-3
    end
    end # SLOW_TESTS

    @testset "DistributedSlack — zero generator P throws ArgumentError" begin
        ps = _sys9.power_system
        gen_P_indices = control_indices(ps, SynchronousMachineStatic, :P)

        # Zero all generator active-power setpoints; indices 1, 3, 5 for case9.
        ctrl_zero_p = copy(_ctrl_balanced)
        for i in 1:3
            ctrl_zero_p[2*i - 1] = 0.0
        end

        state = PowerFlowState(ps, ctrl_zero_p)
        @test_throws ArgumentError pto_step!(state, GaussNewtonSolver(), PtOConfig(free_indices = gen_P_indices))
    end

    # ── IIDM per-unit conversion ──────────────────────────────────────────────
    #
    # Hand-crafted 3-bus network with known SI parameters; verifies the Z_base
    # formula in PowsyblIOExt independently of any load flow result.
    # nominalV=100 kV, baseMVA=100 → Z_base=100 Ω
    # Line 1-2: R=1 Ω, X=5 Ω, b1=b2=0.0005 S → R_pu=0.01, X_pu=0.05, B_pu=0.1

    @testset "IIDM — per-unit conversion (case3)" begin
        sys = build_powersystem(joinpath(_iidm_dir, "case3.xiidm"))
        ps = sys.power_system

        @test ps.n_buses == 3
        @test ps.n_lines == 2

        br12 = ps.branch_injectors[1].component
        @test isapprox(br12.R, 0.01; atol = 1e-10)
        @test isapprox(br12.X, 0.05; atol = 1e-10)
        @test isapprox(br12.B, 0.1; atol = 1e-10)
        @test isapprox(br12.TAP, 1.0; atol = 1e-10)
    end

    # ── Powsybl LF cross-validation ───────────────────────────────────────────

    @testset "Powsybl LF cross-validation (case9)" begin
        voltages = target_variables(_sys9)

        residual = compute_current_balance(_sys9.power_system, voltages, _ctrl_balanced)
        @test maximum(abs.(residual)) ≤ 1e-6

        state = PowerFlowState(_sys9.power_system, _ctrl_balanced)
        stats = solve!(state, GaussNewtonSolver(), PtOConfig(free_indices = [1]);
            solver_options = SolverOptions(max_iterations = 50, tolerance = 1e-8))
        @test stats["converged"] == true
        @test stats["residual_norm"] ≤ 1e-6
        n = _sys9.power_system.n_buses
        @test maximum(abs.(state.voltages[1:n] .- voltages[1:n])) ≤ 1e-3
    end

    @testset "Powsybl LF cross-validation (case14)" begin
        voltages = target_variables(_sys14)
        controls = controls_from_solution(_sys14)
        residual = compute_current_balance(_sys14.power_system, voltages, controls)
        # Threshold is 1e-4 (not 1e-6) because Powsybl's own LF terminal flows
        # at the slack bus have ~7 kW (6e-5 pu) residual from NR convergence.
        @test maximum(abs.(residual)) ≤ 1e-4
    end

    if SLOW_TESTS
    @testset "Powsybl LF cross-validation (case57)" begin
        net = Powsybl.Network.create_ieee57()
        Powsybl.LoadFlow.run_ac(net, _lf_params)
        sys = build_powersystem(net; K_DV = 100.0)
        voltages = target_variables(sys)
        controls = controls_from_solution(sys)
        residual = compute_current_balance(sys.power_system, voltages, controls)
        n = sys.power_system.n_buses
        bus_res = [maximum(abs.([residual[2i-1], residual[2i]])) for i in 1:n]
        # Bus 1 is the distributed-slack reference bus. Powsybl's own terminal flows
        # carry ~0.3 MW of nodal imbalance there — an artifact of the distributed-slack
        # algorithm, not an RPF modelling error. All other buses must satisfy 1e-4.
        @test maximum(bus_res[setdiff(1:n, [1])]) ≤ 1e-4
        @test bus_res[1] ≤ 5e-3
    end

    @testset "Powsybl LF cross-validation (case118)" begin
        net = Powsybl.Network.create_ieee118()
        Powsybl.LoadFlow.run_ac(net, _lf_params)
        sys = build_powersystem(net; K_DV = 100.0)
        voltages = target_variables(sys)
        controls = controls_from_solution(sys)
        residual = compute_current_balance(sys.power_system, voltages, controls)
        n = sys.power_system.n_buses
        bus_res = [maximum(abs.([residual[2i-1], residual[2i]])) for i in 1:n]
        # Bus 69 is the distributed-slack reference bus. Powsybl's own terminal flows
        # carry ~0.25 MW of nodal imbalance there — an artifact of the distributed-slack
        # algorithm, not an RPF modelling error. All other buses must satisfy 1e-4.
        @test maximum(bus_res[setdiff(1:n, [69])]) ≤ 1e-4
        @test bus_res[69] ≤ 5e-3
    end

    @testset "MATPOWER .mat cross-validation — phase shift + shunt Gs (case89pegase)" begin
        # case89pegase carries 3 phase-shifting transformers and 26 buses with
        # shunt conductance Gs.  Both were silently dropped at import before the
        # PowsyblIOExt fix (θ_shift from :alpha, G_pu from g_per_section), leaving
        # the MATPOWER-converged point with a ~0.5 KCL residual under RPF physics.
        # Guards against regression of either, end-to-end through the .mat reader.
        mat = joinpath(Base.pkgdir(ResidualPowerFlow), "data", "matpower", "case89pegase.mat")
        sys = build_powersystem(mat)
        voltages = target_variables(sys)
        controls = controls_from_solution(sys)
        residual = compute_current_balance(sys.power_system, voltages, controls)
        @test maximum(abs.(residual)) ≤ 1e-6
    end
    end # SLOW_TESTS

end
