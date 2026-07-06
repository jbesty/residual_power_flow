using LinearAlgebra: norm

# ── ExplicitAnalytical with non-unit statuses ─────────────────────────────────
@testset "ExplicitAnalytical status weighting" begin
    # Two-bus network: generator at bus 1, line 1→2.
    builder = PowerSystemBuilder{Float64}(2)
    add_one_bus_injector!(builder, 1, SynchronousMachineStatic(1.0))
    add_two_bus_injector!(builder, 1, 2, Line(0.01, 0.05, 0.02))
    ps = build!(builder)

    v = [1.0, 1.0, 0.0]
    u = [1.0, 1.0]

    # Branch tripped (status = 0)
    statuses_branch = [1.0, 0.0]
    J_fd_v  = ResidualPowerFlow.compute_jacobian_voltages(ps, v, u, CentralDifferenceApproximation();         statuses=statuses_branch)
    J_exp_v = ResidualPowerFlow.compute_jacobian_voltages(ps, v, u, ExplicitAnalytical(); statuses=statuses_branch)
    @test norm(Matrix(J_exp_v) - J_fd_v) / max(norm(J_fd_v), 1e-10) < 1e-4

    J_fd_u  = ResidualPowerFlow.compute_jacobian_controls(ps, v, u, CentralDifferenceApproximation();         statuses=statuses_branch)
    J_exp_u = ResidualPowerFlow.compute_jacobian_controls(ps, v, u, ExplicitAnalytical(); statuses=statuses_branch)
    @test norm(Matrix(J_exp_u) - J_fd_u) / max(norm(J_fd_u), 1e-10) < 1e-4

    # Generator out of service (status = 0)
    statuses_gen = [0.0, 1.0]
    J_fd_v2  = ResidualPowerFlow.compute_jacobian_voltages(ps, v, u, CentralDifferenceApproximation();         statuses=statuses_gen)
    J_exp_v2 = ResidualPowerFlow.compute_jacobian_voltages(ps, v, u, ExplicitAnalytical(); statuses=statuses_gen)
    @test norm(Matrix(J_exp_v2) - J_fd_v2) / max(norm(J_fd_v2), 1e-10) < 1e-4

    J_fd_u2  = ResidualPowerFlow.compute_jacobian_controls(ps, v, u, CentralDifferenceApproximation();         statuses=statuses_gen)
    J_exp_u2 = ResidualPowerFlow.compute_jacobian_controls(ps, v, u, ExplicitAnalytical(); statuses=statuses_gen)
    @test norm(Matrix(J_exp_u2) - J_fd_u2) / max(norm(J_fd_u2), 1e-10) < 1e-4
end

isnothing(_CASE9) && return

@testset "Jacobians" begin

    # Use case9 at its known solution as the operating point. The Jacobians at a
    # converged solution are well-conditioned and give a meaningful finite-difference
    # cross-check.
    sys      = _CASE9
    ps       = sys.power_system
    controls = controls_from_solution(sys)
    variables = target_variables(sys)

    n_buses    = ps.n_buses
    n_lines    = ps.n_lines
    n_controls = ps.n_controls
    n_residual = length(ResidualPowerFlow.compute_residual(ps, variables, controls))

    @testset "Shape and finiteness" begin
        J_v = ResidualPowerFlow.compute_jacobian_voltages(ps, variables, controls)
        @test size(J_v) == (n_residual, n_buses + n_lines)
        @test all(isfinite, J_v)

        J_u = ResidualPowerFlow.compute_jacobian_controls(ps, variables, controls)
        @test size(J_u) == (n_residual, n_controls)
        @test all(isfinite, J_u)
    end

    @testset "ForwardDiff vs finite differences (controls)" begin
        J_ad = ResidualPowerFlow.compute_jacobian_controls(ps, variables, controls)
        J_fd = ResidualPowerFlow.compute_jacobian_controls(ps, variables, controls, ResidualPowerFlow.CentralDifferenceApproximation())
        @test isapprox(J_ad, J_fd; atol = 1e-4)
    end

    @testset "ForwardDiff vs finite differences (voltages)" begin
        epsilon = 1e-5
        J_ad = ResidualPowerFlow.compute_jacobian_voltages(ps, variables, controls)
        n_vars = n_buses + n_lines
        J_fd = zeros(n_residual, n_vars)
        for j in 1:n_vars
            vp = copy(variables); vp[j] += epsilon
            vm = copy(variables); vm[j] -= epsilon
            J_fd[:, j] = (ResidualPowerFlow.compute_residual(ps, vp, controls) .-
                          ResidualPowerFlow.compute_residual(ps, vm, controls)) ./ (2 * epsilon)
        end
        @test isapprox(J_ad, J_fd; atol = 1e-4)
    end

end

# ── Full cross-method Jacobian verification (slow only) ──────────────────────

if SLOW_TESTS
@testset "Jacobian method cross-verification" begin
    sys  = _CASE9
    ps   = sys.power_system
    u0   = controls_from_solution(sys)
    v_conv = target_variables(sys)
    v_flat = ResidualPowerFlow.get_flat_start(ps)

    methods_under_test = [ExplicitAnalytical()]

    for (point_name, v) in [("flat start", v_flat), ("converged", v_conv)]
        J_ref_v = ResidualPowerFlow.compute_jacobian_voltages(ps, v, u0, CentralDifferenceApproximation())
        J_ref_u = ResidualPowerFlow.compute_jacobian_controls(ps, v, u0, CentralDifferenceApproximation())
        norm_ref_v = norm(J_ref_v)
        norm_ref_u = norm(J_ref_u)

        for method in methods_under_test
            method_name = string(typeof(method))
            @testset "$method_name vs CentralDifferenceApproximation ($point_name) — voltage Jacobian" begin
                J_v = Matrix(ResidualPowerFlow.compute_jacobian_voltages(ps, v, u0, method))
                @test size(J_v) == size(J_ref_v)
                @test all(isfinite, J_v)
                @test norm(J_v - J_ref_v) / norm_ref_v < 1e-4
            end

            @testset "$method_name vs CentralDifferenceApproximation ($point_name) — control Jacobian" begin
                J_u = Matrix(ResidualPowerFlow.compute_jacobian_controls(ps, v, u0, method))
                @test size(J_u) == size(J_ref_u)
                @test all(isfinite, J_u)
                @test norm(J_u - J_ref_u) / norm_ref_u < 1e-4
            end
        end

    end

    @testset "solve() with ExplicitAnalytical" begin
        state = PowerFlowState(ps, u0)
        opts  = SolverOptions(jacobian_method = ExplicitAnalytical())
        solved, _ = solve(state; solver_options = opts)
        @test solved isa PowerFlowState
    end

    if _powsybl_available
    # Also verify on case14 (more lines, more interesting sparsity)
    sys14 = build_powersystem(Powsybl.Network.create_ieee14(); K_DV = 100.0)
    ps14  = sys14.power_system
    u14   = controls_from_solution(sys14)
    v14   = target_variables(sys14)

    J_ref_v14 = ResidualPowerFlow.compute_jacobian_voltages(ps14, v14, u14, CentralDifferenceApproximation())
    J_ref_u14 = ResidualPowerFlow.compute_jacobian_controls(ps14, v14, u14, CentralDifferenceApproximation())

    for method in methods_under_test
        method_name = string(typeof(method))
        @testset "$method_name vs CentralDifferenceApproximation (case14 converged) — voltage Jacobian" begin
            J_v = Matrix(ResidualPowerFlow.compute_jacobian_voltages(ps14, v14, u14, method))
            @test norm(J_v - J_ref_v14) / norm(J_ref_v14) < 1e-4
        end
        @testset "$method_name vs CentralDifferenceApproximation (case14 converged) — control Jacobian" begin
            J_u = Matrix(ResidualPowerFlow.compute_jacobian_controls(ps14, v14, u14, method))
            @test norm(J_u - J_ref_u14) / norm(J_ref_u14) < 1e-4
        end
    end
    end # _powsybl_available
end
end # SLOW_TESTS
