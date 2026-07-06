@testset "compute_energy_hessian" begin
    ps = _CASE9.power_system
    v  = ResidualPowerFlow.get_flat_start(ps)
    u  = ResidualPowerFlow.controls_from_solution(_CASE9)

    H_dense = compute_energy_hessian(ps, v, u, DenseAD())

    @testset "symmetry — DenseAD" begin
        @test H_dense ≈ H_dense'
    end

    @testset "ExplicitAnalytical oracle vs DenseAD" begin
        H_explicit = compute_energy_hessian(ps, v, u, ExplicitAnalytical())
        @test H_explicit ≈ H_explicit'
        @test isapprox(H_explicit, H_dense; rtol=1e-6)
    end

    @testset "SemiAnalytical oracle vs DenseAD" begin
        H_semi = compute_energy_hessian(ps, v, u, SemiAnalytical())
        @test H_semi ≈ H_semi'
        @test isapprox(H_semi, H_dense; rtol=1e-6)
    end

    @testset "GaussNewtonApproximation matches J^T J" begin
        H_gn = compute_energy_hessian(ps, v, u, GaussNewtonApproximation())
        @test H_gn ≈ H_gn'
        J_v = Matrix(ResidualPowerFlow.compute_jacobian_voltages(ps, v, u, ExplicitAnalytical()))
        J_u = Matrix(ResidualPowerFlow.compute_jacobian_controls(ps, v, u, ExplicitAnalytical()))
        JtJ = [J_v'*J_v  J_v'*J_u; J_u'*J_v  J_u'*J_u]
        @test H_gn ≈ JtJ
    end

    @testset "default method (no method arg) is ExplicitAnalytical" begin
        H_default  = compute_energy_hessian(ps, v, u)
        H_explicit = compute_energy_hessian(ps, v, u, ExplicitAnalytical())
        @test H_default == H_explicit
        @test isapprox(H_default, H_dense; rtol=1e-6)
    end

    @testset "Float32 type stability — ExplicitAnalytical" begin
        H32 = compute_energy_hessian(ps, Float32.(v), Float32.(u), ExplicitAnalytical())
        @test eltype(H32) == Float32
    end

    @testset "statuses kwarg — ExplicitAnalytical matches DenseAD on tripped branch" begin
        n_single  = length(ps.single_bus_injectors)
        n_branches = length(ps.branch_injectors)
        statuses_tripped = ones(n_single + n_branches)
        statuses_tripped[n_single + 1] = 0.0

        H_ea_none    = compute_energy_hessian(ps, v, u, ExplicitAnalytical())
        H_ea_tripped = compute_energy_hessian(ps, v, u, ExplicitAnalytical(); statuses = statuses_tripped)
        H_ad_tripped = compute_energy_hessian(ps, v, u, DenseAD(); statuses = statuses_tripped)

        @test !isapprox(H_ea_none, H_ea_tripped; rtol = 1e-3)
        @test isapprox(H_ea_tripped, H_ad_tripped; rtol = 1e-8)
    end
end
