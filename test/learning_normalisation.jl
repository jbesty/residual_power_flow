isnothing(_CASE9) && return

@testset "DataTransformation" begin
    rng          = MersenneTwister(99)
    power_system = _CASE9.power_system

    dataset = generate_dataset(power_system, CASE9_OC_SETUP, 50; rng = rng)

    controls = controls_matrix(dataset)
    voltages = voltages_matrix(dataset)

    for scheme in (:identity, :standardised)
        @testset "$scheme" begin
            t = fit_data_transformation(dataset, power_system;
                                        controls_scheme = scheme, voltages_scheme = scheme)

            u = controls[1, :]
            v = voltages[1, :]
            @test isapprox(denormalize_controls(t, normalize_controls(t, u)), u; atol = 1e-12)
            @test isapprox(denormalize_voltages(t, normalize_voltages(t, v)), v; atol = 1e-12)

            @test isapprox(denormalize_controls(t, normalize_controls(t, controls)), controls; atol = 1e-12)
            @test isapprox(denormalize_voltages(t, normalize_voltages(t, voltages)), voltages; atol = 1e-12)
        end
    end

    @testset "PowerFlowDataset input" begin
        t = fit_data_transformation(dataset, power_system;
                                    controls_scheme = :standardised,
                                    voltages_scheme = :standardised)
        @test t isa DataTransformation
    end

    @testset "physics_informed" begin
        ctrl = controls_from_solution(_CASE9)
        vars = target_variables(_CASE9)

        single_state = PowerFlowState(power_system, copy(vars), copy(ctrl))
        ps_dataset = PowerFlowDataset(power_system, [single_state])

        t = fit_data_transformation(ps_dataset, power_system;
                                    controls_scheme   = :physics_informed,
                                    voltages_scheme   = :physics_informed,
                                    n_hessian_samples = 1)

        @test isapprox(denormalize_controls(t, normalize_controls(t, ctrl)), ctrl; atol = 1e-10)
        @test isapprox(denormalize_voltages(t, normalize_voltages(t, vars)), vars; atol = 1e-10)
    end

    @testset "Factorisation consistency ($scheme)" for scheme in (:identity, :standardised, :physics_informed)
        # Physics-informed needs a single-sample dataset to avoid averaging zero
        # Hessians on AC-feasible training points; reuse the :physics_informed
        # fixture shape from above.
        t = if scheme == :physics_informed
            ctrl = controls_from_solution(_CASE9)
            vars = target_variables(_CASE9)
            ps_dataset = PowerFlowDataset(
                power_system, [PowerFlowState(power_system, copy(vars), copy(ctrl))],
            )
            fit_data_transformation(ps_dataset, power_system;
                controls_scheme = scheme, voltages_scheme = scheme,
                n_hessian_samples = 1)
        else
            fit_data_transformation(dataset, power_system;
                controls_scheme = scheme, voltages_scheme = scheme)
        end

        # Cholesky factorisation reconstructs the stored matrix.
        @test isapprox(Matrix(t.A_controls_fact), t.A_controls; rtol = 1e-10)
        @test isapprox(Matrix(t.A_voltages_fact), t.A_voltages; rtol = 1e-10)

        # Factorisation-based solve matches a plain backslash on A.
        u = controls[1, :]
        v = voltages[1, :]
        @test isapprox(t.A_controls_fact \ (u .- t.b_controls),
                       t.A_controls       \ (u .- t.b_controls); rtol = 1e-10)
        @test isapprox(t.A_voltages_fact \ (v .- t.b_voltages),
                       t.A_voltages       \ (v .- t.b_voltages); rtol = 1e-10)
    end

    @testset "physics_informed: A_v' * A_v = inv(H_vv)" begin
        ctrl = controls_from_solution(_CASE9)
        vars = target_variables(_CASE9)

        single_state = PowerFlowState(power_system, copy(vars), copy(ctrl))
        ps_dataset = PowerFlowDataset(power_system, [single_state])

        t = fit_data_transformation(ps_dataset, power_system;
                                    controls_scheme   = :identity,
                                    voltages_scheme   = :physics_informed,
                                    n_hessian_samples = 1)

        # Paper Eq. 24a: A_v^T A_v = W_v^{-1} with W_v = H_vv for physics_informed.
        n_u = power_system.n_controls
        z = vcat(ctrl, vars)
        H = ForwardDiff.hessian(
            z -> compute_energy(power_system, z[n_u+1:end], z[1:n_u]),
            z,
        )
        H_vv = H[n_u+1:end, n_u+1:end]
        H_vv = 0.5 * (H_vv + H_vv')

        @test isapprox(t.A_voltages' * t.A_voltages, inv(H_vv); rtol = 1e-6)
    end
end
