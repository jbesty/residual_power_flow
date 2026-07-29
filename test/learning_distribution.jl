isnothing(_CASE9) && return

@testset "subsample" begin
    ps      = _CASE9.power_system
    dataset = generate_dataset(ps, CASE9_OC_SETUP, 8; rng = MersenneTwister(42))

    @testset "happy path — requested sizes" begin
        for k in [1, 3, length(dataset)]
            sub = ResidualPowerFlow.subsample(dataset, k; rng = MersenneTwister(1))
            @test sub isa PowerFlowDataset
            @test length(sub) == k
            @test size(controls_matrix(sub), 1) == k
            @test size(voltages_matrix(sub), 1) == k
        end
    end

    @testset "deterministic RNG" begin
        sub1 = ResidualPowerFlow.subsample(dataset, 3; rng = MersenneTwister(1234))
        sub2 = ResidualPowerFlow.subsample(dataset, 3; rng = MersenneTwister(1234))
        @test controls_matrix(sub1) == controls_matrix(sub2)
        @test voltages_matrix(sub1) == voltages_matrix(sub2)
    end

    @testset "ArgumentError when n > length(dataset)" begin
        @test_throws ArgumentError ResidualPowerFlow.subsample(dataset, length(dataset) + 1)
    end
end

@testset "dataset_distribution" begin
    ps      = _CASE9.power_system
    dataset = generate_dataset(ps, CASE9_OC_SETUP, 30; rng = MersenneTwister(7))
    dd = ResidualPowerFlow.dataset_distribution(dataset, ps; n_hessian_samples = 3, rng = MersenneTwister(11))

    @test dd isa ResidualPowerFlow.DatasetDistribution

    @testset "all ten fields finite" begin
        @test all(isfinite, dd.ū)
        @test all(isfinite, dd.W_controls_point)
        @test all(isfinite, dd.W_controls_mean)
        @test all(isfinite, dd.covariance_point)
        @test all(isfinite, dd.covariance_mean)
        @test isfinite(dd.effective_variance_point)
        @test isfinite(dd.effective_variance_mean)
        @test isfinite(dd.log_volume_point)
        @test isfinite(dd.log_volume_mean)
        @test isfinite(dd.jensen_gap)
    end

    @test dd.effective_variance_mean > 0

    # n_hessian_samples > 1 averaging path exercised (n=3 above; verify mean differs from point)
    @test dd.W_controls_mean != dd.W_controls_point

    # Jensen direction (log_volume_mean vs log_volume_point) requires a convexity
    # argument that is not yet settled — no one-sided assertion added.
end

@testset "_compute_hessian_blocks: analytical path ≈ DenseAD path" begin
    ps      = _CASE9.power_system
    dataset = generate_dataset(ps, CASE9_OC_SETUP, 30; rng = MersenneTwister(7))
    n_samples = 5
    seed      = 11

    # Analytical path (ExplicitAnalytical is now the per-point default in _compute_hessian_blocks)
    H_vv_an, W_an = ResidualPowerFlow._compute_hessian_blocks(
        dataset, ps, n_samples, MersenneTwister(seed))

    # DenseAD reference over identical sample indices (same rng → same shuffle)
    N   = length(dataset)
    n_u = ps.n_controls
    n_v = ps.n_variables
    idx = ResidualPowerFlow.shuffle(MersenneTwister(seed), 1:N)[1:min(n_samples, N)]

    H_vv_sum = zeros(n_v, n_v)
    W_sum    = zeros(n_u, n_u)
    for ii in idx
        u = dataset[ii].controls
        v = dataset[ii].voltages
        H    = compute_energy_hessian(ps, v, u, DenseAD())
        H_vv = H[1:n_v, 1:n_v]
        H_uv = H[n_v+1:end, 1:n_v]
        H_vv = 0.5 * (H_vv + H_vv')
        H_vv_sum += H_vv
        W_sum    += H_uv * (H_vv \ H_uv')
    end
    n        = length(idx)
    H_vv_ref = H_vv_sum / n
    W_ref    = 0.5 * (W_sum / n + (W_sum / n)')

    @test isapprox(H_vv_an, H_vv_ref; rtol = 1e-6)
    @test isapprox(W_an,    W_ref;    rtol = 1e-6)
end
