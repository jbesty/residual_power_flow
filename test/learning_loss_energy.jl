# Loss-energy identity: the unweighted squared error in the non-dimensional
# voltage frame equals the H_vv-weighted squared error in the physical frame
# when A_voltages = H_vv^{-1/2} (the :physics_informed scheme). Pins
# the loss-energy identity underlying the non-dimensionalisation paper.
# Construction is synthetic (no power system); the
# identity is purely algebraic for symmetric SPD A_voltages.

import LinearAlgebra: I, eigen, Symmetric, Diagonal

@testset "loss-energy identity (synthetic SPD)" begin
    rng   = MersenneTwister(1)
    n_v   = 7
    n_u   = 3
    batch = 5

    M    = randn(rng, n_v, n_v)
    H_vv = M * M' + n_v * I
    H_vv = 0.5 * (H_vv + H_vv')

    F          = eigen(Symmetric(H_vv))
    A_voltages = Matrix(F.vectors * Diagonal(1 ./ sqrt.(F.values)) * F.vectors')

    A_controls = Matrix{Float64}(I, n_u, n_u)
    t = DataTransformation(A_controls, zeros(n_u), A_voltages, zeros(n_v))

    voltage_pred = randn(rng, batch, n_v)
    voltage_true = randn(rng, batch, n_v)

    delta_nondim = normalize_voltages(t, voltage_pred) .-
                   normalize_voltages(t, voltage_true)
    delta_phys   = voltage_pred .- voltage_true

    for j in 1:batch
        lhs = sum(abs2, view(delta_nondim, j, :))
        rhs = view(delta_phys, j, :)' * H_vv * view(delta_phys, j, :)
        @test isapprox(lhs, rhs; rtol = 1e-12)
    end
end
