isnothing(_CASE9) && return

# Verification protocol:
# 1. Pull controls from Powsybl's solved state (these give zero residual at Powsybl's voltages).
# 2. Pick a weight vector w over the generator P controls.
# 3. Perturb those controls along -w by a scalar delta; perturbed - target = -delta * w.
# 4. Solve RPF with distributed PtO slack using the same weights w on the same free indices.
#    The joint Newton step moves controls in direction `alpha * w`, which spans exactly the
#    line from the perturbed point back to the target. The unique residual-zero point along
#    that line is the Powsybl solution, so voltages must reproduce Powsybl's to tolerance.

@testset "distributed PtO slack — Powsybl verification" begin

    @testset "case9 base case" begin
        ps       = _CASE9.power_system
        controls_target  = controls_from_solution(_CASE9)

        gen_P_indices = control_indices(ps, SynchronousMachineStatic, :P)
        weights = [0.5, 0.3, 0.2]
        delta   = 0.1

        controls_perturbed = copy(controls_target)
        controls_perturbed[gen_P_indices] .-= delta .* weights

        state = PowerFlowState(ps, controls_perturbed)
        stats = solve_distributed_slack!(state, gen_P_indices;
            weights = weights, tol = 1e-8, max_iterations = 50)

        @test stats["converged"] == true
        @test stats["residual_norm"] ≤ 1e-6

        @test maximum(abs.(state.controls[gen_P_indices] .- controls_target[gen_P_indices])) ≤ 1e-6
    end

end
