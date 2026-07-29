isnothing(_CASE9) && return

# Oracle protocol (option 4 of the spec follow-up discussion):
#
# An external Powsybl oracle cannot be used because Powsybl.jl v0.3 exposes no
# network-mutation or branch-disconnection API. Instead, a fresh RPF PowerSystem
# is rebuilt with the chosen branch omitted entirely from the constructor inputs.
# That fresh PowerSystem has, by construction, the correct cycle basis and
# injection set for the tripped topology, and is used as the oracle for:
#
#   1. The trip path: mutate `statuses` on the original PowerSystem, call
#      `update_topology!`, `solve!`, and compare bus voltage magnitudes and
#      bus angles against the freshly-built reduced-topology solve.
#
#   2. The restore round-trip: restore `statuses`, call `update_topology!`,
#      reset voltages to a flat start, and re-solve. Compare bit-for-bit against
#      the pre-trip solve of the original PowerSystem — any cache leak from the
#      topology mutation path would surface as a bit-level drift.

function _build_power_system_without_line(
    ps::ResidualPowerFlow.PowerSystem{T}, removed_line_index::Int,
) where {T}
    builder = ResidualPowerFlow.PowerSystemBuilder{T}(ps.n_buses)
    for inj in ps.single_bus_injectors
        ResidualPowerFlow.add_one_bus_injector!(builder, inj.bus_id, inj.component)
    end
    for (k, br) in enumerate(ps.branch_injectors)
        k == removed_line_index && continue
        ResidualPowerFlow.add_two_bus_injector!(builder, br.from_bus, br.to_bus, br.component)
    end
    return ResidualPowerFlow.build!(builder)
end

@testset "topology switching — self-consistency oracle" begin

    @testset "controls_from_voltages recovers generator controls" begin
        ps = _CASE9.power_system
        controls = controls_from_solution(_CASE9)

        # Drive to a fully zero-residual point with PtO single slack so that
        # (state.voltages, state.controls) is an exact fixed point of the RPF
        # residual. Then controls_from_voltages must return controls equivalent
        # on the generator slots, and the residual must remain zero everywhere.
        state = PowerFlowState(ps, controls)
        solve_distributed_slack!(state, [1]; tol = 1e-10, max_iterations = 50)

        gen_P  = control_indices(ps, SynchronousMachineStatic, :P)
        gen_Vr = control_indices(ps, SynchronousMachineStatic, :Vref)
        gen_indices  = sort!(vcat(gen_P, gen_Vr))
        load_indices = setdiff(1:length(controls), gen_indices)

        # Identity at an exact fixed point: recovered generator controls match
        # the converged controls; loads untouched; residual still zero.
        recovered = controls_from_voltages(ps, state.voltages, state.controls)
        @test recovered[gen_indices]  ≈ state.controls[gen_indices] rtol = 1e-10
        @test recovered[load_indices] == state.controls[load_indices]
        @test maximum(abs.(compute_current_balance(ps, state.voltages, recovered))) ≤ 1e-8

        # Inversion from perturbed generator controls: generator slots are
        # overwritten to recover the zero-residual controls regardless of input.
        perturbed = copy(state.controls)
        perturbed[gen_indices] .+= 0.3
        recovered_pert = controls_from_voltages(ps, state.voltages, perturbed)
        @test recovered_pert[gen_indices]  ≈ state.controls[gen_indices] rtol = 1e-10
        @test recovered_pert[load_indices] == perturbed[load_indices]
        @test maximum(abs.(compute_current_balance(ps, state.voltages, recovered_pert))) ≤ 1e-8
    end

    @testset "case9 single branch trip/restore" begin
        ps = _CASE9.power_system
        n_single = length(ps.single_bus_injectors)

        # case9 has 9 branches and 9 buses → cycle rank 1 (a single fundamental
        # cycle). Line 1 (the first branch in the RPF edge list) participates
        # in that cycle and leaves the grid connected when tripped.
        tripped_line = 1
        status_idx   = n_single + tripped_line

        controls = controls_from_solution(_CASE9)
        opts     = SolverOptions(max_iterations = 50, tolerance = 1e-8)

        # --- Base solve (full topology) ---
        state_pre = PowerFlowState(ps, controls)
        solve!(state_pre, GaussNewtonSolver(); solver_options = opts)

        pre_voltages = copy(state_pre.voltages)
        pre_controls = copy(state_pre.controls)
        pre_sbc      = copy(state_pre.single_bus_currents)
        pre_bc       = copy(state_pre.branch_currents)

        # --- Trip via status mutation + update_topology! ---
        state = PowerFlowState(ps, controls)
        state.statuses[status_idx] = 0.0
        update_topology!(state)
        solve!(state, GaussNewtonSolver(); solver_options = opts)

        # --- Oracle: fresh reduced PowerSystem without that branch ---
        ps_reduced    = _build_power_system_without_line(ps, tripped_line)
        state_reduced = PowerFlowState(ps_reduced, controls)
        solve!(state_reduced, GaussNewtonSolver(); solver_options = opts)

        @test maximum(abs.(voltage_magnitudes(state) .- voltage_magnitudes(state_reduced))) ≤ 1e-6

        # Active branch angles are directly comparable: _build_power_system_without_line
        # preserves the branch ordering of ps with the tripped branch removed, and
        # update_topology! on the mutated state uses the same cycle basis as ps_reduced.
        active_line_indices = setdiff(1:ps.n_lines, [tripped_line])
        trip_active_angles  = state.voltages[ps.n_buses .+ active_line_indices]
        @test maximum(abs.(trip_active_angles .- branch_angles(state_reduced))) ≤ 1e-6

        # Tripped branch angle must be pinned to zero in the mutated-state path.
        @test state.voltages[ps.n_buses + tripped_line] == 0.0

        # --- Restore via status mutation + update_topology! ---
        # Reset voltages to flat start so the restore solve sees the same
        # initial conditions as the pre-trip solve. Bit-for-bit equality is a
        # determinism + cache-invalidation check: if update_topology! fully
        # restores the cycle basis and clears the Jacobian caches, identical
        # inputs must produce identical bits.
        state.statuses[status_idx] = 1.0
        update_topology!(state)
        state.voltages .= get_flat_start(ps)
        solve!(state, GaussNewtonSolver(); solver_options = opts)

        @test state.voltages            == pre_voltages
        @test state.controls            == pre_controls
        @test state.single_bus_currents == pre_sbc
        @test state.branch_currents     == pre_bc
    end

end
