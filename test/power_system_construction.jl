@testset "Power system construction" begin

    @testset "add_one_bus_injector!" begin
        builder = PowerSystemBuilder{Float64}(3)

        # Add a load at bus 2. It has 2 controls (P, Q), so n_controls grows to 2.
        zip = ConstantPowerLoad(1.0)
        add_one_bus_injector!(builder, 2, zip)
        @test builder.n_controls == 2
        @test length(builder.single_bus_injectors) == 1
        @test builder.single_bus_injectors[1].bus_id == 2
        @test builder.single_bus_injectors[1].component === zip

        # Add a shunt at bus 1. Shunts have no controls, so n_controls stays at 2.
        sh = Shunt(0.1, -0.2)
        add_one_bus_injector!(builder, 1, sh)
        @test builder.n_controls == 2
        @test length(builder.single_bus_injectors) == 2
        @test builder.single_bus_injectors[2].bus_id == 1
    end

    @testset "check_bus_reference rejects out-of-range bus" begin
        builder = PowerSystemBuilder{Float64}(3)
        @test_throws ErrorException check_bus_reference(0, builder)
        @test_throws ErrorException check_bus_reference(4, builder)
    end

    @testset "add_two_bus_injector! rejects same-bus terminals" begin
        builder = PowerSystemBuilder{Float64}(3)
        br = Line(0.01, 0.05, 0.02)
        @test_throws ErrorException add_two_bus_injector!(builder, 2, 2, br)
    end

    @testset "add_two_bus_injector!" begin
        builder = PowerSystemBuilder{Float64}(3)
        br = Line(0.01, 0.05, 0.02)
        add_two_bus_injector!(builder, 1, 2, br)

        @test builder.n_lines == 1
        @test length(builder.branch_injectors) == 1
        @test builder.branch_injectors[1].from_bus == 1
        @test builder.branch_injectors[1].to_bus == 2
        @test builder.branch_injectors[1].component === br
        @test builder.edge_list == [(1, 2)]
    end

    @testset "build!" begin
        builder = PowerSystemBuilder{Float64}(3)
        add_two_bus_injector!(builder, 1, 2, Line(0.01, 0.05, 0.02))
        add_one_bus_injector!(builder, 2, ConstantPowerLoad(1.0))
        add_one_bus_injector!(builder, 1, Shunt(0.1, 0.0))

        ps = build!(builder)
        @test ps isa PowerSystem{Float64}
        @test ps.n_buses == 3
        @test ps.n_lines == 1
        @test ps.n_variables == 4
        @test ps.n_controls == 2
        @test length(ps.single_bus_injectors) == 2
        @test length(ps.branch_injectors) == 1

        # Topology: one branch in a 3-bus tree → no cycles.
        @test ps.n_cycles == 0
    end

    @testset "cycles" begin
        # A triangle (3 nodes, 3 edges) has exactly one independent cycle.
        # The participation vector encodes the signed orientation of each edge in the cycle:
        # each entry is +1, -1, or 0, and all three edges participate.
        edge_list = [(1, 2), (2, 3), (3, 1)]
        cycles, participation = compute_cycles_and_participation(edge_list)
        @test length(cycles) == 1
        @test length(participation) == 1
        @test length(participation[1]) == length(edge_list)
        @test all(abs.(participation[1]) .<= 1)
        @test sum(abs.(participation[1])) == length(edge_list)
    end

    @testset "parallel-edge KVL cycles" begin
        # Two parallel branches between buses 1 and 2: no topological cycle,
        # one trivial KVL cycle enforcing θ_A - θ_B = 0.
        edge_list = [(1, 2), (1, 2)]
        cycles, participation = compute_cycles_and_participation(edge_list)
        @test length(participation) == 1
        trivial = participation[1]
        # Same stored direction → opposite signs.
        @test trivial[1] ==  1
        @test trivial[2] == -1

        # Opposite-direction parallel pair: (1,2) and (2,1)
        edge_list_opp = [(1, 2), (2, 1)]
        _, participation_opp = compute_cycles_and_participation(edge_list_opp)
        @test length(participation_opp) == 1
        trivial_opp = participation_opp[1]
        # Opposite stored direction → same signs
        @test trivial_opp[1] == 1
        @test trivial_opp[2] == 1

        # Triangle with a parallel pair: 1 topological + 1 trivial = 2 cycles.
        edge_list_mix = [(1, 2), (1, 2), (2, 3), (3, 1)]
        _, participation_mix = compute_cycles_and_participation(edge_list_mix)
        @test length(participation_mix) == 2

        # Full PowerSystem build with parallel branches (no triangle)
        builder = PowerSystemBuilder{Float64}(2)
        add_two_bus_injector!(builder, 1, 2, Line(0.01, 0.05, 0.02))
        add_two_bus_injector!(builder, 1, 2, Line(0.02, 0.10, 0.01))  # parallel
        ps = build!(builder)
        @test ps.n_cycles == 1
    end

    @testset "Ybus" begin
        # Single branch: Ybus entries must equal the branch's Y-parameters.
        br = Line(0.01, 0.05, 0.02)
        Ybus = compute_Ybus([br], [1], [2])
        @test size(Ybus) == (2, 2)

        expected = [br.Y_11 br.Y_12; br.Y_21 br.Y_22]
        @test isapprox(Ybus, expected; rtol = 1e-12, atol = 0)

        # Adding a shunt at bus 2 must increment only the [2,2] diagonal element.
        sh = Shunt(0.1, -0.2)
        Ybus_sh = compute_Ybus([br], [1], [2], [sh], [2])
        expected_sh = copy(expected)
        expected_sh[2, 2] += sh.G + im * sh.B
        @test isapprox(Ybus_sh, expected_sh; rtol = 1e-12, atol = 0)
    end

    @testset "angle conversion maps — 3-bus tree" begin
        builder = PowerSystemBuilder{Float64}(3)
        add_two_bus_injector!(builder, 1, 2, Line(0.01, 0.05, 0.02))
        add_two_bus_injector!(builder, 2, 3, Line(0.01, 0.05, 0.02))
        ps = build!(builder)

        # Variables: [V1, V2, V3, θ12, θ23]
        v = [1.0, 1.0, 1.0, 0.1, 0.05]
        bus_angles = convert_branch_angles_to_bus_angles(ps, v; exact=false)
        @test length(bus_angles) == ps.n_buses

        # Round-trip: branch → bus → branch recovers original branch angles
        branch_angles_back = convert_bus_angles_to_branch_angles(ps, bus_angles)
        @test branch_angles_back ≈ [0.1, 0.05] atol=1e-10

        # exact=true must not throw for a tree (A has full row rank → exact round-trip)
        @test_nowarn convert_branch_angles_to_bus_angles(ps, v; exact=true)
    end

    @testset "angle conversion — exact=true throws on cycle mismatch" begin
        # Triangle: 3 buses, 3 branches → 1 cycle.
        # Branch angles that violate KVL cannot round-trip exactly.
        builder = PowerSystemBuilder{Float64}(3)
        add_two_bus_injector!(builder, 1, 2, Line(0.01, 0.05, 0.02))
        add_two_bus_injector!(builder, 2, 3, Line(0.01, 0.05, 0.02))
        add_two_bus_injector!(builder, 3, 1, Line(0.01, 0.05, 0.02))
        ps = build!(builder)

        # KVL-violating branch angles: θ12 + θ23 + θ31 ≠ 0
        v_bad = [1.0, 1.0, 1.0, 0.1, 0.2, 0.3]
        @test_throws ErrorException convert_branch_angles_to_bus_angles(ps, v_bad; exact=true)
    end

    @testset "step! keeps current vectors fresh" begin
        builder = PowerSystemBuilder{Float64}(2)
        add_one_bus_injector!(builder, 1, SynchronousMachineStatic(1.0))
        add_two_bus_injector!(builder, 1, 2, Line(0.01, 0.05, 0.02))
        ps = build!(builder)
        solver = GaussNewtonSolver()

        state1 = PowerFlowState(ps, ones(ps.n_controls))
        step!(state1, solver)
        @test any(!iszero, state1.single_bus_currents)
        @test any(!iszero, state1.branch_currents)
        saved_sbc = copy(state1.single_bus_currents)
        saved_bc  = copy(state1.branch_currents)
        compute_current_balance(state1)
        @test state1.single_bus_currents ≈ saved_sbc
        @test state1.branch_currents ≈ saved_bc
    end

    @testset "step! returns named tuple — slack branches" begin
        builder = PowerSystemBuilder{Float64}(2)
        add_one_bus_injector!(builder, 1, SynchronousMachineStatic(1.0))
        add_one_bus_injector!(builder, 2, ConstantPowerLoad(1.0))
        add_two_bus_injector!(builder, 1, 2, Line(0.01, 0.05, 0.02))
        ps = build!(builder)
        solver = GaussNewtonSolver()
        controls = [0.5, 1.0, 0.3, 0.1]

        # No-slack baseline
        state_ns = PowerFlowState(ps, controls)
        result_ns = step!(state_ns, solver)
        @test haskey(pairs(result_ns), :max_update)
        @test haskey(pairs(result_ns), :residual)
        @test haskey(pairs(result_ns), :variables_update)
        @test haskey(pairs(result_ns), :controls_update)
        @test haskey(pairs(result_ns), :jacobian_voltages)
        @test all(result_ns.controls_update .== 0)
    end

    @testset "diamond network — phantom vs rebuild on branch trip" begin
        # 4-bus diamond: buses 1-2-3-4-1 plus diagonal 2-4.
        # Two independent cycles share the diagonal branch.
        # Trip the diagonal via status mask ("phantom") and compare against
        # rebuilding the system without the diagonal branch.
        builder_full = PowerSystemBuilder{Float64}(4)
        add_one_bus_injector!(builder_full, 1, SynchronousMachineStatic(5.0))
        add_one_bus_injector!(builder_full, 3, SynchronousMachineStatic(5.0))
        add_one_bus_injector!(builder_full, 2, ConstantPowerLoad(1.0))
        add_one_bus_injector!(builder_full, 4, ConstantPowerLoad(1.0))
        add_two_bus_injector!(builder_full, 1, 2, Line(0.01, 0.1, 0.02))  # branch 1
        add_two_bus_injector!(builder_full, 2, 3, Line(0.01, 0.1, 0.02))  # branch 2
        add_two_bus_injector!(builder_full, 3, 4, Line(0.01, 0.1, 0.02))  # branch 3
        add_two_bus_injector!(builder_full, 4, 1, Line(0.01, 0.1, 0.02))  # branch 4
        add_two_bus_injector!(builder_full, 2, 4, Line(0.01, 0.1, 0.02))  # branch 5 (diagonal)
        ps_full = build!(builder_full)
        @test ps_full.n_cycles == 2

        controls = [3.0, 1.10, 0.5, 0.90, 2.0, 0.8, 0.5, 0.2]
        solver_full = GaussNewtonSolver()

        # Solve with all branches active
        state_all = PowerFlowState(ps_full, controls)
        stats_all = solve!(state_all, solver_full)
        @test stats_all["converged"]

        # Phantom: trip diagonal (branch 5) via status mask
        n_single = length(ps_full.single_bus_injectors)
        statuses_phantom = ones(n_single + ps_full.n_lines)
        statuses_phantom[n_single + 5] = 0.0
        state_phantom = PowerFlowState(ps_full, get_flat_start(ps_full), controls, statuses_phantom)
        stats_phantom = solve!(state_phantom, solver_full)
        @test stats_phantom["converged"]

        # Rebuild: construct system without the diagonal branch
        builder_reduced = PowerSystemBuilder{Float64}(4)
        add_one_bus_injector!(builder_reduced, 1, SynchronousMachineStatic(5.0))
        add_one_bus_injector!(builder_reduced, 3, SynchronousMachineStatic(5.0))
        add_one_bus_injector!(builder_reduced, 2, ConstantPowerLoad(1.0))
        add_one_bus_injector!(builder_reduced, 4, ConstantPowerLoad(1.0))
        add_two_bus_injector!(builder_reduced, 1, 2, Line(0.01, 0.1, 0.02))
        add_two_bus_injector!(builder_reduced, 2, 3, Line(0.01, 0.1, 0.02))
        add_two_bus_injector!(builder_reduced, 3, 4, Line(0.01, 0.1, 0.02))
        add_two_bus_injector!(builder_reduced, 4, 1, Line(0.01, 0.1, 0.02))
        ps_reduced = build!(builder_reduced)
        @test ps_reduced.n_cycles == 1

        solver_reduced = GaussNewtonSolver()
        state_rebuild = PowerFlowState(ps_reduced, controls)
        stats_rebuild = solve!(state_rebuild, solver_reduced)
        @test stats_rebuild["converged"]

        # Compare voltage magnitudes and common branch angles (branches 1-4).
        # The phantom approach is INVALID: the stale KVL cycle rows (which still
        # reference the tripped branch's angle variable) produce a different
        # solution from a clean rebuild. This motivates update_topology!.
        V_phantom = state_phantom.voltages[1:4]
        V_rebuild = state_rebuild.voltages[1:4]
        θ_phantom = state_phantom.voltages[5:8]
        θ_rebuild = state_rebuild.voltages[5:8]

        @test !isapprox(V_phantom, V_rebuild; atol=1e-6)
        @test !isapprox(θ_phantom, θ_rebuild; atol=1e-6)

        # update_topology! fixes this: recomputing cycles from active branches
        # produces results matching the clean rebuild.
        state_topo = PowerFlowState(ps_full, get_flat_start(ps_full), controls, statuses_phantom)
        update_topology!(state_topo)
        @test state_topo.n_cycles_active[] == 1
        stats_topo = solve!(state_topo, solver_full)
        @test stats_topo["converged"]

        V_topo = state_topo.voltages[1:4]
        θ_topo = state_topo.voltages[5:8]
        @test V_topo ≈ V_rebuild atol=1e-6
        @test θ_topo ≈ θ_rebuild atol=1e-6

        # Direct Jacobian-cache assertion: compute_jacobian_voltages on state_topo
        # must match the Jacobian of the clean-rebuild system at the same point,
        # after deleting the tripped branch-angle column and the pinning row.
        cycle_kw_topo = (
            cycle_participation = state_topo.active_cycle_participation,
            cycle_admittances   = state_topo.active_cycle_admittances,
            n_cycles            = state_topo.n_cycles_active[],
            _topology_snapshot  = state_topo._topology_snapshot,
        )
        J_topo = ResidualPowerFlow.compute_jacobian_voltages(
            ps_full, state_topo.voltages, controls, ExplicitAnalytical();
            statuses = state_topo.statuses, cycle_kw_topo...,
        )
        J_red = ResidualPowerFlow.compute_jacobian_voltages(
            ps_reduced, state_rebuild.voltages, controls, ExplicitAnalytical();
            statuses = state_rebuild.statuses,
        )
        tripped_col = ps_full.n_buses + 5
        keep_cols   = [j for j in 1:size(J_topo, 2) if j != tripped_col]
        pin_row     = size(J_topo, 1)
        keep_rows   = 1:(pin_row - 1)
        @test size(J_topo[keep_rows, keep_cols]) == size(J_red)
        @test Matrix(J_topo[keep_rows, keep_cols]) ≈ Matrix(J_red) atol = 1e-10

        # compute_residual! / compute_current_balance! shapes after update_topology!.
        n_buses_full = ps_full.n_buses
        r_buf = compute_residual!(state_topo)
        @test length(r_buf) == 2 * n_buses_full + state_topo.n_cycles_active[] + 1
        cb_buf = compute_current_balance!(state_topo)
        @test length(cb_buf) == 2 * n_buses_full

    end

    @testset "angle conversion — exact=true succeeds on loopy KVL-consistent angles" begin
        # Triangle: 3 buses, 3 branches 1→2, 2→3, 3→1, one cycle.
        # KVL-consistent branch angles satisfy θ12 + θ23 + θ31 = 0 along the cycle.
        builder = PowerSystemBuilder{Float64}(3)
        add_two_bus_injector!(builder, 1, 2, Line(0.01, 0.05, 0.02))
        add_two_bus_injector!(builder, 2, 3, Line(0.01, 0.05, 0.02))
        add_two_bus_injector!(builder, 3, 1, Line(0.01, 0.05, 0.02))
        ps = build!(builder)

        v_good = [1.0, 1.0, 1.0, 0.1, 0.2, -0.3]
        bus_angles = convert_branch_angles_to_bus_angles(ps, v_good; exact = true)
        @test length(bus_angles) == ps.n_buses
        branch_angles_back = convert_bus_angles_to_branch_angles(ps, bus_angles)
        @test branch_angles_back ≈ [0.1, 0.2, -0.3] atol = 1e-10
    end

end
