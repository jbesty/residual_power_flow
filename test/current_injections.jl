@testset "Current injections" begin

    # Each component's compute_current_injection is tested in isolation, then the
    # PowerSystem wrapper is tested to verify components are correctly dispatched and
    # their results are assembled into the global current balance.

    @testset "Components" begin

        @testset "SynchronousMachineStatic" begin
            # Controls: [T_M (mechanical torque), V_ref (voltage reference)]
            # Physics: i_d = T_M / V,  i_q = -K_DV * (V_ref - V)
            # T_M=1.21, V_ref=1.0, V=1.1 → i_d = 1.21/1.1 = 1.1, i_q = -2*(1.0-1.1) = 0.2
            sm = SynchronousMachineStatic(2.0)
            current = compute_current_injection(sm, 1.1, [1.21, 1.0])
            @test isapprox(current[1],  1.1; rtol = 1e-12, atol = 0)
            @test isapprox(current[2], 0.2; rtol = 1e-12, atol = 0)
        end

        @testset "ZIPLoad" begin
            # Controls: [P, Q] (active and reactive power consumed, in per unit)
            # Physics: s_injected = -(P + jQ) * (a + b*(V/V_0) + c*(V/V_0)^2)
            #          i = conj(s_injected / V)

            # Generic ZIP with mixed weights (a=0.2, b=0.3, c=0.5):
            # V=0.8, factor=0.2+0.3*0.8+0.5*0.64=0.76, s=-(0.76+0.38i), i=conj(s/0.8)=-0.95+0.475i
            zip = ZIPLoad(1.0, 0.2, 0.3, 0.5)
            current_zip = compute_current_injection(zip, 0.8, [1.0, 0.5])
            @test isapprox(current_zip[1], -0.95;  rtol = 1e-12, atol = 0)
            @test isapprox(current_zip[2],  0.475; rtol = 1e-12, atol = 0)

            # ConstantPowerLoad (a=1): ZIP factor = 1 at all voltages.
            # V=1.0, P=0.5, Q=0.2 → i = conj(-(0.5+0.2i)/1.0) = -0.5+0.2i
            cp = ConstantPowerLoad(1.0)
            current_cp = compute_current_injection(cp, 1.0, [0.5, 0.2])
            @test isapprox(current_cp[1], -0.5; rtol = 1e-12, atol = 0)
            @test isapprox(current_cp[2],  0.2; rtol = 1e-12, atol = 0)

            # ConstantCurrentLoad (b=1): current magnitude is independent of voltage.
            # Verified by checking that V=0.8 and V=1.2 give the same result.
            cc = ConstantCurrentLoad(1.0)
            current_cc_low_v  = compute_current_injection(cc, 0.8, [1.0, 0.4])
            current_cc_high_v = compute_current_injection(cc, 1.2, [1.0, 0.4])
            @test isapprox(current_cc_low_v[1], -1.0; rtol = 1e-12, atol = 0)
            @test isapprox(current_cc_low_v[2],  0.4; rtol = 1e-12, atol = 0)
            @test isapprox(current_cc_low_v, current_cc_high_v; rtol = 1e-12, atol = 0)

            # ConstantImpedanceLoad (c=1): current scales linearly with voltage.
            # V=1.2, P=1.0, Q=0.4 → factor=1.44, s=-(1.44+0.576i), i=conj(s/1.2)=-1.2+0.48i
            ci = ConstantImpedanceLoad(1.0)
            current_ci = compute_current_injection(ci, 1.2, [1.0, 0.4])
            @test isapprox(current_ci[1], -1.2;  rtol = 1e-12, atol = 0)
            @test isapprox(current_ci[2],  0.48; rtol = 1e-12, atol = 0)
        end

        @testset "Shunt" begin
            # No controls. Physics: i = -(G + jB) * V
            sh = Shunt(0.1, -0.2)
            current = compute_current_injection(sh, 1.0, Float64[])
            @test current[1] == -0.1
            @test current[2] == 0.2
        end

        @testset "Branch" begin
            # New signature: (component, v_from_mag, v_to_mag, theta)
            # Returns (i_from, i_to) where each is [i_d, i_q].
            # Each current is expressed in the local frame of its terminal bus:
            #   i_from: from-bus as reference (angle 0), to-bus at angle +theta
            #   i_to:   to-bus as reference (angle 0), from-bus at angle -theta
            br = Line(0.01, 0.05, 0.02)

            i_from, i_to = compute_current_injection(br, 1.0, 1.0, 0.1)
            # From-end in from-bus local frame
            v_from         = complex(1.0, 0.0)
            v_to_from_frame = 1.0 * cis(0.1)
            expected_from  = -(br.Y_11 * v_from + br.Y_12 * v_to_from_frame)
            # To-end in to-bus local frame
            v_to            = complex(1.0, 0.0)
            v_from_to_frame = 1.0 * cis(-0.1)
            expected_to     = -(br.Y_22 * v_to + br.Y_21 * v_from_to_frame)
            @test isapprox(i_from[1], expected_from.re; rtol = 1e-12, atol = 0)
            @test isapprox(i_from[2], expected_from.im; rtol = 1e-12, atol = 0)
            @test isapprox(i_to[1],   expected_to.re;  rtol = 1e-12, atol = 0)
            @test isapprox(i_to[2],   expected_to.im;  rtol = 1e-12, atol = 0)
        end

        @testset "AsymmetricBranch" begin
            # AsymmetricBranch stores pre-computed Y entries directly (Y_11 ≠ Y_22).
            # Verify current injection matches manual Y-matrix calculation.
            Y11 = complex(1.0, -5.0)
            Y12 = complex(-1.0, 5.0)
            Y21 = complex(-1.0, 5.0)
            Y22 = complex(0.8, -4.5)   # intentionally different from Y11
            ab = AsymmetricBranch(Y11, Y12, Y21, Y22)

            i_from, i_to = compute_current_injection(ab, 1.0, 0.95, 0.05)
            # From-end in from-bus local frame
            v_from          = complex(1.0, 0.0)
            v_to_from_frame = 0.95 * cis(0.05)
            expected_from   = -(Y11 * v_from + Y12 * v_to_from_frame)
            # To-end in to-bus local frame
            v_to            = complex(0.95, 0.0)
            v_from_to_frame = 1.0 * cis(-0.05)
            expected_to     = -(Y22 * v_to + Y21 * v_from_to_frame)
            @test isapprox(i_from[1], expected_from.re; rtol = 1e-12, atol = 0)
            @test isapprox(i_from[2], expected_from.im; rtol = 1e-12, atol = 0)
            @test isapprox(i_to[1],   expected_to.re;   rtol = 1e-12, atol = 0)
            @test isapprox(i_to[2],   expected_to.im;   rtol = 1e-12, atol = 0)

            # Symmetric limit: when Y11 == Y22, AsymmetricBranch must agree with Branch.
            br = Line(0.01, 0.05, 0.02)
            ab_sym = AsymmetricBranch(br.Y_11, br.Y_12, br.Y_21, br.Y_22)
            i_from_br, i_to_br = compute_current_injection(br, 1.1, 0.9, 0.1)
            i_from_ab, i_to_ab = compute_current_injection(ab_sym, 1.1, 0.9, 0.1)
            @test isapprox(i_from_ab, i_from_br; rtol = 1e-12, atol = 0)
            @test isapprox(i_to_ab,   i_to_br;   rtol = 1e-12, atol = 0)
        end

        @testset "Control inversion" begin
            # compute_control_from_current_injection is the inverse of compute_current_injection:
            # given a voltage and a desired current, it returns the controls that produce it.
            sm = SynchronousMachineStatic(2.0)
            # T_M = i_d * V = 0.5*1.1 = 0.55,  V_ref = V - i_q/K_DV = 1.1 - (-0.1)/2 = 1.15
            ctrl = ResidualPowerFlow.compute_control_from_current_injection(sm, 1.1, [0.5, -0.1])
            @test isapprox(ctrl[1], 0.55; rtol = 1e-12, atol = 0)
            @test isapprox(ctrl[2], 1.15; rtol = 1e-12, atol = 0)
            # Roundtrip: re-applying the inverted controls must recover the original current.
            current_roundtrip = compute_current_injection(sm, 1.1, ctrl)
            @test isapprox(current_roundtrip[1],  0.5; rtol = 1e-12, atol = 0)
            @test isapprox(current_roundtrip[2], -0.1; rtol = 1e-12, atol = 0)
        end

    end # Components

    @testset "PowerSystem wrapper" begin
        # Verifies that compute_current_injection(ps, ...) correctly dispatches to each
        # component and that compute_current_balance assembles injections via scatter.
        #
        # Network: 2 buses. Bus 1 has a generator (sm) and a shunt (sh).
        # Branch br connects bus 1 → bus 2.
        # Variables: [V_bus1, V_bus2, θ_branch] = [1.1, 1.0, 0.1]
        # Controls: [T_M, V_ref] = [1.21, 1.0]
        sm = SynchronousMachineStatic(2.0)
        sh = Shunt(0.1, -0.2)
        br = Line(0.01, 0.05, 0.02)

        builder = PowerSystemBuilder{Float64}(2)
        add_one_bus_injector!(builder, 1, sm)
        add_one_bus_injector!(builder, 1, sh)
        add_two_bus_injector!(builder, 1, 2, br)
        ps = build!(builder)

        voltages = [1.1, 1.0, 0.1]   # [V_bus1, V_bus2, θ_branch]
        controls = [1.21, 1.0]        # [T_M, V_ref] for sm

        one_bus, two_bus = compute_current_injection(ps, voltages, controls)

        # single_bus_currents: [sm_id, sm_iq, sh_id, sh_iq]
        expected_sm = compute_current_injection(sm, 1.1, [1.21, 1.0])
        expected_sh = compute_current_injection(sh, 1.1, Float64[])
        @test isapprox(one_bus[1:2], expected_sm; rtol = 1e-12, atol = 0)
        @test isapprox(one_bus[3:4], expected_sh; rtol = 1e-12, atol = 0)

        # branch_currents: [i_from_d, i_from_q, i_to_d, i_to_q]
        expected_from, expected_to = compute_current_injection(br, 1.1, 1.0, 0.1)
        @test isapprox(two_bus[1:2], expected_from; rtol = 1e-12, atol = 0)
        @test isapprox(two_bus[3:4], expected_to;   rtol = 1e-12, atol = 0)

        # Current balance at bus 1: sm + sh + i_from
        # Current balance at bus 2: i_to
        balance = compute_current_balance(ps, voltages, controls)
        expected_bus1 = expected_sm .+ expected_sh .+ expected_from
        expected_bus2 = expected_to
        @test isapprox(balance[1:2], expected_bus1; rtol = 1e-12, atol = 0)
        @test isapprox(balance[3:4], expected_bus2; rtol = 1e-12, atol = 0)
    end

    @testset "PowerSystem wrapper — AsymmetricBranch" begin
        # Same structure as the Branch wrapper test, but using AsymmetricBranch.
        sm = SynchronousMachineStatic(2.0)
        Y11 = complex(1.0, -5.0)
        Y12 = complex(-1.0, 5.0)
        Y21 = complex(-1.0, 5.0)
        Y22 = complex(0.8, -4.5)
        ab = AsymmetricBranch(Y11, Y12, Y21, Y22)

        builder = PowerSystemBuilder{Float64}(2)
        add_one_bus_injector!(builder, 1, sm)
        add_two_bus_injector!(builder, 1, 2, ab)
        ps = build!(builder)

        voltages = [1.1, 0.95, 0.05]
        controls = [1.21, 1.0]

        one_bus, two_bus = compute_current_injection(ps, voltages, controls)

        expected_sm = compute_current_injection(sm, 1.1, [1.21, 1.0])
        @test isapprox(one_bus[1:2], expected_sm; rtol = 1e-12, atol = 0)

        expected_from, expected_to = compute_current_injection(ab, 1.1, 0.95, 0.05)
        @test isapprox(two_bus[1:2], expected_from; rtol = 1e-12, atol = 0)
        @test isapprox(two_bus[3:4], expected_to;   rtol = 1e-12, atol = 0)

        # Current balance: bus 1 = sm + i_from, bus 2 = i_to
        balance = compute_current_balance(ps, voltages, controls)
        @test isapprox(balance[1:2], expected_sm .+ expected_from; rtol = 1e-12, atol = 0)
        @test isapprox(balance[3:4], expected_to; rtol = 1e-12, atol = 0)
    end

    @testset "Component status masking" begin
        sm = SynchronousMachineStatic(2.0)
        sh = Shunt(0.1, -0.2)
        br = Line(0.01, 0.05, 0.02)

        builder = PowerSystemBuilder{Float64}(2)
        add_one_bus_injector!(builder, 1, sm)
        add_one_bus_injector!(builder, 1, sh)
        add_two_bus_injector!(builder, 1, 2, br)
        ps = build!(builder)

        voltages = [1.1, 1.0, 0.1]
        controls = [1.21, 1.0]

        n_single = length(ps.single_bus_injectors)   # 2

        # Status [0, 1, 1]: zero the generator (index 1 of 2 single-bus + 1 branch).
        statuses_no_gen = [0.0, 1.0, 1.0]
        one_bus_masked, two_bus_masked = compute_current_injection(ps, voltages, controls; statuses = statuses_no_gen)

        # Generator zeroed, shunt unchanged.
        @test isapprox(one_bus_masked[1], 0.0; atol = 1e-15)
        @test isapprox(one_bus_masked[2], 0.0; atol = 1e-15)
        @test isapprox(one_bus_masked[3:4], compute_current_injection(sh, 1.1, Float64[]); rtol = 1e-12)

        # Branch unchanged when its status is 1.
        expected_from, expected_to = compute_current_injection(br, 1.1, 1.0, 0.1)
        @test isapprox(two_bus_masked[1:2], expected_from; rtol = 1e-12)
        @test isapprox(two_bus_masked[3:4], expected_to;   rtol = 1e-12)

        # Status [1, 1, 0]: zero the branch.
        statuses_no_branch = [1.0, 1.0, 0.0]
        one_bus_full, two_bus_zeroed = compute_current_injection(ps, voltages, controls; statuses = statuses_no_branch)
        @test isapprox(two_bus_zeroed, zeros(4); atol = 1e-15)
        # Single-bus components unchanged.
        one_bus_ref, _ = compute_current_injection(ps, voltages, controls)
        @test isapprox(one_bus_full, one_bus_ref; rtol = 1e-12)

        # All-ones statuses must give the same result as no statuses.
        statuses_all_ones = ones(3)
        one_bus_ones, two_bus_ones = compute_current_injection(ps, voltages, controls; statuses = statuses_all_ones)
        @test isapprox(one_bus_ones, one_bus_ref; rtol = 1e-12)
        one_bus_nostat, two_bus_nostat = compute_current_injection(ps, voltages, controls)
        @test isapprox(two_bus_ones, two_bus_nostat; rtol = 1e-12)

        # PowerFlowState default statuses are all-ones; behaviour identical to pre-status code.
        state_default = PowerFlowState(ps, voltages, controls)
        @test all(state_default.statuses .== 1.0)
        @test length(state_default.statuses) == 2 + 1  # 2 single-bus + 1 physical branch
    end

end  # Current injections

if _powsybl_available
    @testset "Component status — IEEE 9-bus re-routing" begin
        # Solve the full IEEE 9-bus system, then zero one branch and verify that the
        # solved state changes (i.e. the network actually re-routes).
        ps = _CASE9.power_system
        ctrl = controls_from_solution(_CASE9)

        state_full = PowerFlowState(ps, ctrl)
        solved_full, stats_full = solve_distributed_slack(state_full, [1])
        @test stats_full["converged"]

        # Zero branch 1 (the first physical branch): status index n_single + 1.
        n_single = length(ps.single_bus_injectors)
        statuses_no_br1 = copy(solved_full.statuses)
        statuses_no_br1[n_single + 1] = 0.0

        state_no_br1 = PowerFlowState(ps, solved_full.voltages, solved_full.controls, statuses_no_br1)
        solved_no_br1, stats_no_br1 = solve_distributed_slack(state_no_br1, [1])
        @test stats_no_br1["converged"]

        # At least some voltage/angle must differ between the two solutions.
        @test !isapprox(solved_full.voltages, solved_no_br1.voltages; rtol = 1e-4)
    end

    @testset "In-place compute_current_balance! matches allocating variant" begin
        ps   = _CASE9.power_system
        ctrl = controls_from_solution(_CASE9)

        # Solve to get a non-trivial (converged) state
        state, stats = solve_distributed_slack(PowerFlowState(ps, ctrl), [1])
        @test stats["converged"]

        # Allocating reference
        cb_alloc = compute_current_balance(state)

        # In-place variant
        cb_inplace = compute_current_balance!(state)

        # Values must match
        @test cb_inplace ≈ cb_alloc

        # Result is the pre-allocated buffer (pointer identity)
        @test cb_inplace === state.current_balance_buffer
    end

    @testset "In-place compute_residual! matches allocating variant" begin
        ps   = _CASE9.power_system
        ctrl = controls_from_solution(_CASE9)

        # Solve to get a non-trivial (converged) state
        state, stats = solve_distributed_slack(PowerFlowState(ps, ctrl), [1])
        @test stats["converged"]

        # Allocating reference
        r_alloc = compute_residual(state)

        # In-place variant
        r_inplace = compute_residual!(state)

        # Values must match
        @test r_inplace ≈ r_alloc

        # Result shares memory with the pre-allocated buffer (view into it)
        @test pointer(r_inplace) == pointer(state.residual_buffer)
    end
end
