module PowsyblIOExt

using Powsybl
using ResidualPowerFlow
import ResidualPowerFlow: build_powersystem, controls_from_solution, target_variables
import ResidualPowerFlow: PowerSystemBuilder, add_one_bus_injector!, add_two_bus_injector!, build!

export build_powersystem, controls_from_solution, target_variables

# ── build_powersystem ──────────────────────────────────────────────────────────
#
# Load any file format supported by PowSyBl (MATPOWER .m, IIDM XML, CGMES, ...)
# and build a PowerSystem{T} with the standard component order:
#   generators → loads → shunts → lines → transformers
#
# K_DV: RPF-specific voltage droop coefficient for SynchronousMachineStatic.
#   Scalar applies the same value to all generators; a vector gives one entry
#   per generator in the order returned by get_generators().
#   Default 100.0 (IIDM files have no equivalent parameter).
#
# solve_load_flow: run PowSyBl's AC load flow on the network before reading it.
#   The PowerSystem itself is built from topology and component parameters, so
#   this flag does NOT change it. It changes the *operating point* stored in the
#   network, which is what controls_from_solution and target_variables read.
#   Pass true whenever you intend to call either of those: the built-in networks
#   ship a rounded operating point and no generator P/Q at all, so without a
#   solve the derived controls are only setpoint-accurate and the resulting
#   point is NOT an RPF fixed point (‖r‖ ≈ 1e-3 after a solve instead of ≈ 1e-8).
#   Default false, which leaves every existing call byte-identical.
#
# Returns a named tuple with all DataFrames and mappings needed by
# controls_from_solution and target_variables.

const _BUILTIN_NETWORKS = Dict{Symbol, Function}(
    :case9  => Powsybl.Network.create_ieee9,
    :case14 => Powsybl.Network.create_ieee14,
    :case30 => Powsybl.Network.create_ieee30,
    :case57 => Powsybl.Network.create_ieee57,
    :case118 => Powsybl.Network.create_ieee118,
    :case300 => Powsybl.Network.create_ieee300,
)

# Run PowSyBl's AC load flow in place; throw if any connected component fails to
# converge, so a silently unsolved network can never reach controls_from_solution.
function _run_ac_load_flow!(network::Powsybl.Network.NetworkHandle)
    result = Powsybl.LoadFlow.run_ac(network, Powsybl.LoadFlow.load_flow_parameters())
    statuses = result.component_results[!, :status]
    all(==(Powsybl.LoadFlow.CONVERGED), statuses) || throw(ErrorException(
        "AC load flow did not converge (component statuses: $(statuses))"))
    return network
end

function build_powersystem(
    name::Symbol;
    K_DV::Union{Real, AbstractVector{<:Real}} = 100.0,
    baseMVA::Real = 100.0,
    T::Type{<:Real} = Float64,
    solve_load_flow::Bool = false,
)
    creator = get(_BUILTIN_NETWORKS, name, nothing)
    isnothing(creator) && throw(ArgumentError("No built-in network for :$name. Pass a file path instead."))
    return build_powersystem(creator(); K_DV=K_DV, baseMVA=baseMVA, T=T,
                             solve_load_flow=solve_load_flow)
end

function build_powersystem(
    path::AbstractString;
    K_DV::Union{Real, AbstractVector{<:Real}} = 100.0,
    baseMVA::Real = 100.0,
    T::Type{<:Real} = Float64,
    solve_load_flow::Bool = false,
)
    return build_powersystem(Powsybl.Network.load(path); K_DV=K_DV, baseMVA=baseMVA, T=T,
                             solve_load_flow=solve_load_flow)
end

function build_powersystem(
    network::Powsybl.Network.NetworkHandle;
    K_DV::Union{Real, AbstractVector{<:Real}} = 100.0,
    baseMVA::Real = 100.0,
    T::Type{<:Real} = Float64,
    solve_load_flow::Bool = false,
)
    # Mutates the caller's network in place — opt-in only.
    solve_load_flow && _run_ac_load_flow!(network)
    buses_df  = Powsybl.Network.get_buses(network)
    generators_df   = Powsybl.Network.get_generators(network)
    # Drop disconnected generators (empty bus_id): they inject nothing and would
    # KeyError in bus_id_to_int. Filtering here keeps n_controls, the generator
    # loop, and controls_from_solution (which reads the returned generators_df) consistent.
    generators_df   = generators_df[
        [!ismissing(b) && b != "" for b in generators_df.bus_id], :]
    loads_df  = Powsybl.Network.get_loads(network)
    lines_df  = Powsybl.Network.get_lines(network)
    transformers_df  = Powsybl.Network.get_2_windings_transformers(network, true)
    shunts_df = Powsybl.Network.get_shunt_compensators(network)
    voltage_levels_df     = Powsybl.Network.get_voltage_levels(network)

    # voltage level id → nominal_v (kV)
    nominal_voltage_by_level = Dict{String, Float64}(
        voltage_levels_df[i, :id] => Float64(voltage_levels_df[i, :nominal_v])
        for i in axes(voltage_levels_df, 1)
    )

    # bus id (string) → internal integer (1..n_buses, in get_buses() order)
    n_buses = size(buses_df, 1)
    bus_id_to_int = Dict{String, Int}(
        buses_df[i, :id] => i for i in axes(buses_df, 1)
    )

    # bus id → voltage level id (for per-unit conversion)
    voltage_level_by_bus = Dict{String, String}(
        buses_df[i, :id] => buses_df[i, :voltage_level_id]
        for i in axes(buses_df, 1)
    )

    # K_DV per generator
    n_generators   = size(generators_df, 1)
    K_DV_per_generator =
        K_DV isa Real ? fill(Float64(K_DV), n_generators) : Vector{Float64}(K_DV)
    length(K_DV_per_generator) == n_generators ||
        throw(ArgumentError(
            "K_DV length $(length(K_DV_per_generator)) ≠ n_generators $n_generators"))

    builder = PowerSystemBuilder{T}(n_buses)

    # --- Generators ---
    for i in axes(generators_df, 1)
        bus_int = bus_id_to_int[generators_df[i, :bus_id]]
        add_one_bus_injector!(builder, bus_int,
                              SynchronousMachineStatic(T(K_DV_per_generator[i])))
    end

    # --- Loads ---
    for i in axes(loads_df, 1)
        bus_int = bus_id_to_int[loads_df[i, :bus_id]]
        add_one_bus_injector!(builder, bus_int, ConstantPowerLoad(one(T)))
    end

    # --- Shunts ---
    # g/b_per_section are in the linear sections table, section_count in the
    # shunt table. MATPOWER bus Gs (shunt conductance) rides on g_per_section;
    # dropping it leaves an unbalanced active injection at the shunt bus.
    sections_df = Powsybl.Network.get_linear_shunt_compensator_sections(network, true)
    b_per_section_map = Dict{String, Float64}(
        sections_df[i, :id] => Float64(sections_df[i, :b_per_section])
        for i in axes(sections_df, 1)
    )
    g_per_section_map = Dict{String, Float64}(
        sections_df[i, :id] => Float64(sections_df[i, :g_per_section])
        for i in axes(sections_df, 1)
    )
    for i in axes(shunts_df, 1)
        shunt_id = shunts_df[i, :id]
        haskey(b_per_section_map, shunt_id) || continue  # skip non-linear shunts
        bus_id   = shunts_df[i, :bus_id]
        bus_int  = bus_id_to_int[bus_id]
        voltage_level_id    = voltage_level_by_bus[bus_id]
        nominal_voltage = nominal_voltage_by_level[voltage_level_id]
        Z_base   = nominal_voltage^2 / baseMVA
        n_sections    = Int(shunts_df[i, :section_count])
        B_pu     = b_per_section_map[shunt_id] * n_sections * Z_base
        # Built-in (create_ieeeN) networks report g_per_section = NaN for shunts
        # with no conductance; .mat imports report 0.0. Guard NaN → 0.
        g_raw    = g_per_section_map[shunt_id]
        G_pu     = (isnan(g_raw) ? 0.0 : g_raw) * n_sections * Z_base
        add_one_bus_injector!(builder, bus_int, Shunt(T(G_pu), T(B_pu)))
    end

    active_from = Int[]
    active_to   = Int[]

    # --- Lines ---
    # Same-VL lines: standard symmetric pi-model.
    #   Z_base = nominal_voltage_from² / baseMVA,
#   B = (b1+b2)*Z_base (total charging split equally).
    # Cross-VL lines: asymmetric pi-model (AsymmetricBranch).
    #   In Powsybl IIDM, a MATPOWER "line" connecting buses at different nominal voltages
    #   carries large asymmetric b1/b2 shunts that encode the natural
#   turns-ratio correction.
    #   Y_11 = Y_S·Z_base1 + j·B1,  Y_12 = −Y_S·Z_base12,  Y_22 = Y_S·Z_base2 + j·B2
    #   where Z_base1=nomV1²/S, Z_base12=nomV1·nomV2/S, Z_base2=nomV2²/S.
    for i in axes(lines_df, 1)
        from_bus  = bus_id_to_int[lines_df[i, :bus1_id]]
        to_bus    = bus_id_to_int[lines_df[i, :bus2_id]]
        nominal_voltage_from = nominal_voltage_by_level[lines_df[i, :voltage_level1_id]]
        nominal_voltage_to = nominal_voltage_by_level[lines_df[i, :voltage_level2_id]]
        R_SI = Float64(lines_df[i, :r])
        X_SI = Float64(lines_df[i, :x])
        if nominal_voltage_from == nominal_voltage_to
            Z_base = nominal_voltage_from^2 / baseMVA
            R_pu = R_SI / Z_base
            X_pu = X_SI / Z_base
            B_pu = (Float64(lines_df[i, :b1]) + Float64(lines_df[i, :b2])) * Z_base
            add_two_bus_injector!(builder, from_bus, to_bus,
                Branch(T(R_pu), T(X_pu), T(B_pu), one(T), zero(T)))
        else
            Z_base12 = nominal_voltage_from * nominal_voltage_to / baseMVA
            Z_base1  = nominal_voltage_from^2 / baseMVA
            Z_base2  = nominal_voltage_to^2 / baseMVA
            Y_S12    = 1 / complex(R_SI / Z_base12, X_SI / Z_base12)
            TAP      = nominal_voltage_from / nominal_voltage_to
            G1       = Float64(lines_df[i, :g1]) * Z_base1
            B1       = Float64(lines_df[i, :b1]) * Z_base1
            G2       = Float64(lines_df[i, :g2]) * Z_base2
            B2       = Float64(lines_df[i, :b2]) * Z_base2
            Y_11     = Y_S12 * TAP  + complex(G1, B1)
            Y_12     = -Y_S12
            Y_21     = Y_12
            Y_22     = Y_S12 / TAP  + complex(G2, B2)
            add_two_bus_injector!(builder, from_bus, to_bus,
                AsymmetricBranch(Complex{T}(Y_11), Complex{T}(Y_12), Complex{T}(Y_21), Complex{T}(Y_22)))
        end
        push!(active_from, from_bus)
        push!(active_to, to_bus)
    end

    # --- Two-winding transformers ---
    # TAP = (ratedU1/nominal_voltage_from) / (ratedU2/nominal_voltage_to)
    #       (PowSyBl convention)
    # IIDM stores r, x, g, b in Ω/S referred to side 2;
    # Z_base = nominal_voltage_to²/baseMVA.
    # θ_shift: PowSyBl's :alpha (current-tap phase shift, degrees) carries the
    # opposite sign to the MATPOWER `shift` that RPF's convention follows, so
    # θ_shift = deg2rad(-alpha). NaN/missing alpha (no phase tap changer) → 0.
    #
    # Build the admittance matrix directly as an AsymmetricBranch so the
    # magnetizing shunt (g + jb) sits ENTIRELY on side 1 — the IIDM convention
    # places the magnetizing branch on the side-1 (network/primary) terminal:
    #   Y_11 = Y_S/TAP² + (G + jB),  Y_12 = −Y_S/TAP·e^{+jθ},
    #   Y_21 = −Y_S/TAP·e^{−jθ},     Y_22 = Y_S
    # The symmetric Branch π-model instead splits the shunt B/2 across BOTH
    # terminals, which misallocates the shunt reactive across the turns ratio and
    # leaves an equal-and-opposite reactive residual at the two transformer buses
    # (the series loss is unaffected; verified by per-terminal q vs Powsybl).
    # Y_12/Y_21 are identical to the Branch model, so ratio/phase behaviour is
    # unchanged — only the shunt moves.
    has_alpha = hasproperty(transformers_df, :alpha)
    has_g     = hasproperty(transformers_df, :g)
    for i in axes(transformers_df, 1)
        from_bus  = bus_id_to_int[transformers_df[i, :bus1_id]]
        to_bus    = bus_id_to_int[transformers_df[i, :bus2_id]]
        nominal_voltage_from =
            nominal_voltage_by_level[transformers_df[i, :voltage_level1_id]]
        nominal_voltage_to =
            nominal_voltage_by_level[transformers_df[i, :voltage_level2_id]]
        ratedU1   = Float64(transformers_df[i, :rated_u1])
        ratedU2   = Float64(transformers_df[i, :rated_u2])
        Z_base    = nominal_voltage_to^2 / baseMVA
        TAP       = (ratedU1 / nominal_voltage_from) / (ratedU2 / nominal_voltage_to)
        R_pu = Float64(transformers_df[i, :r]) / Z_base
        X_pu = Float64(transformers_df[i, :x]) / Z_base
        B_pu = Float64(transformers_df[i, :b]) * Z_base
        G_pu = (has_g ? Float64(transformers_df[i, :g]) : 0.0) * Z_base
        α_deg   = has_alpha ? Float64(transformers_df[i, :alpha]) : 0.0
        θ_shift = isnan(α_deg) ? 0.0 : deg2rad(-α_deg)
        Y_S  = 1 / complex(R_pu, X_pu)
        Y_11 = (Y_S + complex(G_pu, B_pu)) / TAP^2
        Y_12 = -Y_S * (1 / TAP) * (1 / exp(-im * θ_shift))
        Y_21 = -Y_S * (1 / TAP) * (1 / exp(im * θ_shift))
        Y_22 = Y_S
        add_two_bus_injector!(builder, from_bus, to_bus,
            AsymmetricBranch(Complex{T}(Y_11), Complex{T}(Y_12), Complex{T}(Y_21), Complex{T}(Y_22)))
        push!(active_from, from_bus)
        push!(active_to, to_bus)
    end

    power_system = build!(builder)

    return (
        power_system  = power_system,
        buses_df      = buses_df,
        generators_df       = generators_df,
        loads_df      = loads_df,
        nominal_voltage_by_level   = nominal_voltage_by_level,
        voltage_level_by_bus        = voltage_level_by_bus,
        baseMVA       = Float64(baseMVA),
        K_DV_per_generator      = K_DV_per_generator,
        active_from   = active_from,
        active_to     = active_to,
        bus_id_to_int = bus_id_to_int,
    )
end

# ── controls_from_solution ────────────────────────────────────────────────────
#
# Derive the RPF control vector from the operating point encoded in the
# loaded network (Pg/Qg/Vm on generators, P0/Q0 on loads).
# Component order matches build_powersystem: generators first, then loads.
#
# Requires the network to be in a solved state (Powsybl.LoadFlow.run_ac called
# before build_powersystem).  Uses the solved terminal reactive power (:q) rather
# than the setpoint (:target_q), which may differ for voltage-regulated generators.
#
# The per-generator Pg/Qg below give a first-cut [T_M, V_ref]; the generator
# slots are then refined by controls_from_voltages, which inverts RPF's FULL
# current balance at each generator bus (absorbing the branch charging / shunt
# current flowing into the bus, not just the nameplate Qg).  Matching Qg alone
# leaves an O(1) residual on networks with significant charging (e.g. IEEE 57 /
# 300), so the solved point would not be an RPF fixed point; with the exact
# inversion it is, and a flat-start RPF solve reproduces the PowSyBl solution.

function controls_from_solution(system)
    generators_df      = system.generators_df
    buses_df     = system.buses_df
    loads_df     = system.loads_df
    nominal_voltage_by_level  = system.nominal_voltage_by_level
    voltage_level_by_bus       = system.voltage_level_by_bus
    baseMVA      = system.baseMVA
    K_DV_per_generator     = system.K_DV_per_generator
    power_system = system.power_system

    # Actual solved bus voltage magnitudes (per unit), keyed by bus id.
    # Used instead of target_v so that V_ref is consistent with the LF solution.
    bus_vm_pu = Dict{String, Float64}(
        buses_df[i, :id] =>
            Float64(buses_df[i, :v_mag]) /
            nominal_voltage_by_level[buses_df[i, :voltage_level_id]]
        for i in axes(buses_df, 1)
    )

    controls = zeros(Float64, power_system.n_controls)
    cursor   = 1

    for i in axes(generators_df, 1)
        # Powsybl terminal :p/:q use consumer convention (negative = production).
        # Fall back to target setpoints when LF has not been run (NaN).
        p_raw    = Float64(generators_df[i, :p])
        Pg       = (isnan(p_raw) ? Float64(generators_df[i, :target_p]) : -p_raw) / baseMVA
        q_raw    = Float64(generators_df[i, :q])
        Qg       = (isnan(q_raw) ? Float64(generators_df[i, :target_q]) : -q_raw) / baseMVA
        vm_raw   = get(bus_vm_pu, generators_df[i, :bus_id], NaN)
        nominal_voltage = nominal_voltage_by_level[generators_df[i, :voltage_level_id]]
        Vm       = isnan(vm_raw) ?
                   Float64(generators_df[i, :target_v]) / nominal_voltage : vm_raw

        K_DV  = K_DV_per_generator[i]
        T_M   = Pg                           # T_M = i_d * Vm = (Pg/Vm)*Vm = Pg
        V_ref = Vm + Qg / (Vm * K_DV)       # from i_q = -K_DV*(V_ref - Vm)

        controls[cursor]   = T_M
        controls[cursor+1] = V_ref
        cursor += 2
    end

    for i in axes(loads_df, 1)
        controls[cursor]   = Float64(loads_df[i, :p0]) / baseMVA
        controls[cursor+1] = Float64(loads_df[i, :q0]) / baseMVA
        cursor += 2
    end

    # Refine the generator slots so RPF's full current balance is exactly zero
    # at each generator bus at the solved voltages (the Qg-match above is only a
    # first-cut; it ignores branch charging / shunt current into the gen bus).
    return ResidualPowerFlow.controls_from_voltages(
        power_system, target_variables(system), controls,
    )
end

# ── target_variables ──────────────────────────────────────────────────────────
#
# Build the variable vector [Vm_1..Vm_n; θ_branch_1..θ_branch_m] from the
# bus voltages and branch angle differences in the loaded network.
# Branch order matches build_powersystem: lines first, then transformers.

function target_variables(system)
    buses_df      = system.buses_df
    nominal_voltage_by_level   = system.nominal_voltage_by_level
    bus_id_to_int = system.bus_id_to_int
    voltage_level_by_bus        = system.voltage_level_by_bus
    active_from   = system.active_from
    active_to     = system.active_to
    power_system  = system.power_system

    n_buses   = power_system.n_buses
    n_lines   = power_system.n_lines
    variables = zeros(Float64, n_buses + n_lines)

    bus_angle_rad = Dict{Int, Float64}()
    for i in axes(buses_df, 1)
        bus_id   = buses_df[i, :id]
        bus_index      = bus_id_to_int[bus_id]
        nominal_voltage = nominal_voltage_by_level[voltage_level_by_bus[bus_id]]
        variables[bus_index]     = Float64(buses_df[i, :v_mag]) / nominal_voltage
        bus_angle_rad[bus_index] = deg2rad(Float64(buses_df[i, :v_angle]))
    end

    for ii in eachindex(active_from)
        f = active_from[ii]
        t = active_to[ii]
        variables[n_buses + ii] = bus_angle_rad[t] - bus_angle_rad[f]
    end

    return variables
end

end # module PowsyblIOExt
