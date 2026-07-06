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

function build_powersystem(
    name::Symbol;
    K_DV::Union{Real, AbstractVector{<:Real}} = 100.0,
    baseMVA::Real = 100.0,
    T::Type{<:Real} = Float64,
)
    creator = get(_BUILTIN_NETWORKS, name, nothing)
    isnothing(creator) && throw(ArgumentError("No built-in network for :$name. Pass a file path instead."))
    return build_powersystem(creator(); K_DV=K_DV, baseMVA=baseMVA, T=T)
end

function build_powersystem(
    path::AbstractString;
    K_DV::Union{Real, AbstractVector{<:Real}} = 100.0,
    baseMVA::Real = 100.0,
    T::Type{<:Real} = Float64,
)
    return build_powersystem(Powsybl.Network.load(path); K_DV=K_DV, baseMVA=baseMVA, T=T)
end

function build_powersystem(
    net::Powsybl.Network.NetworkHandle;
    K_DV::Union{Real, AbstractVector{<:Real}} = 100.0,
    baseMVA::Real = 100.0,
    T::Type{<:Real} = Float64,
)
    buses_df  = Powsybl.Network.get_buses(net)
    gens_df   = Powsybl.Network.get_generators(net)
    # Drop disconnected generators (empty bus_id): they inject nothing and would
    # KeyError in bus_id_to_int. Filtering here keeps n_controls, the generator
    # loop, and controls_from_solution (which reads the returned gens_df) consistent.
    gens_df   = gens_df[[!ismissing(b) && b != "" for b in gens_df.bus_id], :]
    loads_df  = Powsybl.Network.get_loads(net)
    lines_df  = Powsybl.Network.get_lines(net)
    xfmrs_df  = Powsybl.Network.get_2_windings_transformers(net, true)
    shunts_df = Powsybl.Network.get_shunt_compensators(net)
    vl_df     = Powsybl.Network.get_voltage_levels(net)

    # voltage level id → nominal_v (kV)
    vl_nominalV = Dict{String, Float64}(
        vl_df[i, :id] => Float64(vl_df[i, :nominal_v])
        for i in axes(vl_df, 1)
    )

    # bus id (string) → internal integer (1..n_buses, in get_buses() order)
    n_buses = size(buses_df, 1)
    bus_id_to_int = Dict{String, Int}(
        buses_df[i, :id] => i for i in axes(buses_df, 1)
    )

    # bus id → voltage level id (for per-unit conversion)
    bus_vl = Dict{String, String}(
        buses_df[i, :id] => buses_df[i, :voltage_level_id]
        for i in axes(buses_df, 1)
    )

    # K_DV per generator
    n_gens   = size(gens_df, 1)
    K_DV_vec = K_DV isa Real ? fill(Float64(K_DV), n_gens) : Vector{Float64}(K_DV)
    length(K_DV_vec) == n_gens ||
        throw(ArgumentError("K_DV length $(length(K_DV_vec)) ≠ n_generators $n_gens"))

    builder = PowerSystemBuilder{T}(n_buses)

    # --- Generators ---
    for i in axes(gens_df, 1)
        bus_int = bus_id_to_int[gens_df[i, :bus_id]]
        add_one_bus_injector!(builder, bus_int, SynchronousMachineStatic(T(K_DV_vec[i])))
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
    sections_df = Powsybl.Network.get_linear_shunt_compensator_sections(net, true)
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
        vl_id    = bus_vl[bus_id]
        nominalV = vl_nominalV[vl_id]
        Z_base   = nominalV^2 / baseMVA
        n_sec    = Int(shunts_df[i, :section_count])
        B_pu     = b_per_section_map[shunt_id] * n_sec * Z_base
        # Built-in (create_ieeeN) networks report g_per_section = NaN for shunts
        # with no conductance; .mat imports report 0.0. Guard NaN → 0.
        g_raw    = g_per_section_map[shunt_id]
        G_pu     = (isnan(g_raw) ? 0.0 : g_raw) * n_sec * Z_base
        add_one_bus_injector!(builder, bus_int, Shunt(T(G_pu), T(B_pu)))
    end

    active_from = Int[]
    active_to   = Int[]

    # --- Lines ---
    # Same-VL lines: standard symmetric pi-model.
    #   Z_base = nominalV1² / baseMVA,  B = (b1+b2)*Z_base (total charging split equally).
    # Cross-VL lines: asymmetric pi-model (AsymmetricBranch).
    #   In Powsybl IIDM, a MATPOWER "line" connecting buses at different nominal voltages
    #   carries large asymmetric b1/b2 shunts that encode the natural turns-ratio correction.
    #   Y_11 = Y_S·Z_base1 + j·B1,  Y_12 = −Y_S·Z_base12,  Y_22 = Y_S·Z_base2 + j·B2
    #   where Z_base1=nomV1²/S, Z_base12=nomV1·nomV2/S, Z_base2=nomV2²/S.
    for i in axes(lines_df, 1)
        from_bus  = bus_id_to_int[lines_df[i, :bus1_id]]
        to_bus    = bus_id_to_int[lines_df[i, :bus2_id]]
        nominalV1 = vl_nominalV[lines_df[i, :voltage_level1_id]]
        nominalV2 = vl_nominalV[lines_df[i, :voltage_level2_id]]
        R_SI = Float64(lines_df[i, :r])
        X_SI = Float64(lines_df[i, :x])
        if nominalV1 == nominalV2
            Z_base = nominalV1^2 / baseMVA
            R_pu = R_SI / Z_base
            X_pu = X_SI / Z_base
            B_pu = (Float64(lines_df[i, :b1]) + Float64(lines_df[i, :b2])) * Z_base
            add_two_bus_injector!(builder, from_bus, to_bus,
                Branch(T(R_pu), T(X_pu), T(B_pu), one(T), zero(T)))
        else
            Z_base12 = nominalV1 * nominalV2 / baseMVA
            Z_base1  = nominalV1^2 / baseMVA
            Z_base2  = nominalV2^2 / baseMVA
            Y_S12    = 1 / complex(R_SI / Z_base12, X_SI / Z_base12)
            TAP      = nominalV1 / nominalV2
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
    # TAP = (ratedU1/nominalV1) / (ratedU2/nominalV2)  (PowSyBl convention)
    # IIDM stores r, x, g, b in Ω/S referred to side 2; Z_base = nominalV2²/baseMVA.
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
    has_alpha = hasproperty(xfmrs_df, :alpha)
    has_g     = hasproperty(xfmrs_df, :g)
    for i in axes(xfmrs_df, 1)
        from_bus  = bus_id_to_int[xfmrs_df[i, :bus1_id]]
        to_bus    = bus_id_to_int[xfmrs_df[i, :bus2_id]]
        nominalV1 = vl_nominalV[xfmrs_df[i, :voltage_level1_id]]
        nominalV2 = vl_nominalV[xfmrs_df[i, :voltage_level2_id]]
        ratedU1   = Float64(xfmrs_df[i, :rated_u1])
        ratedU2   = Float64(xfmrs_df[i, :rated_u2])
        Z_base    = nominalV2^2 / baseMVA
        TAP       = (ratedU1 / nominalV1) / (ratedU2 / nominalV2)
        R_pu = Float64(xfmrs_df[i, :r]) / Z_base
        X_pu = Float64(xfmrs_df[i, :x]) / Z_base
        B_pu = Float64(xfmrs_df[i, :b]) * Z_base
        G_pu = (has_g ? Float64(xfmrs_df[i, :g]) : 0.0) * Z_base
        α_deg   = has_alpha ? Float64(xfmrs_df[i, :alpha]) : 0.0
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
        gens_df       = gens_df,
        loads_df      = loads_df,
        vl_nominalV   = vl_nominalV,
        bus_vl        = bus_vl,
        baseMVA       = Float64(baseMVA),
        K_DV_vec      = K_DV_vec,
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

function controls_from_solution(sys)
    gens_df      = sys.gens_df
    buses_df     = sys.buses_df
    loads_df     = sys.loads_df
    vl_nominalV  = sys.vl_nominalV
    bus_vl       = sys.bus_vl
    baseMVA      = sys.baseMVA
    K_DV_vec     = sys.K_DV_vec
    power_system = sys.power_system

    # Actual solved bus voltage magnitudes (per unit), keyed by bus id.
    # Used instead of target_v so that V_ref is consistent with the LF solution.
    bus_vm_pu = Dict{String, Float64}(
        buses_df[i, :id] =>
            Float64(buses_df[i, :v_mag]) / vl_nominalV[buses_df[i, :voltage_level_id]]
        for i in axes(buses_df, 1)
    )

    controls = zeros(Float64, power_system.n_controls)
    cursor   = 1

    for i in axes(gens_df, 1)
        # Powsybl terminal :p/:q use consumer convention (negative = production).
        # Fall back to target setpoints when LF has not been run (NaN).
        p_raw    = Float64(gens_df[i, :p])
        Pg       = (isnan(p_raw) ? Float64(gens_df[i, :target_p]) : -p_raw) / baseMVA
        q_raw    = Float64(gens_df[i, :q])
        Qg       = (isnan(q_raw) ? Float64(gens_df[i, :target_q]) : -q_raw) / baseMVA
        vm_raw   = get(bus_vm_pu, gens_df[i, :bus_id], NaN)
        nominalV = vl_nominalV[gens_df[i, :voltage_level_id]]
        Vm       = (isnan(vm_raw) ? Float64(gens_df[i, :target_v]) / nominalV : vm_raw)

        K_DV  = K_DV_vec[i]
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
        power_system, target_variables(sys), controls,
    )
end

# ── target_variables ──────────────────────────────────────────────────────────
#
# Build the variable vector [Vm_1..Vm_n; θ_branch_1..θ_branch_m] from the
# bus voltages and branch angle differences in the loaded network.
# Branch order matches build_powersystem: lines first, then transformers.

function target_variables(sys)
    buses_df      = sys.buses_df
    vl_nominalV   = sys.vl_nominalV
    bus_id_to_int = sys.bus_id_to_int
    bus_vl        = sys.bus_vl
    active_from   = sys.active_from
    active_to     = sys.active_to
    power_system  = sys.power_system

    n_buses   = power_system.n_buses
    n_lines   = power_system.n_lines
    variables = zeros(Float64, n_buses + n_lines)

    bus_angle_rad = Dict{Int, Float64}()
    for i in axes(buses_df, 1)
        bus_id   = buses_df[i, :id]
        idx      = bus_id_to_int[bus_id]
        nominalV = vl_nominalV[bus_vl[bus_id]]
        variables[idx]     = Float64(buses_df[i, :v_mag]) / nominalV
        bus_angle_rad[idx] = deg2rad(Float64(buses_df[i, :v_angle]))
    end

    for ii in eachindex(active_from)
        f = active_from[ii]
        t = active_to[ii]
        variables[n_buses + ii] = bus_angle_rad[t] - bus_angle_rad[f]
    end

    return variables
end

end # module PowsyblIOExt
