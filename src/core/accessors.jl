export get_flat_start
export convert_branch_angles_to_bus_angles
export convert_bus_angles_to_branch_angles
export voltage_magnitudes, branch_angles

function get_flat_start(power_system::PowerSystem)
    T = typeof(power_system).parameters[1]
    n_buses = power_system.n_buses
    n_lines = power_system.n_lines
    voltage_start = vcat(ones(T, n_buses), zeros(T, n_lines))
    return voltage_start
end

function convert_branch_angles_to_bus_angles(
    power_system::PowerSystem,
    variables::AbstractVector{<:Real};
    exact::Bool = true,
)

    branch_angles = variables[power_system.n_buses+1:end]
    bus_angles = power_system.branch_to_bus_angle_map * branch_angles

    branch_angles_backwards = power_system.bus_to_branch_angle_map * bus_angles
    if exact && !(branch_angles_backwards ≈ branch_angles)
        error("Branch/bus angle conversion mismatch.")
    end

    return bus_angles
end

function convert_bus_angles_to_branch_angles(
    power_system::PowerSystem,
    bus_angles::AbstractVector{<:Real},
)
    branch_angles = power_system.bus_to_branch_angle_map * bus_angles
    return branch_angles
end

# ── voltage_magnitudes / branch_angles: named views into state.voltages ──────

voltage_magnitudes(state::PowerFlowState) =
    view(state.voltages, 1:state.power_system.n_buses)

branch_angles(state::PowerFlowState) =
    view(state.voltages, state.power_system.n_buses+1:length(state.voltages))

# ── control_indices: query global control-vector indices by component type and slot ──

_slot_offset(::Type{SynchronousMachineStatic}, slot::Symbol) =
    slot === :P    ? 0 :
    slot === :Vref ? 1 :
    throw(ArgumentError("SynchronousMachineStatic has no slot :$slot (valid: :P, :Vref)"))

_slot_offset(::Type{ZIPLoad}, slot::Symbol) =
    slot === :P ? 0 :
    slot === :Q ? 1 :
    throw(ArgumentError("ZIPLoad has no slot :$slot (valid: :P, :Q)"))

function control_indices(power_system::PowerSystem, ComponentType::Type, slot::Symbol)
    offset = _slot_offset(ComponentType, slot)
    indices = Int[]
    control_cursor = 1
    for injector in power_system.single_bus_injectors
        n_component_controls = n_controls(injector.component)
        if injector.component isa ComponentType && n_component_controls > offset
            push!(indices, control_cursor + offset)
        end
        control_cursor += n_component_controls
    end
    return indices
end
