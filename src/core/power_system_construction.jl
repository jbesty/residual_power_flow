
using Graphs
using LinearAlgebra

function identify_cycles(g::SimpleGraph)
    return cycle_basis(g)
end

function get_edge_index_and_direction(
    edge::Tuple{T,T},
    edge_list::Vector{Tuple{T,T}},
) where {T<:Integer}
    positive_matches = findall(e -> e == edge, edge_list)
    negative_matches = findall(e -> e == reverse(edge), edge_list)
    if !isempty(positive_matches)
        return positive_matches[1], 1
    elseif !isempty(negative_matches)
        return negative_matches[1], -1
    else
        throw(ErrorException("Edge not found in the edge list."))
    end
end

function compute_cycles_and_participation(edge_list)

    undirected_graph = SimpleGraph(Graphs.SimpleEdge.(edge_list))
    cycles = cycle_basis(undirected_graph)

    cycle_participations = Vector{Vector{Int64}}()
    for cycle in cycles
        # Participation is indexed in the same order as `edge_list`.
        # Do not size this by `ne(undirected_graph)` because `SimpleGraph` can
        # collapse parallel/duplicate edges, which would break indexing.
        cycle_participation = zeros(Int64, length(edge_list))
        cycle_edges = [(cycle[jj], cycle[jj+1]) for jj = 1:length(cycle)-1]
        append!(cycle_edges, [(cycle[end], cycle[1])])

        for cycle_edge in cycle_edges
            edge_index, direction = get_edge_index_and_direction(cycle_edge, edge_list)
            cycle_participation[edge_index] = direction
        end
        push!(cycle_participations, cycle_participation)
    end

    # Add trivial KVL cycles for parallel branches (multiple edges between the same bus pair).
    # Each extra parallel branch adds one independent KVL equation: θ_A - θ_B = 0.
    canonical_to_indices = Dict{Tuple{Int,Int}, Vector{Int}}()
    for (idx, e) in enumerate(edge_list)
        key = minmax(e[1], e[2])
        push!(get!(canonical_to_indices, key, Int[]), idx)
    end
    for indices in values(canonical_to_indices)
        if length(indices) > 1
            first_idx = indices[1]
            first_edge = edge_list[first_idx]
            for other_idx in indices[2:end]
                other_edge = edge_list[other_idx]
                trivial = zeros(Int64, length(edge_list))
                trivial[first_idx] = 1
                # Same stored direction → opposite signs (θ_A - θ_B = 0).
                # Opposite stored direction → same signs (θ_A + θ_B = 0).
                trivial[other_idx] = first_edge == other_edge ? -1 : 1
                push!(cycle_participations, trivial)
            end
        end
    end

    return cycles, cycle_participations
end

function check_bus_reference(bus_id::Int, builder::PowerSystemBuilder)
    if bus_id > builder.n_buses || bus_id < 1
        error("Bus index $bus_id is not a valid bus.")
    end
end

function add_one_bus_injector!(
    builder::PowerSystemBuilder{T},
    terminal_bus::Int,
    component::Union{ZIPLoad,SynchronousMachineStatic},
) where {T}
    check_bus_reference(terminal_bus, builder)
    push!(builder.single_bus_injectors, InjectorRecord{T}(component, terminal_bus))
    builder.n_controls += n_controls(component)
end

function add_one_bus_injector!(
    builder::PowerSystemBuilder{T},
    terminal_bus::Int,
    component::Shunt,
) where {T}
    check_bus_reference(terminal_bus, builder)
    push!(builder.single_bus_injectors, InjectorRecord{T}(component, terminal_bus))
end

function add_two_bus_injector!(
    builder::PowerSystemBuilder{T},
    from_bus::Int,
    to_bus::Int,
    component::Union{Branch,AsymmetricBranch},
) where {T}
    check_bus_reference(from_bus, builder)
    check_bus_reference(to_bus, builder)
    if from_bus == to_bus
        error("Two bus injector cannot have the same terminal bus.")
    end
    push!(builder.branch_injectors, BranchRecord{T}(component, from_bus, to_bus))
    push!(builder.edge_list, (from_bus, to_bus))
    builder.n_lines = length(builder.branch_injectors)
end

function build!(builder::PowerSystemBuilder{T}) where {T}
    _, cycle_participations = compute_cycles_and_participation(builder.edge_list)
    n_cycles      = length(cycle_participations)
    cycle_admittances = ones(T, n_cycles)
    n_variables   = builder.n_buses + builder.n_lines

    # Signed incidence matrix: A[k, from] = -1, A[k, to] = +1
    # bus_to_branch_angle_map maps bus angles to branch angles (θ_to - θ_from).
    # branch_to_bus_angle_map is the left-inverse (pseudo-inverse) of A.
    # For a tree network, A has full row rank so A * pinv(A) = I (exact round-trip).
    A_dense = zeros(T, builder.n_lines, builder.n_buses)
    for (k, (from_bus, to_bus)) in enumerate(builder.edge_list)
        A_dense[k, from_bus] = -one(T)
        A_dense[k, to_bus] = one(T)
    end
    bus_to_branch_angle_map = sparse(A_dense)
    branch_to_bus_angle_map = Matrix{T}(pinv(A_dense))

    return PowerSystem{T}(
        builder.n_buses,
        builder.n_lines,
        n_variables,
        builder.n_controls,
        n_cycles,
        builder.single_bus_injectors,
        builder.branch_injectors,
        builder.edge_list,
        cycle_participations,
        cycle_admittances,
        branch_to_bus_angle_map,
        bus_to_branch_angle_map,
    )
end

