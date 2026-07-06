using SparseArrays

export Component
export SynchronousMachineStatic
export ZIPLoad, ConstantPowerLoad, ConstantCurrentLoad, ConstantImpedanceLoad
export Shunt
export Branch, Line, TransformerSimple, AsymmetricBranch
export InjectorRecord, BranchRecord
export PowerSystem
export PowerFlowState
const ALLOWED_TYPES = Union{Float64,Float32}

struct SynchronousMachineStatic{T}
    K_DV::T

    function SynchronousMachineStatic(K_DV::T) where {T<:ALLOWED_TYPES}
        return new{T}(K_DV)
    end
end

struct ZIPLoad{T}
    # ZIPLoad: S = a + b (V/V_0) + c (V/V_0)^2
    V_0::T
    a::T
    b::T
    c::T
end

function ConstantPowerLoad(V_0::T) where {T<:ALLOWED_TYPES}
    return ZIPLoad(V_0, one(T), zero(T), zero(T))
end

function ConstantCurrentLoad(V_0::T) where {T<:ALLOWED_TYPES}
    return ZIPLoad(V_0, zero(T), one(T), zero(T))
end

function ConstantImpedanceLoad(V_0::T) where {T<:ALLOWED_TYPES}
    return ZIPLoad(V_0, zero(T), zero(T), one(T))
end

struct Shunt{T}
    G::T # shunt conductance
    B::T # shunt susceptance

    function Shunt(G::T, B::T) where {T<:ALLOWED_TYPES}
        return new{T}(G, B)
    end
end

struct Branch{T}
    R::T # line resistance
    X::T # line reactance
    B::T # line charging susceptance
    TAP::T # transformer tap ratio
    θ_shift::T  # phase shift angle in radians
    Z_S::Complex{T} # line impedance
    Y_S::Complex{T} # line admittance
    Y_11::Complex{T} # Element in Ybus matrix
    Y_12::Complex{T} # Element in Ybus matrix
    Y_21::Complex{T}
    Y_22::Complex{T}

    function Branch(R::T, X::T, B::T, TAP::T, θ_shift::T) where {T<:ALLOWED_TYPES}
        (R == zero(T) && X == zero(T)) && throw(ArgumentError(
            "Branch R=0, X=0 is a zero-impedance element; use a bus merge instead."))
        Z_S = R + X * im
        Y_S = 1 / Z_S
        Y_11 = (Y_S + B / 2 * im) / (TAP^2)
        Y_12 = -Y_S * (1 / TAP) * (1 / exp(-im * θ_shift))
        Y_21 = -Y_S * (1 / TAP) * (1 / exp(im * θ_shift))
        Y_22 = (Y_S + B / 2 * im)

        new{T}(R, X, B, TAP, θ_shift, Z_S, Y_S, Y_11, Y_12, Y_21, Y_22)
    end
end

function Line(R::T, X::T, B::T) where {T<:ALLOWED_TYPES}
    return Branch(R, X, B, one(T), zero(T))
end

function TransformerSimple(X::T) where {T<:ALLOWED_TYPES}
    return Branch(zero(T), X, zero(T), one(T), zero(T))
end

# Asymmetric pi-element for cross-voltage-level lines.
# Stores the pre-computed admittance matrix entries directly, because Y_11 ≠ Y_22
# when the two bus bases differ (cross-VL lines in Powsybl IIDM).
# Derived as:
#   Y_11 = Y_S12·TAP + (G1 + j·B1)  (TAP = nomV1/nomV2, Z_base1 = nomV1²/S,
#                                      G1 = g1_SI·Z_base1, B1 = b1_SI·Z_base1)
#   Y_12 = Y_21 = −Y_S12  (Y_S12 = Z_base12/Z_actual, Z_base12 = nomV1·nomV2/S)
#   Y_22 = Y_S12/TAP + (G2 + j·B2)  (Z_base2 = nomV2²/S,
#                                      G2 = g2_SI·Z_base2, B2 = b2_SI·Z_base2)
# The g1/g2 shunt conductances stored in Powsybl IIDM encode the turns-ratio
# correction so that Re(Y_11 + Y_12) ≈ 0 (zero active injection at flat start).
struct AsymmetricBranch{T}
    Y_11::Complex{T}
    Y_12::Complex{T}
    Y_21::Complex{T}
    Y_22::Complex{T}
end

# ── Component union ───────────────────────────────────────────────────────────

const Component{T} = Union{SynchronousMachineStatic{T}, ZIPLoad{T}, Shunt{T}, Branch{T}, AsymmetricBranch{T}}

# ── Component records ─────────────────────────────────────────────────────────
# Replace parallel arrays with cohesive per-component structs.

struct InjectorRecord{T<:ALLOWED_TYPES}
    component::Union{SynchronousMachineStatic{T}, ZIPLoad{T}, Shunt{T}}
    bus_id::Int
end

struct BranchRecord{T<:ALLOWED_TYPES}
    component::Union{Branch{T}, AsymmetricBranch{T}}
    from_bus::Int
    to_bus::Int
end

# ── PowerSystem (immutable) ───────────────────────────────────────────────────

struct PowerSystem{T<:ALLOWED_TYPES}
    n_buses::Int
    n_lines::Int
    n_variables::Int
    n_controls::Int
    n_cycles::Int

    single_bus_injectors::Vector{InjectorRecord{T}}
    branch_injectors::Vector{BranchRecord{T}}

    edge_list::Vector{Tuple{Int,Int}}
    cycle_participation::Vector{Vector{Int}}
    cycle_admittances::Vector{T}

    branch_to_bus_angle_map::Matrix{T}
    bus_to_branch_angle_map::SparseArrays.SparseMatrixCSC{T, Int}
end

# ── PowerSystemBuilder (mutable, used for incremental construction) ───────────

mutable struct PowerSystemBuilder{T<:ALLOWED_TYPES}
    n_buses::Int
    n_controls::Int
    n_lines::Int
    single_bus_injectors::Vector{InjectorRecord{T}}
    branch_injectors::Vector{BranchRecord{T}}
    edge_list::Vector{Tuple{Int,Int}}
end

function PowerSystemBuilder{T}(n_buses::Int) where {T<:ALLOWED_TYPES}
    return PowerSystemBuilder{T}(
        n_buses, 0, 0,
        InjectorRecord{T}[],
        BranchRecord{T}[],
        Tuple{Int,Int}[],
    )
end

# ── PowerFlowState ────────────────────────────────────────────────────────────

struct PowerFlowState{T<:ALLOWED_TYPES}
    power_system        :: PowerSystem{T}
    voltages            :: Vector{T}
    controls            :: Vector{T}
    statuses            :: Vector{T}
    # Per-component current injections, populated as a side effect of
    # compute_residual(state::PowerFlowState).
    # single_bus_currents: length 2·n_single  ([i_d, i_q] per injector)
    # branch_currents:     length 4·n_branches ([i_d,i_q] from-end then [i_d,i_q] to-end per branch)
    single_bus_currents :: Vector{T}
    branch_currents     :: Vector{T}
    # Pre-allocated workspace buffers for in-place solver operations.
    # residual_buffer:          length 2·n_buses + n_cycles (worst-case)
    # current_balance_buffer:   length 2·n_buses
    # update_buffer:            length n_variables (= n_buses + n_lines)
    # controls_update_buffer:   length n_controls
    residual_buffer          :: Vector{T}
    current_balance_buffer   :: Vector{T}
    update_buffer            :: Vector{T}
    controls_update_buffer   :: Vector{T}
    # Topology-aware cycle data, updated by update_topology!.
    # Initialised from PowerSystem at construction (all branches active).
    active_cycle_participation :: Vector{Vector{Int}}
    active_cycle_admittances   :: Vector{T}
    n_cycles_active            :: Base.RefValue{Int}
    _topology_snapshot         :: Vector{Bool}
    # Cached sparsity patterns and Jacobian buffers (computed lazily, invalidated by update_topology!).
    _voltage_sparsity_cache    :: Base.RefValue{Union{Nothing, SparseMatrixCSC{Bool, Int}}}
    _control_sparsity_cache    :: Base.RefValue{Union{Nothing, SparseMatrixCSC{Bool, Int}}}
    _voltage_jacobian_cache    :: Base.RefValue{Union{Nothing, SparseMatrixCSC{T, Int}}}
    _control_jacobian_cache    :: Base.RefValue{Union{Nothing, SparseMatrixCSC{T, Int}}}
end

# 4-arg constructor: statuses provided, current vectors initialised to zeros.
function PowerFlowState(
    power_system::PowerSystem{T},
    voltages::AbstractVector,
    controls::AbstractVector,
    statuses::AbstractVector,
) where {T<:ALLOWED_TYPES}
    n_single   = length(power_system.single_bus_injectors)
    n_branches = length(power_system.branch_injectors)
    n_buses    = power_system.n_buses
    n_cycles   = power_system.n_cycles
    n_variables = power_system.n_variables
    n_controls = power_system.n_controls
    return PowerFlowState(
        power_system,
        voltages isa Vector{T} ? voltages : Vector{T}(voltages),
        controls isa Vector{T} ? controls : Vector{T}(controls),
        statuses isa Vector{T} ? statuses : Vector{T}(statuses),
        zeros(T, 2 * n_single),
        zeros(T, 4 * n_branches),
        zeros(T, 2 * n_buses + power_system.n_lines),
        zeros(T, 2 * n_buses),
        zeros(T, n_variables),
        zeros(T, n_controls),
        [copy(p) for p in power_system.cycle_participation],
        copy(power_system.cycle_admittances),
        Ref(n_cycles),
        fill(true, n_branches),
        Ref{Union{Nothing, SparseMatrixCSC{Bool, Int}}}(nothing),
        Ref{Union{Nothing, SparseMatrixCSC{Bool, Int}}}(nothing),
        Ref{Union{Nothing, SparseMatrixCSC{T, Int}}}(nothing),
        Ref{Union{Nothing, SparseMatrixCSC{T, Int}}}(nothing),
    )
end

# 3-arg constructor: default all-ones statuses.
function PowerFlowState(
    power_system::PowerSystem{T},
    voltages::AbstractVector,
    controls::AbstractVector,
) where {T<:ALLOWED_TYPES}
    n = length(power_system.single_bus_injectors) + power_system.n_lines
    return PowerFlowState(power_system, voltages, controls, ones(T, n))
end

# 2-arg constructor: flat start for voltages.
function PowerFlowState(power_system::PowerSystem{T}, controls::AbstractVector) where {T<:ALLOWED_TYPES}
    voltages = vcat(ones(T, power_system.n_buses), zeros(T, power_system.n_lines))
    return PowerFlowState(power_system, voltages, controls)
end


