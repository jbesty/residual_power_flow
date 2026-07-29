module LuxExt

using Lux
using LinearAlgebra: I
using Random: AbstractRNG, default_rng
import ComponentArrays: ComponentArray

using ResidualPowerFlow
import ResidualPowerFlow: NeuralSolver, DataTransformation, PowerSystem,
    _solver_to_portable, _solver_from_portable

# MLPArchitecture is a Lux-adjacent architecture-config type — it has no Lux
# dep itself but is only used by the Lux-backed NeuralSolver constructor
# below, so it lives in this extension. Access from user code:
#     using ResidualPowerFlow.LuxExt: MLPArchitecture
# (requires Lux to be loaded).
Base.@kwdef struct MLPArchitecture
    n_hidden_layers::Int = 2
    n_neurons::Int       = 64
end

# Identity DataTransformation used as the default when the NeuralSolver is
# constructed without an explicit transformation. Defined here because the
# only caller is the constructor below.
function _identity_transformation(power_system::PowerSystem, ::Type{T} = Float64) where {T<:AbstractFloat}
    n_u = power_system.n_controls
    n_v = power_system.n_variables
    return DataTransformation(
        Matrix{T}(I, n_u, n_u), zeros(T, n_u),
        Matrix{T}(I, n_v, n_v), zeros(T, n_v),
    )
end

# ── Weight initialisation ─────────────────────────────────────────────────────

_small_uniform(rng, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* 0.2f0

_weight_init_fn(init::Symbol) = init == :glorot_uniform ? glorot_uniform   :
                                 init == :glorot_normal  ? glorot_normal   :
                                 init == :he_uniform     ? kaiming_uniform :
                                 init == :small_uniform  ? _small_uniform  :
                                 init == :data_driven    ? glorot_uniform  :
                                 throw(ArgumentError("Unknown init :$init. Valid: :data_driven, :glorot_uniform, :glorot_normal, :he_uniform, :small_uniform"))

# ── NeuralSolver constructor ──────────────────────────────────────────────────
#
# Builds an untrained NeuralSolver from a DataTransformation.
# The DataTransformation stores affine maps (A, b) for controls and voltages.
# normalize_* / denormalize_* handle the full physical ↔ non-dimensional round-trip.

function NeuralSolver(
    transformation::DataTransformation;
    n_hidden_layers::Int = 1,
    n_neurons::Int       = 100,
    use_layer_norm::Bool = false,
    init::Symbol         = :data_driven,
    rng::AbstractRNG     = default_rng(),
)
    n_u = size(transformation.A_controls, 1)
    n_v = size(transformation.A_voltages, 1)

    weight_init = _weight_init_fn(init)

    layers = []
    if use_layer_norm
        push!(layers, Dense(n_u, n_neurons; init_weight = weight_init))
        for _ in 1:n_hidden_layers
            # dims=1 normalises per-sample across the feature dimension.
            # The default dims=Colon() mixes the batch dimension into the
            # statistics, which makes batch and single-sample calls produce
            # different outputs — a silent inference-time regression.
            push!(layers, LayerNorm((n_neurons,), tanh; dims = 1))
            push!(layers, Dense(n_neurons, n_neurons; init_weight = weight_init))
        end
        pop!(layers)
        push!(layers, Dense(n_neurons, n_v; init_weight = weight_init))
    else
        push!(layers, Dense(n_u, n_neurons, tanh; init_weight = weight_init))
        for _ in 1:n_hidden_layers - 1
            push!(layers, Dense(n_neurons, n_neurons, tanh; init_weight = weight_init))
        end
        push!(layers, Dense(n_neurons, n_v; init_weight = weight_init))
    end
    model = Chain(layers...)

    ps, st = Lux.setup(rng, model)
    ps = Lux.f64(ps)

    return NeuralSolver(model, ComponentArray(ps), st, transformation, Base.RefValue{Any}(nothing))
end

# ── NeuralSolver(power_system; architecture, transformation) ─────────────────
#
# Convenience constructor using config types. Pass a fitted DataTransformation
# (from `fit_data_transformation`); if omitted, the transformation defaults to
# identity and the caller is expected to train in physical units.

function NeuralSolver(
    power_system::PowerSystem;
    architecture::MLPArchitecture = MLPArchitecture(),
    transformation = nothing,
    rng::AbstractRNG = default_rng(),
)
    T = Float64
    data_trans = transformation isa DataTransformation ? transformation :
                 _identity_transformation(power_system, T)

    n_u = power_system.n_controls
    n_v = power_system.n_variables

    layers = Any[Dense(n_u, architecture.n_neurons, tanh; init_weight = glorot_uniform)]
    for _ in 1:(architecture.n_hidden_layers - 1)
        push!(layers, Dense(architecture.n_neurons, architecture.n_neurons, tanh; init_weight = glorot_uniform))
    end
    push!(layers, Dense(architecture.n_neurons, n_v; init_weight = glorot_uniform))
    model = Chain(layers...)

    ps, st = Lux.setup(rng, model)
    ps = Lux.f64(ps)

    return NeuralSolver(model, ComponentArray(ps), st, data_trans, Base.RefValue{Any}(nothing))
end

# ── Portable serialization (Part E) ───────────────────────────────────────────
#
# A NeuralSolver is saved as a plain NamedTuple of Base types — architecture
# (n_hidden_layers / n_neurons / use_layer_norm / n_u / n_v), the flat (best-
# validation) parameters, and the DataTransformation's affine maps — NOT the live
# Lux Chain. load rebuilds the model from the architecture and copies the params
# in, so a published model reloads and predicts identically without the original
# training session and is robust across Lux versions. Layout/dims are read from
# the parameter ComponentArray (Dense ⇒ :weight, LayerNorm ⇒ :scale), avoiding any
# dependence on Lux struct field names.

function _solver_to_portable(solver::NeuralSolver)
    ps = solver.parameters
    ks = collect(keys(ps))
    isdense(k) = :weight in keys(ps[k])
    isln(k)    = :scale  in keys(ps[k])
    dense_ks = filter(isdense, ks)
    W_first  = ps[dense_ks[1]].weight
    W_last   = ps[dense_ks[end]].weight
    use_layer_norm = any(isln, ks)
    n_u, n_neurons, n_v = size(W_first, 2), size(W_first, 1), size(W_last, 1)
    n_hidden_layers = use_layer_norm ? count(isln, ks) : (length(dense_ks) - 1)
    transformation = solver.transformation
    return (format          = "neural-mlp/v1",
            n_hidden_layers  = n_hidden_layers,
            n_neurons        = n_neurons,
            use_layer_norm   = use_layer_norm,
            n_u              = n_u,
            n_v              = n_v,
            params           = Vector{Float64}(vec(solver.parameters)),
            A_controls       = Matrix{Float64}(transformation.A_controls),
            b_controls       = Vector{Float64}(transformation.b_controls),
            A_voltages       = Matrix{Float64}(transformation.A_voltages),
            b_voltages       = Vector{Float64}(transformation.b_voltages))
end

function _solver_from_portable(nt::NamedTuple)
    nt.format == "neural-mlp/v1" ||
        error("unsupported saved solver format $(repr(nt.format))")
    transformation = DataTransformation(nt.A_controls, nt.b_controls, nt.A_voltages, nt.b_voltages)
    solver = NeuralSolver(transformation;
        n_hidden_layers = nt.n_hidden_layers,
        n_neurons       = nt.n_neurons,
        use_layer_norm  = nt.use_layer_norm,
        init            = :glorot_uniform)
    length(solver.parameters) == length(nt.params) ||
        error("saved parameter length $(length(nt.params)) ≠ rebuilt architecture " *
              "$(length(solver.parameters)) — architecture mismatch")
    copyto!(solver.parameters, nt.params)
    return solver
end

end # module
