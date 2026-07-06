export SamplerStrategy, BalancedStrategy, FeasibleStrategy
export AbstractSamplerConfig, SamplerDiagnostics
export sampler_strategy, sampler_id, strategy_name
export generate_dataset_with_diagnostics

# ── Sampler-strategy seam ─────────────────────────────────────────────────────
#
# A dataset run is (grid, strategy, config). The strategy is selected by a name
# (the `[sampler] strategy` TOML key) which resolves to a `SamplerStrategy`
# singleton; the per-strategy knobs live in an `AbstractSamplerConfig` subtype.
# Each strategy implements the single seam method
#
#     _generate(::Strategy, ps, cfg, n; rng) -> (PowerFlowDataset, SamplerDiagnostics)
#
# `generate_dataset` (the public, back-compatible entry) returns the bare dataset;
# `generate_dataset_with_diagnostics` returns the pair. The diagnostics channel is
# present now so the dataset-generation-robustness rebase becomes a change *inside*
# a strategy (filling diagnostics content) rather than another interface re-cut.

# Per-strategy sampler knob block. Concrete subtypes live alongside each strategy
# (SamplerConfig = balanced, ImportanceSamplerConfig, PerturbationSamplerConfig).
abstract type AbstractSamplerConfig end

# Strategy singletons. The registry below maps the TOML name to the type.
abstract type SamplerStrategy end
struct BalancedStrategy <: SamplerStrategy end
struct FeasibleStrategy <: SamplerStrategy end

const _SAMPLER_STRATEGIES = Dict{String, DataType}(
    "balanced" => BalancedStrategy,
    "feasible" => FeasibleStrategy,
)

# Resolve a TOML `strategy` name to its singleton. Total: an unknown name raises a
# clear ArgumentError naming the unknown strategy and listing the registered ones.
function sampler_strategy(name::AbstractString)
    haskey(_SAMPLER_STRATEGIES, name) || throw(ArgumentError(
        "unknown sampler strategy $(repr(name)); registered strategies are " *
        join(sort(collect(keys(_SAMPLER_STRATEGIES))), ", ")))
    return _SAMPLER_STRATEGIES[name]()
end

# Reverse map (strategy -> TOML name), used in the provenance digest.
strategy_name(::BalancedStrategy) = "balanced"
strategy_name(::FeasibleStrategy) = "feasible"

# Per-strategy algorithm tag stamped into the artifact (freshness key). Bump the
# version suffix when a change to that strategy's file alters generated values for
# a fixed seed. `SAMPLER_ID` (config.jl) is kept as the balanced tag for back-compat.
sampler_id(::BalancedStrategy) = "balanced-oc/v1"
sampler_id(::FeasibleStrategy) = "feasible/v1"

# ── Diagnostics return channel ────────────────────────────────────────────────
#
# Minimal, strategy-agnostic container. `capacity` is NaN where a strategy has no
# calibrated capacity; `extra` carries strategy-specific diagnostics (e.g. ESS,
# per-σ accept rates). The dataset-generation-robustness spec fills more of this
# (yield, accept-tag tally) without changing the seam signature.
struct SamplerDiagnostics
    capacity :: Float64
    attempts :: Int
    accepted :: Int
    extra    :: Dict{String, Any}
end

SamplerDiagnostics(; capacity::Real = NaN, attempts::Integer = 0,
                   accepted::Integer = 0, extra::AbstractDict = Dict{String,Any}()) =
    SamplerDiagnostics(Float64(capacity), Int(attempts), Int(accepted),
                       Dict{String,Any}(string(k) => v for (k, v) in extra))

# ── Public, strategy-dispatched entries ───────────────────────────────────────
#
# `_strategy_for(cfg)` maps a config type to its strategy; the per-config methods
# live alongside each strategy. `_generate` is the seam each strategy implements.

generate_dataset_with_diagnostics(
    power_system::PowerSystem, cfg::AbstractSamplerConfig, n::Int;
    rng::AbstractRNG = default_rng(),
) = _generate(_strategy_for(cfg), power_system, cfg, n; rng)

generate_dataset(
    power_system::PowerSystem, cfg::AbstractSamplerConfig, n::Int;
    rng::AbstractRNG = default_rng(),
) = first(_generate(_strategy_for(cfg), power_system, cfg, n; rng))
