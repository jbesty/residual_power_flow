module JLD2Ext

using JLD2
using ResidualPowerFlow
import ResidualPowerFlow: save_dataset, load_dataset, save_solver, load_solver
import ResidualPowerFlow: dataset_is_current, save_config_dataset, load_train_dataset, load_test_dataset
import ResidualPowerFlow: load_or_generate_datasets, generate_dataset
import ResidualPowerFlow: PowerFlowDataset, PowerFlowSolver, PowerFlowState, PowerSystem, DatasetConfig
import ResidualPowerFlow: NeuralSolver, _solver_to_portable, _solver_from_portable
import ResidualPowerFlow: ALLOWED_TYPES, dataset_path, _config_hash
import ResidualPowerFlow: voltages_matrix, controls_matrix, converged, feasible, stationary
import ResidualPowerFlow: compute_current_balance
import ResidualPowerFlow: SCHEMA_VERSION, SAMPLER_ID, sampler_id, rpf_version, _git_sha, config_digest
import ResidualPowerFlow: DatasetSignature, signature

using Dates
using Random: MersenneTwister

# ── Provenance / schema metadata ───────────────────────────────────────────────
#
# Every written dataset carries a `metadata/` block: a schema version, package
# version + git SHA, the sampler algorithm tag, a config digest (resolved
# semantics, not bytes), an optional system signature, and reserved keys the
# generation-robustness follow-up will populate. `_provenance_metadata` builds
# the always-present part; callers merge in signature / config-specific keys.

function _provenance_metadata()
    return Dict{String,Any}(
        "schema_version" => SCHEMA_VERSION,
        "rpf_version"    => rpf_version(),
        "git_sha"        => _git_sha(),
        "sampler_id"     => SAMPLER_ID,
        "generated_at"   => string(now()),
        "config_digest"  => "unknown",   # overridden by the config-aware writer
        "has_signature"  => false,       # overridden when a signature is written
        # Reserved for the generation-robustness follow-up spec (calibrate-once,
        # capacity sharing, accept-tag yield). Stamped now so that spec fills
        # values, not schema.
        "capacity"       => "unknown",
        "load_range"     => "unknown",
        "yield"          => "unknown",
    )
end

function _signature_metadata(sig::DatasetSignature)
    return Dict{String,Any}(
        "has_signature"          => true,
        "signature/system_name"  => sig.system_name,
        "signature/n_controls"   => sig.n_controls,
        "signature/n_variables"  => sig.n_variables,
    )
end

function _write_metadata!(f, md::AbstractDict)
    for (k, v) in md
        f["metadata/$(k)"] = v
    end
end

# Both flags are stored per split: `feasible` (‖r‖≈0 energy criterion, also
# written as `converged` for back-compatibility) and `stationary` (the GN
# acceptance gate). Recomputed from the states at write time.
function _write_split!(f, split::AbstractString, ds::PowerFlowDataset)
    fe = Vector{Bool}(feasible(ds))
    f["$(split)/controls"]   = controls_matrix(ds)
    f["$(split)/voltages"]   = voltages_matrix(ds)
    f["$(split)/feasible"]   = fe
    f["$(split)/stationary"] = Vector{Bool}(stationary(ds))
    f["$(split)/converged"]  = fe
end

# ── Dataset I/O ──────────────────────────────────────────────────────────────

function save_dataset(path::AbstractString, dataset::PowerFlowDataset; split::Symbol = :train)
    sig = DatasetSignature("unknown",
                           dataset.power_system.n_controls,
                           dataset.power_system.n_variables)
    md = merge(_provenance_metadata(), _signature_metadata(sig))
    jldopen(path, "w") do f
        _write_metadata!(f, md)
        _write_split!(f, string(split), dataset)
    end
end

function save_dataset(
    path::AbstractString,
    train::PowerFlowDataset,
    test::PowerFlowDataset;
    metadata::Dict = Dict(),
)
    md = merge(_provenance_metadata(), Dict{String,Any}(string(k) => v for (k, v) in metadata))
    jldopen(path, "w") do f
        _write_metadata!(f, md)
        _write_split!(f, "train", train)
        _write_split!(f, "test", test)
    end
end

# Validate a file's provenance before trusting its contents. Three outcomes:
# a legacy file (no schema_version) loads with an unverified-provenance warning;
# an over-version file is a hard error naming both versions; a stamped file with
# a signature has its control/variable counts checked against `ps`.
function _check_provenance(f, path::AbstractString, ps::PowerSystem)
    schema = get(f, "metadata/schema_version", 0)
    if schema == 0
        @warn "Dataset at $(path) has no schema_version (legacy file): unverified " *
              "provenance, signature check skipped."
        return
    end
    schema > SCHEMA_VERSION && error(
        "Dataset at $(path) has schema_version $(schema) but this ResidualPowerFlow " *
        "build knows only up to $(SCHEMA_VERSION). Upgrade the package to load it.")

    get(f, "metadata/has_signature", false) || begin
        @warn "Dataset at $(path) carries no signature: skipping signature check " *
              "(system match unverified)."
        return
    end

    sys_name = get(f, "metadata/signature/system_name", "unknown")
    n_ctrl   = get(f, "metadata/signature/n_controls", -1)
    n_var    = get(f, "metadata/signature/n_variables", -1)
    n_ctrl == ps.n_controls || error(
        "Dataset/system signature mismatch at $(path): stored n_controls=$(n_ctrl) " *
        "but the supplied PowerSystem has $(ps.n_controls) (stored system_name=$(sys_name)).")
    n_var == ps.n_variables || error(
        "Dataset/system signature mismatch at $(path): stored n_variables=$(n_var) " *
        "but the supplied PowerSystem has $(ps.n_variables) (stored system_name=$(sys_name)).")
    return
end

function load_dataset(path::AbstractString, split::Symbol, power_system::PowerSystem{T}) where {T<:ALLOWED_TYPES}
    key = string(split)
    jldopen(path, "r") do f
        _check_provenance(f, path, power_system)
        controls  = Matrix{T}(f["$(key)/controls"])
        voltages  = Matrix{T}(f["$(key)/voltages"])
        N = size(controls, 1)
        states = [PowerFlowState(power_system, voltages[i, :], controls[i, :]) for i in 1:N]
        for s in states
            compute_current_balance(s)
        end
        return PowerFlowDataset(power_system, states)
    end
end

# ── Solver persistence ───────────────────────────────────────────────────────

# NeuralSolver: write the portable form (architecture + params + transformation),
# rebuilt from scratch on load (no live Lux Chain serialised). See LuxExt.
function save_solver(path::AbstractString, solver::NeuralSolver)
    JLD2.save(path, "portable", _solver_to_portable(solver))
    return path
end

# Other solvers (Linear/Quadratic surrogates — plain matrices) serialise directly.
function save_solver(path::AbstractString, solver::PowerFlowSolver)
    JLD2.save(path, "solver", solver)
    return path
end

# Format-aware: a "portable" key is a rebuilt NeuralSolver; "solver" is a directly
# serialised surrogate solver.
function load_solver(path::AbstractString)
    data = JLD2.load(path)
    haskey(data, "portable") && return _solver_from_portable(data["portable"])
    return data["solver"]
end

# ── Config-aware dataset I/O ─────────────────────────────────────────────────

# Freshness keys on resolved semantics + code version, never raw bytes. Current
# iff the stored config_digest matches the resolved config AND the stored
# rpf_version / sampler_id match the running ones. A digest match with a
# version/sampler mismatch returns false AND warns, so the caller regenerates
# deliberately rather than loading silently-drifted data.
function dataset_is_current(config::DatasetConfig, config_path::AbstractString)
    path = dataset_path(config)
    isfile(path) || return false
    jldopen(path, "r") do f
        stored_digest  = get(f, "metadata/config_digest", "")
        stored_digest == config_digest(config) || return false

        stored_version = get(f, "metadata/rpf_version", "")
        stored_sampler = get(f, "metadata/sampler_id", "")
        running_sampler = sampler_id(config)
        if stored_version != rpf_version() || stored_sampler != running_sampler
            @warn "Dataset at $(path) matches the resolved config but was generated " *
                  "with a different RPF version / sampler — regenerate deliberately." *
                  " stored rpf_version=$(stored_version) running=$(rpf_version());" *
                  " stored sampler_id=$(stored_sampler) running=$(running_sampler)."
            return false
        end
        return true
    end
end

function save_config_dataset(
    config::DatasetConfig,
    config_path::AbstractString,
    train::PowerFlowDataset,
    test::PowerFlowDataset,
)
    mkpath(config.output_dir)
    sig = DatasetSignature(config.system_name,
                           train.power_system.n_controls,
                           train.power_system.n_variables)
    metadata = merge(
        Dict{String,Any}(
            "config_name"   => config.name,
            "config_hash"   => _config_hash(config_path),   # legacy byte hash, provenance only
            "config_digest" => config_digest(config),
            "sampler_id"    => sampler_id(config),           # per-strategy tag (overrides default)
            "n_train"       => config.n_train,
            "n_test"        => config.n_test,
            "seed_train"    => config.seed_train,
            "seed_test"     => config.seed_test,
            "slack_indices" => config.slack_indices,
        ),
        _signature_metadata(sig),
    )
    save_dataset(dataset_path(config), train, test; metadata)
end

function load_train_dataset(config::DatasetConfig, power_system::PowerSystem)
    load_dataset(dataset_path(config), :train, power_system)
end

function load_test_dataset(config::DatasetConfig, power_system::PowerSystem)
    load_dataset(dataset_path(config), :test, power_system)
end

function load_or_generate_datasets(
    config::DatasetConfig,
    power_system::PowerSystem,
    config_path::AbstractString;
    force::Bool = false,
)
    if !force && dataset_is_current(config, config_path)
        println("Dataset up-to-date at $(dataset_path(config)), loading…")
        train = load_train_dataset(config, power_system)
        test  = load_test_dataset(config, power_system)
        return (; train, test)
    end

    println("Generating dataset ($(config.n_train) train, $(config.n_test) test)…")
    seed_train_rng = MersenneTwister(config.seed_train)
    seed_test_rng  = MersenneTwister(config.seed_test)

    train = generate_dataset(power_system, config.sampler, config.n_train;
                             rng = seed_train_rng)
    test  = generate_dataset(power_system, config.sampler, config.n_test;
                             rng = seed_test_rng)

    save_config_dataset(config, config_path, train, test)
    println("Saved to $(dataset_path(config))")
    return (; train, test)
end

end # module
