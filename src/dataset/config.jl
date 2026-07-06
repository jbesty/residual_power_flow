
export DatasetConfig, DatasetSignature
export load_dataset_config, build_powersystem
export dataset_path, signature
export SCHEMA_VERSION, SAMPLER_ID, config_digest, rpf_version

using TOML
using SHA
using Dates

# ── DatasetConfig ──────────────────────────────────────────────────────────────
#
# Parsed from a TOML file. Fully describes how to reproduce a dataset.
# The config name is derived from the filename (without extension).

struct DatasetConfig
    name          :: String           # derived from filename, e.g. "case9_standard"
    system_type   :: Symbol           # :matpower (only supported type currently)
    system_name   :: String           # "case9", "case14", etc.
    K_DV          :: Union{Float64, Vector{Float64}}  # RPF voltage droop; scalar broadcasts to all generators
    sampler       :: AbstractSamplerConfig  # per-strategy operating-condition sampler knobs
    n_train       :: Int
    n_test        :: Int
    seed_train    :: Int
    seed_test     :: Int
    slack_indices :: Vector{Int}      # generator control indices designated as slack; empty = infeasible (no PtO)
    output_dir    :: String           # absolute path, resolved relative to config file
end

# ── DatasetSignature ───────────────────────────────────────────────────────────

struct DatasetSignature
    system_name :: String
    n_controls  :: Int
    n_variables :: Int
end

# ── Config I/O ────────────────────────────────────────────────────────────────

# Build the sampler config from a [sampler] TOML table, dispatched on the resolved
# strategy. Every key is optional and falls back to that strategy's default; an
# absent [sampler] block (strategy defaulting to "balanced") yields all defaults.
# The `strategy` key itself is consumed by the resolver, not by the field parsers.
function _sampler_from_toml(::BalancedStrategy, s::AbstractDict)
    g(k, d) = haskey(s, k) ? s[k] : d
    return SamplerConfig(
        qp_min        = Float64(g("qp_min", -0.3)),
        qp_max        = Float64(g("qp_max", 0.4)),
        vtarget_min   = Float64(g("vtarget_min", 1.03)),
        vtarget_max   = Float64(g("vtarget_max", 1.05)),
        offset_p      = Float64(g("offset_p", 0.04)),
        offset_q      = Float64(g("offset_q", 0.02)),
        tau_p_rel     = Float64(g("tau_p_rel", 0.03)),
        v_min         = Float64(g("v_min", 0.85)),
        v_max         = Float64(g("v_max", 1.15)),
        k_prescreen   = Int(g("k_prescreen", 8)),
        v_prescreen   = Float64(g("v_prescreen", 0.60)),
        max_iters     = Int(g("max_iters", 100)),
        load_max_frac = Float64(g("load_max_frac", 1.0)),
        load_min_frac = Float64(g("load_min_frac", 0.4)),
        cap_target    = Float64(g("cap_target", 0.6)),
        cap_samples   = Int(g("cap_samples", 120)),
        dispatchable  = Vector{Bool}(g("dispatchable", Bool[])),
        max_attempts_mult = Int(g("max_attempts_mult", 60)),
        close_slack   = Bool(g("close_slack", false)),
        tau_feasible  = Float64(g("tau_feasible", 1.0e-8)),
    )
end

function _sampler_from_toml(::FeasibleStrategy, s::AbstractDict)
    g(k, d) = haskey(s, k) ? s[k] : d
    return FeasibleSamplerConfig(
        s_total_min = Float64(g("s_total_min", 1.0)),
        s_total_max = Float64(g("s_total_max", 4.0)),
        pf_min      = Float64(g("pf_min", 0.9)),
        pf_max      = Float64(g("pf_max", 1.0)),
        vset_min    = Float64(g("vset_min", 1.0)),
        vset_max    = Float64(g("vset_max", 1.05)),
        tau         = Float64(g("tau", 1.0e-8)),
        max_iters   = Int(g("max_iters", 100)),
        max_attempts_mult = Int(g("max_attempts_mult", 60)),
    )
end

function load_dataset_config(path::AbstractString)
    d   = TOML.parsefile(path)
    sys = d["system"]
    ds  = d["dataset"]
    out = d["output"]
    samp = get(d, "sampler", Dict{String,Any}())

    raw_dir    = out["dir"]
    output_dir = isabspath(raw_dir) ? raw_dir :
                 normpath(joinpath(dirname(abspath(path)), raw_dir))

    name  = splitext(basename(path))[1]
    haskey(sys, "K_DV") || error(
        "K_DV must be specified in [system] " *
        "(e.g. K_DV = 100.0 for a uniform fallback, or K_DV = [v1, v2, ...] per generator)"
    )
    K_DV_raw = sys["K_DV"]
    K_DV  = K_DV_raw isa AbstractVector ? Float64.(K_DV_raw) : Float64(K_DV_raw)

    slack_indices = Int.(get(ds, "slack_indices", [1]))

    # Strategy selection: the [sampler] strategy key (default "balanced") resolves
    # to a SamplerStrategy; the per-strategy field parser then reads its own knobs.
    strat = sampler_strategy(String(get(samp, "strategy", "balanced")))

    return DatasetConfig(
        name,
        Symbol(sys["type"]),
        sys["name"],
        K_DV,
        _sampler_from_toml(strat, samp),
        ds["n_train"],
        ds["n_test"],
        ds["seed_train"],
        ds["seed_test"],
        slack_indices,
        output_dir,
    )
end

# ── Paths ─────────────────────────────────────────────────────────────────────

dataset_path(config::DatasetConfig) = joinpath(config.output_dir, "data.jld2")

# ── Power system construction ─────────────────────────────────────────────────

function build_powersystem(config::DatasetConfig)
    return build_powersystem(Symbol(config.system_name); K_DV = config.K_DV)
end

# ── Signature ─────────────────────────────────────────────────────────────────

signature(config::DatasetConfig, ps::PowerSystem) =
    DatasetSignature(config.system_name, ps.n_controls, ps.n_variables)

# ── Dataset I/O (config-aware) ────────────────────────────────────────────────

_config_hash(config_path::AbstractString) = bytes2hex(sha256(read(config_path)))

# ── Schema + provenance contract ──────────────────────────────────────────────
#
# SCHEMA_VERSION: integer stamped into every written dataset's metadata. Bump it
# when the on-disk metadata layout changes in a way a loader must know about.
# Loading a file whose schema_version exceeds this constant is a hard error.
const SCHEMA_VERSION = 1

# SAMPLER_ID: algorithm tag for the operating-condition sampler that produced the
# data. BUMP DISCIPLINE — increment the version suffix whenever a change to
# sampling_operating_condition.jl alters the *generated* control/voltage values
# for a fixed seed (e.g. the calibrate_capacity non-monotone fix). A value-
# preserving refactor does not need a bump. Datasets stamped with an older
# sampler_id are reported stale (with a warning) by `dataset_is_current` so the
# caller regenerates deliberately rather than loading silently-drifted data.
const SAMPLER_ID = "balanced-oc/v1"

# Package version string (semver from Project.toml), always available. Used both
# as provenance and as part of the freshness check. The git SHA is recorded
# separately (`_git_sha`) as pure provenance and is NOT part of the freshness
# key, so an ordinary commit does not force regeneration.
function rpf_version()
    v = pkgversion(@__MODULE__)
    return v === nothing ? "unknown" : string(v)
end

# Best-effort short git SHA of the package checkout. Guarded: any failure (git
# not on PATH, not a checkout, detached state) degrades to "unknown" — never
# propagated, never fails a write.
function _git_sha()
    try
        dir = pkgdir(@__MODULE__)
        dir === nothing && return "unknown"
        sha = readchomp(Cmd(`git rev-parse --short HEAD`; dir = dir))
        return isempty(sha) ? "unknown" : sha
    catch
        return "unknown"
    end
end

# Strategy + per-strategy sampler_id for a resolved config.
_strategy(config::DatasetConfig) = _strategy_for(config.sampler)
sampler_id(config::DatasetConfig) = sampler_id(_strategy(config))

# Digest over the *resolved* semantics that affect generated data: system_name,
# K_DV, the sample counts, the seeds, the selected strategy selector AND every
# field of the active strategy's config. Deliberately excludes output_dir, the
# config name (filename) and the raw source bytes (comments / whitespace / key
# order), so a cosmetic TOML edit does not force a regen while any data-affecting
# change does. slack_indices is excluded because the balanced sampler ignores it
# (solves with plain GN, per CLAUDE.md). Adding the strategy selector to the digest
# is a deliberate one-time change: existing datasets restamp on next regeneration.
function config_digest(config::DatasetConfig)
    io = IOBuffer()
    print(io, "system_name=", config.system_name, ";")
    print(io, "K_DV=", config.K_DV, ";")
    print(io, "n_train=", config.n_train, ";")
    print(io, "n_test=", config.n_test, ";")
    print(io, "seed_train=", config.seed_train, ";")
    print(io, "seed_test=", config.seed_test, ";")
    sc = config.sampler
    print(io, "strategy=", strategy_name(_strategy_for(sc)), ";")
    for f in fieldnames(typeof(sc))
        print(io, "sampler.", f, "=", getfield(sc, f), ";")
    end
    return bytes2hex(sha256(take!(io)))
end

