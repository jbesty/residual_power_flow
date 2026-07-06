# Dataset-artifact contract: schema version, provenance, signature check,
# resolved-semantics digest, two-flag storage, legacy load. Unit tier — builds
# tiny datasets from the case9 fixture (no sampler); the infeasible-centred
# feasible/stationary distinction needs the sampler and lives in the slow tier
# (learning_dataset.jl).

import ResidualPowerFlow: SCHEMA_VERSION, SAMPLER_ID, config_digest, rpf_version
import ResidualPowerFlow: feasible, stationary
import ResidualPowerFlow: sampler_strategy, sampler_id, BalancedStrategy,
    ImportanceStrategy, PerturbationStrategy

function _write_contract_config(path::AbstractString;
                                seed_train::Int = 1, dir::String = "out",
                                comment::Bool = false)
    lead = comment ? "# a harmless comment, semantics unchanged\n\n" : ""
    write(path, """
    $(lead)[system]
    type = "matpower"
    name = "case9"
    K_DV = [130.0, 21.0, 13.0]

    [dataset]
    n_train = 2
    n_test = 2
    seed_train = $(seed_train)
    seed_test = 2
    slack_indices = []

    [output]
    dir = "$(replace(dir, "\\" => "/"))"

    [sampler]
    cap_samples = 20
    """)
end

@testset "config_digest over resolved semantics" begin
    tmpdir = mktempdir()
    try
        a = joinpath(tmpdir, "a.toml"); _write_contract_config(a; seed_train = 1, dir = "out_a")
        # Differs only in comment, whitespace and output_dir — resolved semantics
        # identical, so the digest must match.
        b = joinpath(tmpdir, "b.toml"); _write_contract_config(b; seed_train = 1, dir = "out_b", comment = true)
        # Differs in a data-affecting field (seed_train).
        c = joinpath(tmpdir, "c.toml"); _write_contract_config(c; seed_train = 2, dir = "out_a")

        ca = load_dataset_config(a)
        cb = load_dataset_config(b)
        cc = load_dataset_config(c)

        @test config_digest(ca) == config_digest(cb)   # cosmetic / output_dir insensitive
        @test config_digest(ca) != config_digest(cc)   # data-field sensitive
        @test length(config_digest(ca)) == 64          # SHA-256 hex
    finally
        rm(tmpdir; recursive = true, force = true)
    end
end

# Strategy-aware digest: the selector and the active strategy's knobs both feed the
# digest; cosmetic edits do not. Pure (no power system needed).
function _write_strategy_config(path; strategy::String, sampler_lines::String = "",
                                dir::String = "out", comment::Bool = false)
    lead = comment ? "# harmless comment\n\n" : ""
    write(path, """
    $(lead)[system]
    type = "matpower"
    name = "case9"
    K_DV = [130.0, 21.0, 13.0]

    [dataset]
    n_train = 2
    n_test = 2
    seed_train = 1
    seed_test = 2

    [output]
    dir = "$(replace(dir, "\\" => "/"))"

    [sampler]
    strategy = "$(strategy)"
    $(sampler_lines)
    """)
end

@testset "config_digest is strategy-aware" begin
    tmpdir = mktempdir()
    try
        bal = joinpath(tmpdir, "bal.toml"); _write_strategy_config(bal; strategy = "balanced")
        fea = joinpath(tmpdir, "fea.toml")
        _write_strategy_config(fea; strategy = "feasible", sampler_lines = "s_total_max = 4.0")
        # Same strategy, one knob changed.
        fea2 = joinpath(tmpdir, "fea2.toml")
        _write_strategy_config(fea2; strategy = "feasible", sampler_lines = "s_total_max = 5.0")
        # Same feasible config, cosmetic edit + different output_dir only.
        feac = joinpath(tmpdir, "feac.toml")
        _write_strategy_config(feac; strategy = "feasible", sampler_lines = "s_total_max = 4.0",
                               dir = "other", comment = true)

        d_bal  = config_digest(load_dataset_config(bal))
        d_fea  = config_digest(load_dataset_config(fea))
        d_fea2 = config_digest(load_dataset_config(fea2))
        d_feac = config_digest(load_dataset_config(feac))

        @test d_bal != d_fea     # strategy selector changes the digest
        @test d_fea != d_fea2    # strategy-specific knob changes the digest
        @test d_fea == d_feac    # cosmetic / output_dir insensitive
    finally
        rm(tmpdir; recursive = true, force = true)
    end
end

@testset "sampler strategy selection is total" begin
    import ResidualPowerFlow: sampler_strategy, sampler_id, BalancedStrategy, FeasibleStrategy
    @test sampler_strategy("balanced") isa BalancedStrategy
    @test sampler_strategy("feasible") isa FeasibleStrategy
    @test sampler_id(sampler_strategy("balanced")) == "balanced-oc/v1"
    @test sampler_id(sampler_strategy("feasible")) == "feasible/v1"
    err = try; sampler_strategy("nope"); nothing; catch e; e; end
    @test err isa ArgumentError
    @test occursin("nope", sprint(showerror, err))
    @test occursin("balanced", sprint(showerror, err))   # lists registered names
end

isnothing(_CASE9) && return

let
    power_system = _CASE9.power_system
    vars = target_variables(_CASE9)
    ctrl = controls_from_solution(_CASE9)
    _mk(n) = PowerFlowDataset(power_system,
                              [PowerFlowState(power_system, copy(vars), copy(ctrl)) for _ in 1:n])
    train = _mk(2)
    test  = _mk(2)

    @testset "stamped writer round-trip: provenance + signature present" begin
        tmpdir = mktempdir()
        try
            cfg_path = joinpath(tmpdir, "case9_contract.toml")
            _write_contract_config(cfg_path, dir = joinpath(tmpdir, "out"))
            cfg = load_dataset_config(cfg_path)

            save_config_dataset(cfg, cfg_path, train, test)
            path = dataset_path(cfg)
            @test isfile(path)

            jldopen(path, "r") do f
                @test f["metadata/schema_version"] == SCHEMA_VERSION
                @test f["metadata/sampler_id"] == SAMPLER_ID
                @test !isempty(f["metadata/rpf_version"])
                @test f["metadata/git_sha"] isa AbstractString   # SHA or "unknown"
                @test !isempty(f["metadata/git_sha"])
                @test !isempty(f["metadata/config_digest"])
                @test f["metadata/seed_train"] == cfg.seed_train
                @test f["metadata/seed_test"]  == cfg.seed_test
                @test f["metadata/has_signature"] == true
                @test f["metadata/signature/system_name"] == "case9"
                @test f["metadata/signature/n_controls"]  == power_system.n_controls
                @test f["metadata/signature/n_variables"] == power_system.n_variables
                # Reserved provenance keys are stamped (follow-up fills values).
                @test f["metadata/capacity"]   == "unknown"
                @test f["metadata/load_range"] == "unknown"
                @test f["metadata/yield"]      == "unknown"
                # Both flags stored per split.
                @test f["train/feasible"]   isa Vector{Bool}
                @test f["train/stationary"] isa Vector{Bool}
                @test f["test/feasible"]    isa Vector{Bool}
                @test f["test/stationary"]  isa Vector{Bool}
            end

            loaded = load_dataset(path, :train, power_system)
            @test loaded isa PowerFlowDataset
            @test length(feasible(loaded))   == 2     # both flags available on load
            @test length(stationary(loaded)) == 2
        finally
            rm(tmpdir; recursive = true, force = true)
        end
    end

    @testset "signature mismatch raises on load" begin
        path = tempname() * ".jld2"
        try
            jldopen(path, "w") do f
                f["metadata/schema_version"]        = SCHEMA_VERSION
                f["metadata/has_signature"]         = true
                f["metadata/signature/system_name"] = "case9"
                f["metadata/signature/n_controls"]  = power_system.n_controls + 1
                f["metadata/signature/n_variables"] = power_system.n_variables
                f["train/controls"] = controls_matrix(train)
                f["train/voltages"] = voltages_matrix(train)
            end
            @test_throws ErrorException load_dataset(path, :train, power_system)
        finally
            rm(path; force = true)
        end
    end

    @testset "over-version raises on load" begin
        path = tempname() * ".jld2"
        try
            jldopen(path, "w") do f
                f["metadata/schema_version"] = SCHEMA_VERSION + 1
                f["train/controls"] = controls_matrix(train)
                f["train/voltages"] = voltages_matrix(train)
            end
            @test_throws ErrorException load_dataset(path, :train, power_system)
        finally
            rm(path; force = true)
        end
    end

    @testset "legacy file loads with unverified-provenance warning" begin
        path = tempname() * ".jld2"
        try
            # Old layout: only the pre-contract metadata, no schema_version.
            jldopen(path, "w") do f
                f["metadata/config_hash"] = "deadbeef"
                f["metadata/n_train"]     = 2
                f["train/controls"]  = controls_matrix(train)
                f["train/voltages"]  = voltages_matrix(train)
                f["train/converged"] = Vector{Bool}(feasible(train))
            end
            loaded = @test_logs (:warn,) match_mode = :any load_dataset(path, :train, power_system)
            @test loaded isa PowerFlowDataset
            @test length(loaded) == 2
        finally
            rm(path; force = true)
        end
    end

    @testset "freshness keys on semantics + version, not bytes" begin
        tmpdir = mktempdir()
        try
            cfg_path = joinpath(tmpdir, "case9_fresh.toml")
            _write_contract_config(cfg_path, dir = joinpath(tmpdir, "out"))
            cfg = load_dataset_config(cfg_path)
            save_config_dataset(cfg, cfg_path, train, test)

            @test dataset_is_current(cfg, cfg_path)

            # Comment/whitespace-only edit: resolved config unchanged ⇒ still current.
            write(cfg_path, read(cfg_path, String) * "\n\n# trailing comment\n")
            cfg2 = load_dataset_config(cfg_path)
            @test dataset_is_current(cfg2, cfg_path)

            # Sampler-id drift: digest matches but the stored sampler tag is stale
            # ⇒ not current AND warns.
            path = dataset_path(cfg2)
            jldopen(path, "w") do f
                f["metadata/schema_version"] = SCHEMA_VERSION
                f["metadata/config_digest"]  = config_digest(cfg2)
                f["metadata/rpf_version"]    = rpf_version()
                f["metadata/sampler_id"]     = "balanced-oc/v0"
            end
            @test_logs (:warn,) match_mode = :any dataset_is_current(cfg2, cfg_path)
            @test !dataset_is_current(cfg2, cfg_path)
        finally
            rm(tmpdir; recursive = true, force = true)
        end
    end
end
