isnothing(_CASE9) && return

function _write_io_config(path::AbstractString, output_dir::AbstractString;
                          slack_indices::String = "[]", seed_train::Int = 1)
    write(path, """
    [system]
    type = "matpower"
    name = "case9"
    K_DV = [130.0, 21.0, 13.0]

    [dataset]
    n_train = 2
    n_test = 2
    seed_train = $seed_train
    seed_test = 2
    slack_indices = $slack_indices

    [output]
    dir = "$(replace(output_dir, "\\" => "/"))"

    [sampler]
    cap_samples = 20
    """)
end

@testset "load_or_generate_datasets" begin
    power_system = _CASE9.power_system

    @testset "cold cache → generate + save" begin
        tmpdir = mktempdir()
        try
            cfg_path = joinpath(tmpdir, "case9_io.toml")
            out_dir  = joinpath(tmpdir, "out")
            _write_io_config(cfg_path, out_dir)
            cfg = load_dataset_config(cfg_path)

            result = load_or_generate_datasets(cfg, power_system, cfg_path)
            @test result.train isa PowerFlowDataset
            @test result.test  isa PowerFlowDataset
            @test length(result.train) == 2
            @test length(result.test)  == 2
            @test isfile(dataset_path(cfg))
            @test dataset_is_current(cfg, cfg_path)
        finally
            rm(tmpdir; recursive = true, force = true)
        end
    end

    @testset "warm cache → load" begin
        tmpdir = mktempdir()
        try
            cfg_path = joinpath(tmpdir, "case9_io.toml")
            out_dir  = joinpath(tmpdir, "out")
            _write_io_config(cfg_path, out_dir)
            cfg = load_dataset_config(cfg_path)

            first  = load_or_generate_datasets(cfg, power_system, cfg_path)
            second = load_or_generate_datasets(cfg, power_system, cfg_path)
            @test controls_matrix(second.train) ≈ controls_matrix(first.train)
            @test voltages_matrix(second.train) ≈ voltages_matrix(first.train)
            @test controls_matrix(second.test)  ≈ controls_matrix(first.test)
            @test voltages_matrix(second.test)  ≈ voltages_matrix(first.test)
        finally
            rm(tmpdir; recursive = true, force = true)
        end
    end

    @testset "freshness keys on resolved semantics, not raw bytes" begin
        tmpdir = mktempdir()
        try
            cfg_path = joinpath(tmpdir, "case9_io.toml")
            out_dir  = joinpath(tmpdir, "out")
            _write_io_config(cfg_path, out_dir)
            cfg = load_dataset_config(cfg_path)

            load_or_generate_datasets(cfg, power_system, cfg_path)
            @test dataset_is_current(cfg, cfg_path)

            # A trailing-newline / comment edit changes the file bytes but not
            # the resolved config, so the dataset stays current — no spurious
            # multi-hour regen. (Old byte-hash behaviour would have flagged it.)
            write(cfg_path, read(cfg_path, String) * "\n# harmless comment\n")
            cfg_ws = load_dataset_config(cfg_path)
            @test dataset_is_current(cfg_ws, cfg_path)

            # A data-affecting change (different seed) flips the digest, so the
            # dataset is no longer current and regenerates.
            _write_io_config(cfg_path, out_dir; seed_train = 99)
            cfg_new = load_dataset_config(cfg_path)
            @test !dataset_is_current(cfg_new, cfg_path)
            result = load_or_generate_datasets(cfg_new, power_system, cfg_path)
            @test result.train isa PowerFlowDataset
            @test dataset_is_current(cfg_new, cfg_path)
        finally
            rm(tmpdir; recursive = true, force = true)
        end
    end

    @testset "force = true → regenerate and overwrite" begin
        tmpdir = mktempdir()
        try
            cfg_path = joinpath(tmpdir, "case9_io.toml")
            out_dir  = joinpath(tmpdir, "out")
            _write_io_config(cfg_path, out_dir)
            cfg = load_dataset_config(cfg_path)

            load_or_generate_datasets(cfg, power_system, cfg_path)
            mtime_before = mtime(dataset_path(cfg))
            sleep(1.1)  # ensure mtime resolution differs

            result = load_or_generate_datasets(cfg, power_system, cfg_path; force = true)
            @test result.train isa PowerFlowDataset
            @test isfile(dataset_path(cfg))
            @test mtime(dataset_path(cfg)) > mtime_before
        finally
            rm(tmpdir; recursive = true, force = true)
        end
    end

    @testset "non-empty slack_indices still generates (balanced sampler ignores PtO)" begin
        tmpdir = mktempdir()
        try
            cfg_path = joinpath(tmpdir, "case9_io_pto.toml")
            out_dir  = joinpath(tmpdir, "out")
            _write_io_config(cfg_path, out_dir; slack_indices = "[1]")
            cfg = load_dataset_config(cfg_path)
            @test cfg.slack_indices == [1]

            result = load_or_generate_datasets(cfg, power_system, cfg_path)
            @test result.train isa PowerFlowDataset
            @test length(result.train) == 2
            @test length(result.test)  == 2
            @test isfile(dataset_path(cfg))
        finally
            rm(tmpdir; recursive = true, force = true)
        end
    end
end

@testset "LBFGSTraining resolves from parent module" begin
    nste = Base.get_extension(ResidualPowerFlow, :NeuralSolverTrainingExt)
    @test ResidualPowerFlow.LBFGSTraining(max_iterations = 1) isa nste.NeuralTraining
    @test ResidualPowerFlow.AdamTraining(n_epochs = 1)        isa nste.NeuralTraining
end
