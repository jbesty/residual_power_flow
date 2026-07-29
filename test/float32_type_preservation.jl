@testset "Float32 type preservation in solve!" begin
    builder = PowerSystemBuilder{Float32}(2)
    add_one_bus_injector!(builder, 1, SynchronousMachineStatic(Float32(1.0)))
    add_two_bus_injector!(builder, 1, 2, Line(Float32(0.01), Float32(0.05), Float32(0.02)))
    ps = build!(builder)

    v0 = get_flat_start(ps)
    u  = Float32[1.0, 1.0]

    state = PowerFlowState(ps, v0, u)
    solve!(state, GaussNewtonSolver())

    @test eltype(state.voltages) == Float32
    @test eltype(state.controls) == Float32
end
