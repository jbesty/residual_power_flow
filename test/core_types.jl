@testset "Core types" begin

    # Float32 and Float64 are the only allowed numeric types for components and power systems.
    @test ResidualPowerFlow.Float64 <: ResidualPowerFlow.ALLOWED_TYPES
    @test ResidualPowerFlow.Float32 <: ResidualPowerFlow.ALLOWED_TYPES
    @test !(Int <: ResidualPowerFlow.ALLOWED_TYPES)

    @testset "SynchronousMachineStatic" begin
        sm = SynchronousMachineStatic(0.7)
        @test sm isa ResidualPowerFlow.Component{Float64}
        @test sm.K_DV == 0.7

        sm32 = SynchronousMachineStatic(Float32(0.25))
        @test sm32 isa ResidualPowerFlow.Component{Float32}
        @test sm32.K_DV == Float32(0.25)
    end

    @testset "ZIP loads" begin
        V0 = 1.0
        # Each convenience constructor sets exactly one ZIP coefficient to 1, the others to 0.
        cp = ConstantPowerLoad(V0)
        @test cp isa ResidualPowerFlow.Component{Float64}
        @test cp.V_0 == V0
        @test cp.a == 1.0
        @test cp.b == 0.0
        @test cp.c == 0.0

        cc = ConstantCurrentLoad(V0)
        @test cc.a == 0.0
        @test cc.b == 1.0
        @test cc.c == 0.0

        ci = ConstantImpedanceLoad(V0)
        @test ci.a == 0.0
        @test ci.b == 0.0
        @test ci.c == 1.0

        V0f = Float32(1.02)
        cp32 = ConstantPowerLoad(V0f)
        @test cp32 isa ResidualPowerFlow.Component{Float32}
        @test cp32.V_0 == V0f
        @test cp32.a == one(Float32)
        @test cp32.b == zero(Float32)
        @test cp32.c == zero(Float32)

        # Direct ZIPLoad constructor with arbitrary weights (need not sum to 1).
        zip = ZIPLoad(1.05, 0.2, 0.3, 0.5)
        @test zip isa ResidualPowerFlow.Component{Float64}
        @test zip.V_0 == 1.05
        @test zip.a == 0.2
        @test zip.b == 0.3
        @test zip.c == 0.5
    end

    @testset "Shunt" begin
        sh = Shunt(0.1, -0.2)
        @test sh isa ResidualPowerFlow.Component{Float64}
        @test sh.G == 0.1
        @test sh.B == -0.2
    end

    @testset "Branch / Line / TransformerSimple" begin
        R = 0.01
        X = 0.05
        B = 0.02
        TAP = 1.1
        θ = 0.1
        br = Branch(R, X, B, TAP, θ)
        @test br isa ResidualPowerFlow.Component{Float64}
        @test br.R == R
        @test br.X == X
        @test br.B == B
        @test br.TAP == TAP
        @test br.θ_shift == θ

        # Verify the precomputed admittance parameters against the standard π-model
        # formulas for a branch with tap ratio TAP and phase shift θ.
        Z_S = R + X * im
        Y_S = 1 / Z_S
        Y_11 = (Y_S + B / 2 * im) / (TAP^2)
        Y_12 = -Y_S * (1 / TAP) * (1 / exp(-im * θ))
        Y_21 = -Y_S * (1 / TAP) * (1 / exp(im * θ))
        Y_22 = (Y_S + B / 2 * im)

        @test br.Z_S == Z_S
        @test isapprox(br.Y_S, Y_S; rtol = 1e-12, atol = 0)
        @test isapprox(br.Y_11, Y_11; rtol = 1e-12, atol = 0)
        @test isapprox(br.Y_12, Y_12; rtol = 1e-12, atol = 0)
        @test isapprox(br.Y_21, Y_21; rtol = 1e-12, atol = 0)
        @test isapprox(br.Y_22, Y_22; rtol = 1e-12, atol = 0)

        # Line and TransformerSimple are convenience constructors for degenerate Branch cases.
        ln = Line(R, X, B)
        br_ref = Branch(R, X, B, 1.0, 0.0)   # TAP=1, θ_shift=0
        @test ln.Z_S == br_ref.Z_S
        @test ln.Y_11 == br_ref.Y_11
        @test ln.Y_12 == br_ref.Y_12
        @test ln.Y_21 == br_ref.Y_21
        @test ln.Y_22 == br_ref.Y_22

        tr = TransformerSimple(X)             # R=0, B=0, TAP=1, θ_shift=0
        @test tr.R == 0.0
        @test tr.X == X
        @test tr.B == 0.0
        @test tr.TAP == 1.0
        @test tr.θ_shift == 0.0
    end

    @testset "Type constraints" begin
        # Component is a union alias — no T-bound at the type level, but constructors reject non-ALLOWED_TYPES.
        @test Core.apply_type(ResidualPowerFlow.Component, Float64) ==
              ResidualPowerFlow.Component{Float64}

        @test_throws TypeError Core.apply_type(ResidualPowerFlow.PowerSystem, Int)
        @test Core.apply_type(ResidualPowerFlow.PowerSystem, Float32) ==
              ResidualPowerFlow.PowerSystem{Float32}

        # Inner constructors that take typed arguments reject integers (no method for T=Int).
        @test_throws MethodError SynchronousMachineStatic(1)
        @test_throws MethodError ConstantPowerLoad(1)
        @test_throws MethodError ConstantCurrentLoad(1)
        @test_throws MethodError ConstantImpedanceLoad(1)
        @test_throws MethodError Shunt(1, 2)
        @test_throws MethodError Branch(1, 1, 1, 1, 1)

        # Zero-impedance branch must throw.
        @test_throws ArgumentError Branch(0.0, 0.0, 0.0, 1.0, 0.0)
        @test_throws ArgumentError Branch(0.0f0, 0.0f0, 0.0f0, 1.0f0, 0.0f0)
    end

end
