@testset "Matrices" begin
    k = 1.49
    sigma_chi = 0.286
    lambda_chi = 0.157
    mi_xi = -0.0125
    sigma_xi = 0.145
    mi_xi_star = 0.0115
    rho_xi_chi = 0.3

    s = [
        0.042;
        0.006;
    ]


    p = SchwartzSmith.SSParams(k,sigma_chi,lambda_chi,mi_xi,sigma_xi,mi_xi_star,rho_xi_chi,s)

    D = Vector{Float64}(undef, 0)

    @testset "SSParams" begin

        @test p.k ≈ 1.49 atol = 1e-4 rtol = 1e-4
        @test p.σ_χ ≈ 0.286 atol = 1e-4 rtol = 1e-4
        @test p.λ_χ ≈ 0.157 atol = 1e-4 rtol = 1e-4
        @test p.μ_ξ ≈ -0.0125 atol = 1e-4 rtol = 1e-4
        @test p.σ_ξ ≈ 0.145 atol = 1e-4 rtol = 1e-4
        @test p.μ_ξ_star ≈ 0.0115 atol = 1e-4 rtol = 1e-4
        @test p.ρ_ξχ ≈ 0.3 atol = 1e-4 rtol = 1e-4
        @test p.s ≈ [0.042; 0.006] atol = 1e-4 rtol = 1e-4
    end

    @testset "Matrices Functions" begin
        T = [10.0; 20.0];

        @test SchwartzSmith.A(T[1], p) ≈ 0.13683 atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.A(T[2], p) ≈ 0.356955 atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.V(p) ≈ [1.764e-3 0; 0 3.6e-5] atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.W(p, 1) ≈ [0.026054 0.006468; 0.006468 0.021025] atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.G(p, 0, 1) ≈ [0.225373 0; 0 1] atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.G(p, 4, 1) ≈ [0.225373 0 0 0 0 0; 0 1 0 0 0 0;0 0 1 0 0 0;0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1] atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.c(p, 0, 1) ≈ [0; -0.0125] atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.c(p, 4, 1) ≈ [0; -0.0125; 0; 0; 0; 0] atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.d(T, p) ≈ [0.13683; 0.356955] atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.F(T, p, D) ≈ [3.38074e-7 1; 1e-8 1] atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.F(T, p, [1.; 0.; 0.; 0.]) ≈ [3.38074e-7 1 1 0 0 0; 1e-8 1 1 0 0 0] atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.R(0) ≈ [1 0; 0 1] atol = 1e-4 rtol = 1e-4
        @test SchwartzSmith.R(4) ≈ [1 0; 0 1; 0 0; 0 0; 0 0; 0 0] atol = 1e-4 rtol = 1e-4

    end
end
