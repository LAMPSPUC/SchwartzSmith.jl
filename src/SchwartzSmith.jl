module SchwartzSmith

using Optim
using LinearAlgebra
using Distributions

export schwartzsmith, estimated_prices_states, simulate, forecast

include("create_matrices.jl")
include("auxiliar.jl")
include("kalman_filter.jl")
include("sqrt_kalman_filter.jl")
include("estimation.jl")
include("simulation.jl")
include("forecast.jl")

"""
    schwartzsmith(ln_F::Matrix{Typ}, T::Matrix{Typ}; delta_t::Int = 1, seed::Vector{Typ} = -0.2*rand(Typ, 7 + size(ln_F, 2))) where Typ

Estimation of the Schwartz Smith model. Returns the estimated parameters.
"""
function schwartzsmith(ln_F::Matrix{Typ}, T::Matrix{Typ}; delta_t::Int = 1, seed = nothing, average_T::String = "true") where Typ

    # Parameters estimation
    n, prods = size(ln_F)

    # Calculation of average time to maturity
    if average_T == "true"
        T_M = Matrix{Float64}(undef, n, prods)

        for i in 1:n, j in 1:prods
            T_M[i, j] = mean(T[:, j])
        end
        T = T_M
    end

    # Test with different seeds
    if seed == nothing
        seed_l = -0.2*rand(Typ, 7 + size(ln_F, 2))
        min_l = compute_likelihood(ln_F, T, seed_l, delta_t)
        for i = 1:5
            seed = -0.2*rand(Typ, 7 + size(ln_F, 2))
            calc_l = compute_likelihood(ln_F, T, seed, delta_t)

            if calc_l < min_l
                min_l = calc_l
                seed_l = seed
            end
        end
        seed = seed_l
    end

    optseed = optimize(psi -> compute_likelihood(ln_F, T, psi, delta_t), seed, LBFGS(), Optim.Options(f_tol = 1e-6, g_tol = 1e-6, show_trace = true))

    opt_param = optseed.minimizer

    # Results
    k = exp(opt_param[1])
    sigma_chi = exp(opt_param[2])
    lambda_chi = opt_param[3]
    mi_xi = opt_param[4]
    sigma_xi = exp(opt_param[5])
    mi_xi_star = opt_param[6]
    rho_xi_chi = -1 + 2/(1 + exp(-opt_param[7]))
    s = exp.(opt_param[8:end])
    p = SSParams(k, sigma_chi, lambda_chi, mi_xi, sigma_xi, mi_xi_star, rho_xi_chi, s)

    return p, seed, optseed
end

"""
    estimated_prices_states(p::SSParams{Typ}, T::Matrix{Typ}, ln_F::Matrix{Typ}; delta_t::Int = 1) where Typ

Returns the prices and the kalman filter struct estimated by the model.
"""
function estimated_prices_states(p::SSParams{Typ}, T::Matrix{Typ}, ln_F::Matrix{Typ}; delta_t::Int = 1, average_T::String = "true") where Typ
    n, prods = size(T)
    y = Array{Typ, 2}(undef, n, prods)

    # Calculation of average time to maturity
    if average_T == "true"
        T_M = Matrix{Float64}(undef, n, prods)

        for i in 1:n, j in 1:prods
            T_M[i, j] = mean(T[:, j])
        end
        T = T_M
    end

    f = kalman_filter(ln_F, T, p, delta_t)

    for t in 1:n
        y[t, :] = d(T[t, :], p) + F(T[t, :], p) * f.att_kf[t, :]
    end

    return y, f
end


end
