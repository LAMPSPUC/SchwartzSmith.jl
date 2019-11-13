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
    schwartzsmith(ln_F::Matrix{Typ}, T::Matrix{Typ}; delta_t::Int = 1, seed::Vector{Typ} = calc_seed(ln_F, T, 5, delta_t)) where Typ

Estimation of the Schwartz Smith model with a matrix of time to maturity as an input. Returns the estimated parameters.
"""
function schwartzsmith(ln_F::Matrix{Typ}, T::Matrix{Typ}; delta_t::Int = 1, seed::Vector{Typ} = calc_seed(ln_F, T, 5, delta_t)) where Typ

    # Parameters estimation
    n, prods = size(ln_F)

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
    schwartzsmith(ln_F::Matrix{Typ}, T_V::Vector{Typ}; delta_t::Int = 1, seed::Vector{Typ} = calc_seed(ln_F, T_V, 5, delta_t)) where Typ

Estimation of the Schwartz Smith model with a vector of average time to maturity as an input. Returns the estimated parameters.
"""
function schwartzsmith(ln_F::Matrix{Typ}, T_V::Vector{Typ}; delta_t::Int = 1, seed::Vector{Typ} = calc_seed(ln_F, T_V, 5, delta_t)) where Typ

    # Parameters estimation
    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    p, seed, optseed = schwartzsmith(ln_F, T; seed = seed, delta_t = delta_t)

    return p, seed, optseed
end


"""
    estimated_prices_states(p::SSParams{Typ}, T::Matrix{Typ}, ln_F::Matrix{Typ}; delta_t::Int = 1) where Typ

Returns the prices and the kalman filter struct estimated by the model. Matrix of time to maturity as an input.
"""
function estimated_prices_states(p::SSParams{Typ}, T::Matrix{Typ}, ln_F::Matrix{Typ}; delta_t::Int = 1) where Typ
    n, prods = size(T)
    y = Array{Typ, 2}(undef, n, prods)

    f = kalman_filter(ln_F, T, p, delta_t)

    for t in 1:n
        y[t, :] = d(T[t, :], p) + F(T[t, :], p) * f.att_kf[t, :]
    end

    return y, f
end

"""
    estimated_prices_states(p::SSParams{Typ}, T_V::Vector{Typ}, ln_F::Matrix{Typ}; delta_t::Int = 1) where Typ

Returns the prices and the kalman filter struct estimated by the model. Vector of average time to maturity as an input.
"""
function estimated_prices_states(p::SSParams{Typ}, T_V::Vector{Typ}, ln_F::Matrix{Typ}; delta_t::Int = 1) where Typ
    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    y, f = estimated_prices_states(p, T, ln_F; delta_t = delta_t)

    return y, f
end

end
