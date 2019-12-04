module SchwartzSmith

using Optim
using LinearAlgebra
using Distributions

export schwartzsmith, estimated_prices_states, simulate, forecast, calc_seed, kalman_filter

include("create_matrices.jl")
include("auxiliar.jl")
include("kalman_filter.jl")
include("sqrt_kalman_filter.jl")
include("estimation.jl")
include("simulation.jl")
include("forecast.jl")
include("estimated_prices.jl")

"""
    schwartzsmith(ln_F::Matrix{Typ}, T::Matrix{Typ}; delta_t::Int = 1, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

Estimation of the Schwartz Smith model with a matrix of time to maturity as an input. Returns the estimated parameters.
"""
function schwartzsmith(ln_F::Matrix{Typ}, T::Matrix{Typ}; delta_t::Int = 1, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

    # Alocate memory
    n, prods = size(ln_F)
    s = 0 # No seasonality to be modelled
    D = Matrix{Float64}(undef, 0, 0)

    n_psi         = 7 + prods
    n_seeds       = size(seeds, 2)
    @assert size(seeds, 1) == n_psi
    loglikelihood = Vector{Typ}(undef, n_seeds)
    psitilde      = Matrix{Typ}(undef, n_psi, n_seeds)
    optseeds      = Vector{Optim.OptimizationResults}(undef, n_seeds)

    # Optimization
    for i in 1:n_seeds
        try
            println("-------------------- Seed ", i, "--------------------")
            # optimize
            optseed = optimize(psi -> compute_likelihood(ln_F, T, D, psi, delta_t), seeds[:, i], LBFGS(), Optim.Options(f_tol = 1e-6, g_tol = 1e-6, show_trace = true))
            # allocate log_lik and minimizer
            loglikelihood[i] = -optseed.minimum
            psitilde[:, i] = optseed.minimizer
            optseeds[i] = optseed
        catch err
            println(err)
        end
    end

    # Query results
    log_lik, best_seed = findmax(loglikelihood)

    opt_param = psitilde[:, best_seed]

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

    return p, seeds[:, best_seed], optseeds[best_seed]
end

"""
    schwartzsmith(ln_F::Matrix{Typ}, T_V::Vector{Typ}; delta_t::Int = 1, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

Estimation of the Schwartz Smith model with a vector of average time to maturity as an input. Returns the estimated parameters.
"""
function schwartzsmith(ln_F::Matrix{Typ}, T_V::Vector{Typ}; delta_t::Int = 1, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

    # Parameters estimation
    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    p, seed, optseeds = schwartzsmith(ln_F, T; delta_t = delta_t, seeds = seeds)

    return p, seed, optseeds
end

"""
    schwartzsmith(ln_F::Matrix{Typ}, T::Matrix{Typ}, dates::Vector{Int64}, s::Int64; delta_t::Int = 1, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

Estimation of the Schwartz Smith model with a matrix of time to maturity as an input. Returns the estimated parameters.
"""
function schwartzsmith(ln_F::Matrix{Typ}, T::Matrix{Typ}, dates::Vector{Int64}, s::Int64; delta_t::Int = 1, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

    # Alocate memory
    n, prods = size(ln_F)
    D = calc_D(s, dates)

    n_psi         = 7 + prods
    n_seeds       = size(seeds, 2)
    @assert size(seeds, 1) == n_psi
    loglikelihood = Vector{Typ}(undef, n_seeds)
    psitilde      = Matrix{Typ}(undef, n_psi, n_seeds)
    optseeds      = Vector{Optim.OptimizationResults}(undef, n_seeds)

    # Optimization
    for i in 1:n_seeds
        try
            println("-------------------- Seed ", i, "--------------------")
            # optimize
            optseed = optimize(psi -> compute_likelihood(ln_F, T, D, psi, delta_t), seeds[:, i], LBFGS(), Optim.Options(f_tol = 1e-6, g_tol = 1e-6, show_trace = true))
            # allocate log_lik and minimizer
            loglikelihood[i] = -optseed.minimum
            psitilde[:, i] = optseed.minimizer
            optseeds[i] = optseed
        catch err
            println(err)
        end
    end

    # Query results
    log_lik, best_seed = findmax(loglikelihood)

    opt_param = psitilde[:, best_seed]

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

    return p, seeds[:, best_seed], optseeds[best_seed]
end

"""
    schwartzsmith(ln_F::Matrix{Typ}, T_V::Vector{Typ}, dates::Vector{Int64}, s::Int64; delta_t::Int = 1, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

Estimation of the Schwartz Smith model with a vector of average time to maturity as an input. Returns the estimated parameters.
"""
function schwartzsmith(ln_F::Matrix{Typ}, T_V::Vector{Typ}, dates::Vector{Int64}, s::Int64; delta_t::Int = 1, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

    # Parameters estimation
    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    p, seed, optseeds = schwartzsmith(ln_F, T, dates, s; delta_t = delta_t, seeds = seeds)

    return p, seed, optseeds
end


end
