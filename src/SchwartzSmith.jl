module SchwartzSmith

using Optim
using LinearAlgebra
using Distributions

export schwartzsmith, estimated_prices, simulate, forecast, calc_seed, sqrt_kalman_filter, calc_D

include("auxiliar.jl")
include("create_matrices.jl")
include("sqrt_kalman_filter.jl")
include("sqrt_filtered_state.jl")
include("sqrt_smoother.jl")
include("estimated_prices.jl")
include("estimation.jl")
include("forecast.jl")
include("simulation.jl")

"""
    schwartzsmith(ln_F::Matrix{Typ}, T::Matrix{Typ}; delta_t, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

Estimation of the Schwartz Smith model with a matrix of time to maturity as an input. Returns the estimated parameters.
"""
function schwartzsmith(ln_F::VecOrMat{Typ}, T::Matrix{Typ}; delta_t, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ
    ln_F = ln_F[:, :]

    # Alocate memory
    n, prods = size(ln_F)
    n_exp = 0 # No exogenous variables
    X = Matrix{Float64}(undef, 0, 0)

    n_psi         = 7 + prods
    n_seeds       = size(seeds, 2)
    @assert size(seeds, 1) == n_psi
    loglikelihood = Vector{Typ}(undef, 0)
    psitilde_temp = Vector{Vector{Float64}}(undef, 0)
    optseeds      = Vector{Optim.OptimizationResults}(undef, 0)

    all_loglik = Vector{Float64}(undef, 0)

    # Optimization
    for i in 1:n_seeds
        try
            println("-------------------- Seed ", i, "--------------------")
            # optimize
            optseed = optimize(psi -> compute_likelihood(ln_F, T, X, psi, delta_t), seeds[:, i], LBFGS(), Optim.Options(f_tol = 1e-6, g_tol = 1e-6, show_trace = true))
            push!(loglikelihood, -optseed.minimum)
            push!(psitilde_temp, optseed.minimizer)
            push!(optseeds, optseed)
            push!(all_loglik, -optseed.minimum)
        catch err
            println(err)
        end
    end

    psitilde = Matrix{Typ}(undef, n_psi, length(psitilde_temp))

    for i in 1:length(psitilde_temp)
        psitilde[:, i] = psitilde_temp[i]
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

    return p, seeds[:, best_seed], optseeds[best_seed], psitilde, all_loglik
end

"""
    schwartzsmith(ln_F::Matrix{Typ}, T_V::Vector{Typ}; delta_t, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

Estimation of the Schwartz Smith model with a vector of time to maturity as an input. Returns the estimated parameters.
"""
function schwartzsmith(ln_F::VecOrMat{Typ}, T_V::Vector{Typ}; delta_t, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ
    ln_F = ln_F[:, :]

    # Parameters estimation
    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    p, seed, optseeds, psitilde, all_loglik = schwartzsmith(ln_F, T; delta_t = delta_t, seeds = seeds)

    return p, seed, optseeds, psitilde, all_loglik
end

"""
    schwartzsmith(ln_F::Matrix{Typ}, T::Matrix{Typ}, X::VecOrMat; delta_t, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

Estimation of the Schwartz Smith model with a matrix of time to maturity and exogenous variables as input. Returns the estimated parameters.
"""
function schwartzsmith(ln_F::VecOrMat{Typ}, T::Matrix{Typ}, X::VecOrMat; delta_t, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ
    ln_F = ln_F[:, :]

    # Alocate memory
    n, prods = size(ln_F)

    X = X[:,:]
    n_exp    = size(X, 2)

    n_psi         = 7 + prods
    n_seeds       = size(seeds, 2)
    @assert size(seeds, 1) == n_psi
    loglikelihood = Vector{Typ}(undef, 0)
    psitilde_temp = Vector{Vector{Float64}}(undef, 0)
    optseeds      = Vector{Optim.OptimizationResults}(undef, 0)

    all_loglik = Vector{Float64}(undef, 0)

    # Optimization
    for i in 1:n_seeds
        try
            println("-------------------- Seed ", i, "--------------------")
            # optimize
            optseed = optimize(psi -> compute_likelihood(ln_F, T, X, psi, delta_t), seeds[:, i], LBFGS(), Optim.Options(f_tol = 1e-6, g_tol = 1e-6, show_trace = true))
            push!(loglikelihood, -optseed.minimum)
            push!(psitilde_temp, optseed.minimizer)
            push!(optseeds, optseed)
            push!(all_loglik, -optseed.minimum)
        catch err
            println(err)
        end
    end

    psitilde = Matrix{Typ}(undef, n_psi, length(psitilde_temp))

    for i in 1:length(psitilde_temp)
        psitilde[:, i] = psitilde_temp[i]
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

    return p, seeds[:, best_seed], optseeds[best_seed], psitilde, all_loglik
end

"""
    schwartzsmith(ln_F::Matrix{Typ}, T_V::Vector{Typ}, X::VecOrMat; delta_t, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ

Estimation of the Schwartz Smith model with a vector of time to maturity and exogenous variables as input. Returns the estimated parameters.
"""
function schwartzsmith(ln_F::VecOrMat{Typ}, T_V::Vector{Typ}, X::VecOrMat; delta_t, seeds::VecOrMat{Typ} = calc_seed(ln_F, 10)) where Typ
    ln_F = ln_F[:, :]

    # Parameters estimation
    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    p, seed, optseeds, psitilde, all_loglik = schwartzsmith(ln_F, T, X; delta_t = delta_t, seeds = seeds)

    return p, seed, optseeds, psitilde, all_loglik
end


end
