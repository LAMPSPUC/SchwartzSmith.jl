"""
    simulate(T::Matrix{Float64}, N::Int, S::Int, p::SSParams, f::Filter{Float64}; delta_t::Int = 1)

Simulate S future scenarios up to N steps ahead. Matrix of time to maturity as input.
"""
function simulate(T::Matrix{Float64}, N::Int, S::Int, p::SSParams, f::Filter{Float64}; delta_t::Int = 1)
    att_kf = f.att_kf

    n = size(att_kf, 1)
    prods = size(T, 2)
    X_t = Vector{Float64}(undef, 0)
    n_exp = 0

    # Covariance matrices
    Q = W(p, delta_t)
    H = V(p)

    # Distribution of the state space errors
    dist_ω = MvNormal(zeros(2), Q)
    dist_v = MvNormal(zeros(prods), H)

    y_sim = Array{Float64, 3}(undef, N, prods, S)
    x_sim = Array{Float64, 3}(undef, N, 2, S)

    for i in 1:S

        ω = rand(dist_ω, N)'
        v = rand(dist_v, N)'

        # Initial values
        x_sim[1, :, i] = c(p, n_exp, delta_t) + G(p, n_exp, delta_t) * att_kf[n, :] + R(n_exp) * ω[1, :]
        y_sim[1, :, i] = d(T[1, :], p) + F(T[1, :], p, X_t) * x_sim[1, :, i] + v[1, :]

        # Simulating future scenarios
        for t = 2:N
            x_sim[t, :, i] = c(p, n_exp, delta_t) + G(p, n_exp, delta_t) * x_sim[t-1, :, i] + R(n_exp) * ω[t, :]
            y_sim[t, :, i] = d(T[t, :], p) + F(T[t, :], p, X_t) * x_sim[t, :, i] + v[t, :]
        end
    end

    return x_sim, y_sim
end

"""
    simulate(T_V::Matrix{Float64}, N::Int, S::Int, p::SSParams, f::Filter{Float64}; delta_t_v::Int = 1)

Simulate S future scenarios up to N steps ahead. Vector of average time to maturity as input.
"""
function simulate(T_V::Vector{Float64}, N::Int, S::Int, p::SSParams, f::Filter{Float64}; delta_t_v::Int = 1)
    prods = length(T_V)
    T = Matrix{Float64}(undef, N, prods)

    # Representation of the time to maturity matrix
    for i in 1:N, j in 1:prods
        T[i, j] = T_V[j]
    end

    x_sim, y_sim = simulate(T, N, S, p, f; delta_t = delta_t_v)
end

"""
    simulate(T::Matrix{Float64}, X::VecOrMat, N::Int, S::Int, p::SSParams, f::Filter{Float64}; delta_t::Int = 1)

Simulate S future scenarios up to N steps ahead. Matrix of time to maturity and exogenous variables as input.
"""
function simulate(T::Matrix{Float64}, X::VecOrMat, N::Int, S::Int, p::SSParams, f::Filter{Float64}; delta_t::Int = 1)
    att_kf = f.att_kf

    n = size(att_kf, 1)
    prods = size(T, 2)

    X = X[:, :]
    n_exp = size(X, 2)

    # Covariance matrices
    Q = W(p, delta_t)
    H = V(p)

    # Distribution of the state space errors
    dist_ω = MvNormal(zeros(2), Q)
    dist_v = MvNormal(zeros(prods), H)

    y_sim = Array{Float64, 3}(undef, N, prods, S)
    x_sim = Array{Float64, 3}(undef, N, 2 + n_exp, S)

    for i in 1:S

        ω = rand(dist_ω, N)'
        v = rand(dist_v, N)'

        # Initial values
        x_sim[1, :, i] = c(p, n_exp, delta_t) + G(p, n_exp, delta_t) * att_kf[n, :] + R(n_exp) * ω[1, :]
        y_sim[1, :, i] = d(T[1, :], p) + F(T[1, :], p, X[1, :]) * x_sim[1, :, i] + v[1, :]

        # Simulating future scenarios
        for t = 2:N
            x_sim[t, :, i] = c(p, n_exp, delta_t) + G(p, n_exp, delta_t) * x_sim[t-1, :, i] + R(n_exp) * ω[t, :]
            y_sim[t, :, i] = d(T[t, :], p) + F(T[t, :], p, X[t, :]) * x_sim[t, :, i] + v[t, :]
        end
    end

    return x_sim, y_sim
end

"""
    simulate(T_V::Matrix{Float64}, X::VecOrMat, N::Int, S::Int, p::SSParams, f::Filter{Float64}; delta_t_v::Int = 1)

Simulate S future scenarios up to N steps ahead. Vector of average time to maturity and exogenous variables as input.
"""
function simulate(T_V::Vector{Float64}, X::VecOrMat, N::Int, S::Int, p::SSParams, f::Filter{Float64}; delta_t_v::Int = 1)
    prods = length(T_V)
    T = Matrix{Float64}(undef, N, prods)

    # Representation of the time to maturity matrix
    for i in 1:N, j in 1:prods
        T[i, j] = T_V[j]
    end

    x_sim, y_sim = simulate(T, X, N, S, p, f; delta_t = delta_t_v)
end
