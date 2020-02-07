"""
    estimated_prices(n::Int64, T::Matrix{Typ}, p::SSParams{Typ}; delta_t) where Typ

Returns the prices estimated by the model, the Square Root Kalman Filter and smoother results. Matrix of time to maturity as an input.
"""
function estimated_prices(ln_F::VecOrMat{Typ}, T::Matrix{Typ}, p::SSParams{Typ}; delta_t) where Typ
    ln_F = ln_F[:, :]

    n, prods = size(ln_F)
    y     = Array{Typ, 2}(undef, n, prods)
    X_t   = Vector{Float64}(undef, 0)

    sqrt_f          = sqrt_kalman_filter(ln_F, T, p, delta_t)
    smoother        = sqrt_smoother(T, sqrt_f, p, delta_t)
    filtered_states = sqrt_filtered_state(T, sqrt_f, p, delta_t)

    for t in 1:n
        y[t, :] = d(T[t, :], p) + F(T[t, :], p, X_t) * smoother.alpha[t, :]
    end

    filter = Filter(sqrt_f, filtered_states.att_kf, filtered_states.Ptt_kf)

    return y, filter, smoother
end

"""
    estimated_prices(n::Int64, T_V::Vector{Typ}, p::SSParams{Typ}; delta_t) where Typ

Returns the prices estimated by the model, the Square Root Kalman Filter and smoother results. Vector of time to maturity as an input.
"""
function estimated_prices(ln_F::VecOrMat{Typ}, T_V::Vector{Typ}, p::SSParams{Typ}; delta_t) where Typ
    ln_F = ln_F[:, :]

    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    y, filter, smoother = estimated_prices(ln_F, T, p; delta_t = delta_t)

    return y, filter, smoother
end

"""
    estimated_prices(n::Int64, T::Matrix{Typ}, X::VecOrMat, p::SSParams{Typ}; delta_t) where Typ

Returns the prices estimated by the model, the Square Root Kalman Filter and smoother results. Matrix of time to maturity and exogenous variables as input.
"""
function estimated_prices(ln_F::VecOrMat{Typ}, T::Matrix{Typ}, X::VecOrMat, p::SSParams{Typ}; delta_t) where Typ
    ln_F = ln_F[:, :]

    n, prods = size(ln_F)
    y = Array{Typ, 2}(undef, n, prods)
    X = X[:,:]

    sqrt_f          = sqrt_kalman_filter(ln_F, T, X, p, delta_t)
    smoother        = sqrt_smoother(T, X, sqrt_f, p, delta_t)
    filtered_states = sqrt_filtered_state(T, X, sqrt_f, p, delta_t)

    for t in 1:n
        y[t, :] = d(T[t, :], p) + F(T[t, :], p, X[t, :]) * smoother.alpha[t, :]
    end

    filter = Filter(sqrt_f, filtered_states.att_kf, filtered_states.Ptt_kf)

    return y, filter, smoother
end

"""
    estimated_prices(n::Int64, T_V::Vector{Typ}, X::VecOrMat, p::SSParams{Typ}; delta_t) where Typ

Returns the prices estimated by the model, the Square Root Kalman Filter and smoother results. Vector of time to maturity and exogenous variables as input.
"""
function estimated_prices(ln_F::VecOrMat{Typ}, T_V::Vector{Typ}, X::VecOrMat, p::SSParams{Typ}; delta_t) where Typ
    ln_F = ln_F[:, :]

    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    y, filter, smoother = estimated_prices(ln_F, T, X, p; delta_t = delta_t)

    return y, filter, smoother
end
