"""
    estimated_prices_states(p::SSParams{Typ}, T::Matrix{Typ}, ln_F::Matrix{Typ}; delta_t::Int = 1) where Typ

Returns the prices and the kalman filter struct estimated by the model. Matrix of time to maturity as an input.
"""
function estimated_prices_states(p::SSParams{Typ}, T::Matrix{Typ}, ln_F::Matrix{Typ}; delta_t::Int = 1) where Typ
    n, prods = size(T)
    y = Array{Typ, 2}(undef, n, prods)
    D_t = Vector{Float64}(undef, 0)

    f = kalman_filter(ln_F, T, p, delta_t)

    for t in 1:n
        y[t, :] = d(T[t, :], p) + F(T[t, :], p, D_t) * f.att_kf[t, :]
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

"""
    estimated_prices_states(p::SSParams{Typ}, T::Matrix{Typ}, ln_F::Matrix{Typ}, dates::Vector{Int64}, s::Int64; delta_t::Int = 1) where Typ

Returns the prices and the kalman filter struct estimated by the model. Matrix of time to maturity as an input.
"""
function estimated_prices_states(p::SSParams{Typ}, T::Matrix{Typ}, ln_F::Matrix{Typ}, dates::Vector{Int64}, s::Int64; delta_t::Int = 1) where Typ
    D = calc_D(s, dates)

    n, prods = size(T)
    y = Array{Typ, 2}(undef, n, prods)

    f = kalman_filter(ln_F, T, D, p, delta_t)

    for t in 1:n
        y[t, :] = d(T[t, :], p) + F(T[t, :], p, D[t, :]) * f.att_kf[t, :]
    end

    return y, f
end

"""
    estimated_prices_states(p::SSParams{Typ}, T_V::Vector{Typ}, ln_F::Matrix{Typ}, dates::Vector{Int64}, s::Int64; delta_t::Int = 1) where Typ

Returns the prices and the kalman filter struct estimated by the model. Vector of average time to maturity as an input.
"""
function estimated_prices_states(p::SSParams{Typ}, T_V::Vector{Typ}, ln_F::Matrix{Typ}, dates::Vector{Int64}, s::Int64; delta_t::Int = 1) where Typ
    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    y, f = estimated_prices_states(p, T, ln_F, dates, s; delta_t = delta_t)

    return y, f
end
