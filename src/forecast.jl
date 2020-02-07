"""
    forecast(T::Matrix{Typ}, N::Int, p::SSParams{Typ}, f::Filter{Typ}; delta_t) where Typ

Returns the forecasts N steps ahead. Matrix of time to maturity as an input.
"""
function forecast(T::Matrix{Typ}, N::Int, p::SSParams{Typ}, f::Filter{Typ}; delta_t = 1) where Typ
    prods = size(T, 2)
    X_t = Vector{Float64}(undef, 0)
    n_exp = 0

    F_kf = gram_in_time(f.sqrt_f.sqrtF_kf)
    P_kf = gram_in_time(f.sqrt_f.sqrtP_kf)

    # Initial values
    a0 = f.att_kf[end, :]
    P0 = P_kf[:, :, end]
    F0 = F_kf[:, :, end]

    # State and variance forecasts
    a = Matrix{Typ}(undef, N, 2)
    P = Array{Typ, 3}(undef, 2, 2, N)
    F_f = Array{Typ, 3}(undef, prods, prods, N)

    # Probability distribution
    dist = Vector{Distribution}(undef, N)

    # Initialization
    a[1, :]    = G(p, n_exp, delta_t) * a0 + c(p, n_exp, delta_t)
    P[:, :, 1] = G(p, n_exp, delta_t) * P0 * G(p, n_exp, delta_t)' + R(n_exp) * W(p, delta_t) * R(n_exp)'
    F_f[:, :, 1] = F(T[1, :], p, X_t) * P[:, :, 1] * F(T[1, :], p, X_t)' + V(p)
    ensure_pos_sym!(F_f, 1)
    dist[1]    = MvNormal(vec(F(T[1, :], p, X_t) * a[1, :] + d(T[1, :], p)), F_f[:, :, 1])

    for t = 2:N
        a[t, :]    = G(p, n_exp, delta_t) * a[t-1, :] + c(p, n_exp, delta_t)
        P[:, :, t] = G(p, n_exp, delta_t) * P[:, :, t-1] * G(p, n_exp, delta_t)' + R(n_exp) * W(p, delta_t) * R(n_exp)'
        F_f[:, :, t] = F(T[t, :], p, X_t) * P[:, :, t] * F(T[t, :], p, X_t)' + V(p)
        ensure_pos_sym!(F_f, t)
        dist[t]    = MvNormal(vec(F(T[t, :], p, X_t) * a[t, :] + d(T[t, :], p)), F_f[:, :, t])
    end

    forec = Matrix{Typ}(undef, N, prods)
    for t = 1:N
        forec[t, :] = mean(dist[t])
    end

    return forec
end

"""
    forecast(T_V::Matrix{Typ}, N::Int, p::SSParams{Typ}, f::Filter{Typ}; delta_t_v = 1) where Typ

Returns the forecasts N steps ahead. Vector of time to maturity as an input.
"""
function forecast(T_V::Vector{Typ}, N::Int, p::SSParams{Typ}, f::Filter{Typ}; delta_t_v = 1) where Typ
    prods = length(T_V)
    T = Matrix{Typ}(undef, N, prods)

    # Representation of the time to maturity matrix
    for i in 1:N, j in 1:prods
        T[i, j] = T_V[j]
    end

    forec = forecast(T, N, p, f; delta_t = delta_t_v)

    return forec
end

"""
    forecast(T::Matrix{Typ}, X::VecOrMat, N::Int, p::SSParams{Typ}, f::Filter{Typ}; delta_t) where Typ

Returns the forecasts N steps ahead. Matrix of time to maturity and exogenous variables as input.
"""
function forecast(T::Matrix{Typ}, X::VecOrMat, N::Int, p::SSParams{Typ}, f::Filter{Typ}; delta_t = 1) where Typ
    prods = size(T, 2)

    X = X[:,:]
    n_exp = size(X, 2)

    F_kf = gram_in_time(f.sqrt_f.sqrtF_kf)
    P_kf = gram_in_time(f.sqrt_f.sqrtP_kf)

    # Initial values
    a0 = f.att_kf[end, :]
    P0 = P_kf[:, :, end]
    F0 = F_kf[:, :, end]

    # State and variance forecasts
    a = Matrix{Typ}(undef, N, 2 + n_exp)
    P = Array{Typ, 3}(undef, 2 + n_exp, 2 + n_exp, N)
    F_f = Array{Typ, 3}(undef, prods, prods, N)

    # Probability distribution
    dist = Vector{Distribution}(undef, N)

    # Initialization
    a[1, :]    = G(p, n_exp, delta_t) * a0 + c(p, n_exp, delta_t)
    P[:, :, 1] = G(p, n_exp, delta_t) * P0 * G(p, n_exp, delta_t)' + R(n_exp) * W(p, delta_t) * R(n_exp)'
    F_f[:, :, 1] = F(T[1, :], p, X[1, :]) * P[:, :, 1] * F(T[1, :], p, X[1, :])' + V(p)
    ensure_pos_sym!(F_f, 1)
    dist[1]    = MvNormal(vec(F(T[1, :], p, X[1, :]) * a[1, :] + d(T[1, :], p)), F_f[:, :, 1])

    for t = 2:N
        a[t, :]    = G(p, n_exp, delta_t) * a[t-1, :] + c(p, n_exp, delta_t)
        P[:, :, t] = G(p, n_exp, delta_t) * P[:, :, t-1] * G(p, n_exp, delta_t)' + R(n_exp) * W(p, delta_t) * R(n_exp)'
        F_f[:, :, t] = F(T[t, :], p, X[t, :]) * P[:, :, t] * F(T[t, :], p, X[t, :])' + V(p)
        ensure_pos_sym!(F_f, t)
        dist[t]    = MvNormal(vec(F(T[t, :], p, X[t, :]) * a[t, :] + d(T[t, :], p)), F_f[:, :, t])
    end

    forec = Matrix{Typ}(undef, N, prods)
    for t = 1:N
        forec[t, :] = mean(dist[t])
    end

    return forec
end

"""
    forecast(T_V::Matrix{Typ}, X::VecOrMat, N::Int, p::SSParams{Typ}, f::Filter{Typ}; delta_t_v = 1) where Typ

Returns the forecasts N steps ahead. Vector of time to maturity and exogenous variables as input.
"""
function forecast(T_V::Vector{Typ}, X::VecOrMat, N::Int, p::SSParams{Typ}, f::Filter{Typ}; delta_t_v = 1) where Typ
    prods = length(T_V)
    T = Matrix{Typ}(undef, N, prods)

    # Representation of the time to maturity matrix
    for i in 1:N, j in 1:prods
        T[i, j] = T_V[j]
    end

    forec = forecast(T, X, N, p, f; delta_t = delta_t_v)

    return forec
end
