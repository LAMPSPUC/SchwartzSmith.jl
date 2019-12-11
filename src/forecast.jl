"""
    forecast(f::Filter{Typ}, T::Matrix{Typ}, p::SSParams{Typ}, N::Int; delta_t = 1) where Typ

Returns the mean square error forecasts N steps ahead. Matrix of time to maturity as an input.
"""
function forecast(f::Filter{Typ}, T::Matrix{Typ}, p::SSParams{Typ}, N::Int; delta_t = 1) where Typ
    n, prods = size(T)
    D_t = Vector{Float64}(undef, 0)
    s = 0

    # Initial values
    a0 = f.att_kf[end, :]
    P0 = f.P_kf[:, :, end]
    F0 = f.F_kf[:, :, end]

    # State and variance forecasts
    a = Matrix{Typ}(undef, N, 2)
    P = Array{Typ, 3}(undef, 2, 2, N)
    F_f = Array{Typ, 3}(undef, prods, prods, N)

    # Probability distribution
    dist = Vector{Distribution}(undef, N)

    # Initialization
    a[1, :]    = G(p, s, delta_t) * a0 + c(p, s, delta_t)
    P[:, :, 1] = G(p, s, delta_t) * P0 * G(p, s, delta_t)' + R(s) * W(p, delta_t) * R(s)'
    F_f[:, :, 1] = F(T[1, :], p, D_t) * P[:, :, 1] * F(T[1, :], p, D_t)' + V(p)
    ensure_pos_sym!(F_f, 1)
    dist[1]    = MvNormal(vec(F(T[1, :], p, D_t) * a[1, :] + d(T[1, :], p)), F_f[:, :, 1])

    for t = 2:N
        a[t, :]    = G(p, s, delta_t) * a[t-1, :] + c(p, s, delta_t)
        P[:, :, t] = G(p, s, delta_t) * P[:, :, t-1] * G(p, s, delta_t)' + R(s) * W(p, delta_t) * R(s)'
        F_f[:, :, t] = F(T[t, :], p, D_t) * P[:, :, t] * F(T[t, :], p, D_t)' + V(p)
        ensure_pos_sym!(F_f, t)
        dist[t]    = MvNormal(vec(F(T[t, :], p, D_t) * a[t, :] + d(T[t, :], p)), F_f[:, :, t])
    end

    forec = Matrix{Typ}(undef, N, prods)
    for t = 1:N
        forec[t, :] = mean(dist[t])
    end

    return forec
end

"""
    forecast(f::Filter{Typ}, T_V::Vector{Typ}, p::SSParams{Typ}, N::Int; delta_t_v = 1) where Typ

Returns the mean square error forecasts N steps ahead. Vector of average time to maturity as input.
"""
function forecast(f::Filter{Typ}, T_V::Vector{Typ}, p::SSParams{Typ}, N::Int; delta_t_v = 1) where Typ
    prods = length(T_V)
    T = Matrix{Typ}(undef, N, prods)

    # Representation of the time to maturity matrix
    for i in 1:N, j in 1:prods
        T[i, j] = T_V[j]
    end

    forec = forecast(f, T, p, N; delta_t = delta_t_v)

    return forec
end

"""
    forecast(f::Filter{Typ}, T::Matrix{Typ}, dates::Vector{Int64}, s::Int64, p::SSParams{Typ}, N::Int; delta_t = 1) where Typ

Returns the mean square error forecasts N steps ahead. Matrix of time to maturity as an input.
"""
function forecast(f::Filter{Typ}, T::Matrix{Typ}, dates::Vector{Int64}, s::Int64, p::SSParams{Typ}, N::Int; delta_t = 1) where Typ
    n, prods = size(T)
    D = calc_D(s, dates)
    s = size(D, 2)
    
    # Initial values
    a0 = f.att_kf[end, :]
    P0 = f.P_kf[:, :, end]
    F0 = f.F_kf[:, :, end]

    # State and variance forecasts
    a = Matrix{Typ}(undef, N, 2 + s)
    P = Array{Typ, 3}(undef, 2 + s, 2 + s, N)
    F_f = Array{Typ, 3}(undef, prods, prods, N)

    # Probability distribution
    dist = Vector{Distribution}(undef, N)

    # Initialization
    a[1, :]    = G(p, s, delta_t) * a0 + c(p, s, delta_t)
    P[:, :, 1] = G(p, s, delta_t) * P0 * G(p, s, delta_t)' + R(s) * W(p, delta_t) * R(s)'
    F_f[:, :, 1] = F(T[1, :], p, D[1, :]) * P[:, :, 1] * F(T[1, :], p, D[1, :])' + V(p)
    ensure_pos_sym!(F_f, 1)
    dist[1]    = MvNormal(vec(F(T[1, :], p, D[1, :]) * a[1, :] + d(T[1, :], p)), F_f[:, :, 1])

    for t = 2:N
        a[t, :]    = G(p, s, delta_t) * a[t-1, :] + c(p, s, delta_t)
        P[:, :, t] = G(p, s, delta_t) * P[:, :, t-1] * G(p, s, delta_t)' + R(s) * W(p, delta_t) * R(s)'
        F_f[:, :, t] = F(T[t, :], p, D[t, :]) * P[:, :, t] * F(T[t, :], p, D[t, :])' + V(p)
        ensure_pos_sym!(F_f, t)
        dist[t]    = MvNormal(vec(F(T[t, :], p, D[t, :]) * a[t, :] + d(T[t, :], p)), F_f[:, :, t])
    end

    forec = Matrix{Typ}(undef, N, prods)
    for t = 1:N
        forec[t, :] = mean(dist[t])
    end

    return forec
end

"""
    forecast(f::Filter{Typ}, T_V::Vector{Typ}, dates::Vector{Int64}, s::Int64, p::SSParams{Typ}, N::Int; delta_t_v = 1) where Typ

Returns the mean square error forecasts N steps ahead. Vector of average time to maturity as input.
"""
function forecast(f::Filter{Typ}, T_V::Vector{Typ}, dates::Vector{Int64}, s::Int64, p::SSParams{Typ}, N::Int; delta_t_v = 1) where Typ
    prods = length(T_V)
    T = Matrix{Typ}(undef, N, prods)

    # Representation of the time to maturity matrix
    for i in 1:N, j in 1:prods
        T[i, j] = T_V[j]
    end

    forec = forecast(f, T, dates, s, p, N; delta_t = delta_t_v)

    return forec
end
