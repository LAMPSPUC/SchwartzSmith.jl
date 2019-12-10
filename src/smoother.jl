"""
    smoother(ln_F::Matrix{Typ}, T::Matrix{Typ}, f::Filter{Typ}, p::SSParams{Typ}, D::Matrix{Float64}, delta_t::Int64) where Typ

Function to obtain the smoothed states when seasonality is included.
"""
function smoother(ln_F::Matrix{Typ}, T::Matrix{Typ}, f::Filter{Typ}, p::SSParams{Typ}, D::Array{Float64, 3}, delta_t::Int64) where Typ
    # Dimensions data
    n, prods = size(ln_F)
    m = size(f.att_kf, 2)
    s = size(D, 3)

    # Load filter data
    a_kf = f.a_kf
    v_kf = f.v_kf
    F_kf = f.F_kf
    P_kf = f.P_kf

    # Smoothed state and its covariance
    alpha = Matrix{Typ}(undef, n, m)
    V     = Array{Typ, 3}(undef, m, m, n)
    L     = Array{Typ, 3}(undef, m, m, n)
    r     = Matrix{Typ}(undef, n, m)
    N     = Array{Typ}(undef, m, m, n)

    # Initialization
    N[:, :, end] = zeros(m, m)
    r[end, :]    = zeros(m, 1)

    for t = n:-1:2
        Z_kf = F(T[t, :], p, D[:, t, :])
        T_kf = G(p, s, delta_t)
        d_kf = d(T[t, :], p)
        c_kf = c(p, s, delta_t)
        K    = T_kf * P_kf[:, :, t] * Z_kf' * pinv(F_kf[:, :, t])

        Z_transp_invF  = Z_kf' * pinv(F_kf[:, :, t])
        L[:, :, t]     = T_kf - K * Z_kf
        r[t - 1, :]    = Z_transp_invF * v_kf[t, :] + L[:, :, t]' * r[t, :]
        N[:, :, t - 1] = Z_transp_invF * Z_kf + L[:, :, t]' * N[:, :, t] * L[:, :, t]

        alpha[t, :] = a_kf[t, :] + P_kf[:, :, t] * r[t - 1, :]
        V[:, :, t]  = P_kf[:, :, t] - P_kf[:, :, t] * N[:, :, t - 1] * P_kf[:, :, t]
    end

    # Last iteration
    Z_kf = F(T[1, :], p, D[:, 1, :])
    T_kf = G(p, s, delta_t)
    d_kf = d(T[1, :], p)
    c_kf = c(p, s, delta_t)
    K    = T_kf * P_kf[:, :, 1] * Z_kf' * pinv(F_kf[:, :, 1])

    Z_transp_invF = Z_kf' * pinv(F_kf[:, :, 1])
    L[:, :, 1]    = T_kf - K * Z_kf
    r0            = Z_transp_invF * v_kf[1, :] + L[:, :, 1]' * r[1, :]
    N0            = Z_transp_invF * Z_kf + L[:, :, 1]' * N[:, :, 1] * L[:, :, 1]

    alpha[1, :] = a_kf[1, :] + P_kf[:, :, 1] * r0
    V[:, :, 1]  = P_kf[:, :, 1] - P_kf[:, :, 1] * N0 * P_kf[:, :, 1]

    return Smoother(alpha, V)
end

"""
    smoother(ln_F::Matrix{Typ}, T::Matrix{Typ}, f::Filter{Typ}, p::SSParams{Typ}, delta_t::Int64) where Typ

Function to obtain the smoothed states without seasonality.
"""
function smoother(ln_F::Matrix{Typ}, T::Matrix{Typ}, f::Filter{Typ}, p::SSParams{Typ}, delta_t::Int64) where Typ
    D_t = Matrix{Float64}(undef, 0, 0)
    s = 0

    # Dimensions data
    n, prods = size(ln_F)
    m = size(f.att_kf, 2)

    # Load filter data
    a_kf = f.a_kf
    v_kf = f.v_kf
    F_kf = f.F_kf
    P_kf = f.P_kf

    # Smoothed state and its covariance
    alpha = Matrix{Typ}(undef, n, m)
    V     = Array{Typ, 3}(undef, m, m, n)
    L     = Array{Typ, 3}(undef, m, m, n)
    r     = Matrix{Typ}(undef, n, m)
    N     = Array{Typ}(undef, m, m, n)

    # Initialization
    N[:, :, end] = zeros(m, m)
    r[end, :]    = zeros(m, 1)

    for t = n:-1:2
        Z_kf = F(T[t, :], p, D_t)
        T_kf = G(p, s, delta_t)
        d_kf = d(T[t, :], p)
        c_kf = c(p, s, delta_t)
        K    = T_kf * P_kf[:, :, t] * Z_kf' * pinv(F_kf[:, :, t])

        Z_transp_invF  = Z_kf' * pinv(F_kf[:, :, t])
        L[:, :, t]     = T_kf - K * Z_kf
        r[t - 1, :]    = Z_transp_invF * v_kf[t, :] + L[:, :, t]' * r[t, :]
        N[:, :, t - 1] = Z_transp_invF * Z_kf + L[:, :, t]' * N[:, :, t] * L[:, :, t]

        alpha[t, :] = a_kf[t, :] + P_kf[:, :, t] * r[t - 1, :]
        V[:, :, t]  = P_kf[:, :, t] - P_kf[:, :, t] * N[:, :, t - 1] * P_kf[:, :, t]
    end

    # Last iteration
    Z_kf = F(T[1, :], p, D_t)
    T_kf = G(p, s, delta_t)
    d_kf = d(T[1, :], p)
    c_kf = c(p, s, delta_t)
    K    = T_kf * P_kf[:, :, 1] * Z_kf' * pinv(F_kf[:, :, 1])

    Z_transp_invF = Z_kf' * pinv(F_kf[:, :, 1])
    L[:, :, 1]    = T_kf - K * Z_kf
    r0            = Z_transp_invF * v_kf[1, :] + L[:, :, 1]' * r[1, :]
    N0            = Z_transp_invF * Z_kf + L[:, :, 1]' * N[:, :, 1] * L[:, :, 1]

    alpha[1, :] = a_kf[1, :] + P_kf[:, :, 1] * r0
    V[:, :, 1]  = P_kf[:, :, 1] - P_kf[:, :, 1] * N0 * P_kf[:, :, 1]

    return Smoother(alpha, V)
end

mutable struct Smoother{T}
    alpha::Matrix{T}
    V::Array{T, 3}
end
