"""
    sqrt_smoother(T::Matrix{Typ}, sqrt_f::SquareRootFilter{Typ}, p::SSParams{Typ}, delta_t64) where Typ

Square Root smoother without exogenous variables.
"""
function sqrt_smoother(T::Matrix{Typ}, sqrt_f::SquareRootFilter{Typ}, p::SSParams{Typ}, delta_t64) where Typ
    X_t = Vector{Float64}(undef, 0)
    n_exp = 0

    # Dimensions data
    n, prods = size(T)
    m        = size(sqrt_f.a_kf, 2)

    # Load filter data
    a_kf     = sqrt_f.a_kf
    v_kf     = sqrt_f.v_kf
    sqrtF_kf = sqrt_f.sqrtF_kf
    sqrtP_kf = sqrt_f.sqrtP_kf
    K_kf     = sqrt_f.K_kf

    # Smoothed state and its covariance
    alpha = Matrix{Typ}(undef, n, m)
    V     = Array{Typ, 3}(undef, m, m, n)
    L     = Array{Typ, 3}(undef, m, m, n)
    r     = Matrix{Typ}(undef, n, m)
    sqrtN = Array{Typ, 3}(undef, m, m, n)

    # Initialization
    sqrtN[:, :, end] = zeros(m, m)
    r[end, :]        = zeros(m, 1)

    for t = n:-1:2
        Z_kf = F(T[t, :], p, X_t)
        T_kf = G(p, n_exp, delta_t)
        d_kf = d(T[t, :], p)
        c_kf = c(p, n_exp, delta_t)

        L[:, :, t]  = T_kf - K_kf[:, :, t] * Z_kf
        r[t - 1, :] = Z_kf' * pinv(sqrtF_kf[:, :, t] * sqrtF_kf[:, :, t]') * v_kf[t, :] + L[:, :, t]' * r[t, :]
        Nstar = [Z_kf' * pinv(sqrtF_kf[:, :, t]) L[:, :, t]' * sqrtN[:, :, t]]

        # QR decomposition of auxiliary matrix Nstar
        NstarG             = Nstar * qr(Nstar').Q
        sqrtN[:, :, t - 1] = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        P = gram(sqrtP_kf[:, :, t])
        N = gram(sqrtN[:, :, t - 1])
        alpha[t, :] = a_kf[t, :] + P * r[t - 1, :]
        V[:, :, t] = P - (P * N * P)
    end

    Z_kf = F(T[1, :], p, X_t)
    T_kf = G(p, n_exp, delta_t)
    d_kf = d(T[1, :], p)
    c_kf = c(p, n_exp, delta_t)

    L[:, :, 1] = T_kf - K_kf[:, :, 1] * Z_kf
    r_0    = Z_kf' * pinv(gram(sqrtF_kf[:, :, 1])) * v_kf[1, :] + L[:, :, 1]' * r[1, :]
    Nstar  = [Z_kf' * pinv(sqrtF_kf[:, :, 1]) L[:, :, 1]' * sqrtN[:, :, 1]]
    G_sm   = qr(Nstar').Q
    NstarG = Nstar * G_sm

    sqrtN_0 = NstarG[1:m, 1:m]
    P_1     = gram(sqrtP_kf[:, :, 1])
    alpha[1, :] = a_kf[1, :] + P_1 * r_0
    V[:, :, 1]  = P_1 - (P_1 * gram(sqrtN_0) * P_1)

    return Smoother(alpha, V)
end

"""
    sqrt_smoother(T::Matrix{Typ}, X::Matrix{Typ}, sqrt_f::SquareRootFilter{Typ}, p::SSParams{Typ}, delta_t64) where Typ

Square Root smoother for when exogenous variables are included.
"""
function sqrt_smoother(T::Matrix{Typ}, X::Matrix, sqrt_f::SquareRootFilter{Typ}, p::SSParams{Typ}, delta_t64) where Typ
    n_exp = size(X, 2)

    # Dimensions data
    n, prods = size(T)
    m        = size(sqrt_f.a_kf, 2)

    # Load filter data
    a_kf     = sqrt_f.a_kf
    v_kf     = sqrt_f.v_kf
    sqrtF_kf = sqrt_f.sqrtF_kf
    sqrtP_kf = sqrt_f.sqrtP_kf
    K_kf     = sqrt_f.K_kf

    # Smoothed state and its covariance
    alpha = Matrix{Typ}(undef, n, m)
    V     = Array{Typ, 3}(undef, m, m, n)
    L     = Array{Typ, 3}(undef, m, m, n)
    r     = Matrix{Typ}(undef, n, m)
    sqrtN = Array{Typ, 3}(undef, m, m, n)

    # Initialization
    sqrtN[:, :, end] = zeros(m, m)
    r[end, :]        = zeros(m, 1)

    for t = n:-1:2
        Z_kf = F(T[t, :], p, X[t, :])
        T_kf = G(p, n_exp, delta_t)
        d_kf = d(T[t, :], p)
        c_kf = c(p, n_exp, delta_t)

        L[:, :, t]  = T_kf - K_kf[:, :, t] * Z_kf
        r[t - 1, :] = Z_kf' * pinv(sqrtF_kf[:, :, t] * sqrtF_kf[:, :, t]') * v_kf[t, :] + L[:, :, t]' * r[t, :]
        Nstar = [Z_kf' * pinv(sqrtF_kf[:, :, t]) L[:, :, t]' * sqrtN[:, :, t]]

        # QR decomposition of auxiliary matrix Nstar
        NstarG             = Nstar * qr(Nstar').Q
        sqrtN[:, :, t - 1] = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        P = gram(sqrtP_kf[:, :, t])
        N = gram(sqrtN[:, :, t - 1])
        alpha[t, :] = a_kf[t, :] + P * r[t - 1, :]
        V[:, :, t] = P - (P * N * P)
    end

    Z_kf = F(T[1, :], p, X[1, :])
    T_kf = G(p, n_exp, delta_t)
    d_kf = d(T[1, :], p)
    c_kf = c(p, n_exp, delta_t)

    L[:, :, 1] = T_kf - K_kf[:, :, 1] * Z_kf
    r_0    = Z_kf' * pinv(gram(sqrtF_kf[:, :, 1])) * v_kf[1, :] + L[:, :, 1]' * r[1, :]
    Nstar  = [Z_kf' * pinv(sqrtF_kf[:, :, 1]) L[:, :, 1]' * sqrtN[:, :, 1]]
    G_sm      = qr(Nstar').Q
    NstarG = Nstar * G_sm

    sqrtN_0 = NstarG[1:m, 1:m]
    P_1     = gram(sqrtP_kf[:, :, 1])
    alpha[1, :] = a_kf[1, :] + P_1 * r_0
    V[:, :, 1]  = P_1 - (P_1 * gram(sqrtN_0) * P_1)

    return Smoother(alpha, V)
end

mutable struct Smoother{T}
    alpha::Matrix{T}
    V::Array{T, 3}
end
