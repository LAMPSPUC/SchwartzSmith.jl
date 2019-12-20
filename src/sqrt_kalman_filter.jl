"""
    sqrt_kalman_filter(ln_F::Matrix{Typ}, T::Matrix{Typ}, p::SSParams, delta_t::Int) where Typ

Square Root Kalman Filter with a matrix of time to maturity as input.
"""
function sqrt_kalman_filter(ln_F::Matrix{Typ}, T::Matrix{Typ}, p::SSParams, delta_t::Int) where Typ
    n, prods = size(ln_F)
    X_t = Vector{Float64}(undef, 0)
    n_exp = 0

    # Predictive state and its covariance matrix
    a_kf     = Matrix{Typ}(undef, n+1, 2)
    sqrtP_kf = Array{Typ, 3}(undef, 2, 2, n+1)

    # Innovation and its sqrt-covariance
    v_kf     = Matrix{Typ}(undef, n, prods)
    sqrtF_kf = Array{Typ, 3}(undef, prods, prods, n)
    K_kf     = Array{Typ, 3}(undef, 2, prods, n)

    # One-step forecast (inovation) error and its covariance matrix
    v_kf = Matrix{Typ}(undef, n, prods)
    F_kf = Array{Typ, 3}(undef, prods, prods, n)

    # Auxiliary matrices
    U2star = Array{Typ, 3}(undef, 2, prods, n)

    # Initialization
    a_kf[1, :]    = zeros(2, 1)
    sqrtP_kf[:, :, 1] = 1e1 .* Matrix(I, 2, 2)

    sqrt_Q = cholesky(W(p, delta_t)).L
    sqrt_H = cholesky(V(p)).L

    # Pre-allocating for performance
    zeros_pr = zeros(prods, 2)
    zeros_mp = zeros(2, prods)
    range1   = (prods + 1):(prods + 2)
    range2   = 1:prods
    sqrtH_zeros_pr  = [sqrt_H zeros_pr]
    zeros_mp_RsqrtQ = [zeros_mp R(n_exp)*sqrt_Q]

    # Square-root Kalman filter
    for t = 1:n
        Z_kf = F(T[t, :], p, X_t)
        T_kf = G(p, n_exp, delta_t)
        d_kf = d(T[t, :], p)
        c_kf = c(p, n_exp, delta_t)

        v_kf[t, :] = ln_F[t, :] - Z_kf * a_kf[t, :] - d_kf
        # Manipulation of auxiliary matrices
        U         = [Z_kf * sqrtP_kf[:, :, t] sqrtH_zeros_pr;
                     T_kf * sqrtP_kf[:, :, t] zeros_mp_RsqrtQ]
        G_kf      = qr(Matrix(U')).Q
        Ustar     = U*G_kf
        U2star[:, :, t] = Ustar[range1, range2]
        sqrtF_kf[:, :, t]  = Ustar[range2, range2]

        # Kalman gain and predictive state update
        K_kf[:, :, t]       = U2star[:, :, t]*pinv(sqrtF_kf[:, :, t])
        a_kf[t+1, :]        = T_kf*a_kf[t, :] + K_kf[:, :, t]*v_kf[t, :] + c_kf
        sqrtP_kf[:, :, t+1] = Ustar[range1, range1]
    end

    return SquareRootFilter(a_kf, v_kf, sqrtP_kf, sqrtF_kf, K_kf)
end

"""
    sqrt_kalman_filter(ln_F::Matrix{Typ}, T_V::Vector{Typ}, p::SSParams, delta_t::Int) where Typ

Square Root Kalman Filter with a vector of maturity as an input.
"""
function sqrt_kalman_filter(ln_F::Matrix{Typ}, T_V::Vector{Typ}, p::SSParams, delta_t::Int) where Typ
    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    sqrt_f = sqrt_kalman_filter(ln_F, T, p, delta_t)

    return sqrt_f
end

"""
    sqrt_kalman_filter(ln_F::Matrix{Typ}, T::Matrix{Typ}, X::VecOrMat, p::SSParams, delta_t::Int) where Typ

Square Root Kalman Filter with a matrix of time to maturity and exogenous variables as input.
"""
function sqrt_kalman_filter(ln_F::Matrix{Typ}, T::Matrix{Typ}, X::VecOrMat, p::SSParams, delta_t::Int) where Typ
    X = X[:, :]

    n, prods = size(ln_F)
    n_exp = size(X, 2)

    # Predictive state and its covariance matrix
    a_kf     = Matrix{Typ}(undef, n+1, 2 + n_exp)
    sqrtP_kf = Array{Typ, 3}(undef, 2 + n_exp, 2 + n_exp, n+1)

    # Innovation and its sqrt-covariance
    v_kf     = Matrix{Typ}(undef, n, prods)
    sqrtF_kf = Array{Typ, 3}(undef, prods, prods, n)
    K_kf     = Array{Typ, 3}(undef, 2 + n_exp, prods, n)

    # One-step forecast (inovation) error and its covariance matrix
    v_kf = Matrix{Typ}(undef, n, prods)
    F_kf = Array{Typ, 3}(undef, prods, prods, n)

    # Auxiliary matrices
    U2star = Array{Typ, 3}(undef, 2 + n_exp, prods, n)

    # Initialization
    a_kf[1, :]    = zeros(2 + n_exp, 1)
    sqrtP_kf[:, :, 1] = 1e1 .* Matrix(I, 2 + n_exp, 2 + n_exp)

    sqrt_Q = cholesky(W(p, delta_t)).L
    sqrt_H = cholesky(V(p)).L

    # Pre-allocating for performance
    zeros_pr = zeros(prods, 2)
    zeros_mp = zeros(2 + n_exp, prods)
    range1   = (prods + 1):(prods + 2 + n_exp)
    range2   = 1:prods
    sqrtH_zeros_pr  = [sqrt_H zeros_pr]
    zeros_mp_RsqrtQ = [zeros_mp R(n_exp)*sqrt_Q]

    # Square-root Kalman filter
    for t = 1:n
        Z_kf = F(T[t, :], p, X[t, :])
        T_kf = G(p, n_exp, delta_t)
        d_kf = d(T[t, :], p)
        c_kf = c(p, n_exp, delta_t)

        v_kf[t, :] = ln_F[t, :] - Z_kf * a_kf[t, :] - d_kf
        # Manipulation of auxiliary matrices
        U         = [Z_kf * sqrtP_kf[:, :, t] sqrtH_zeros_pr;
                     T_kf * sqrtP_kf[:, :, t] zeros_mp_RsqrtQ]
        G_kf      = qr(Matrix(U')).Q
        Ustar     = U*G_kf
        U2star[:, :, t] = Ustar[range1, range2]
        sqrtF_kf[:, :, t]  = Ustar[range2, range2]

        # Kalman gain and predictive state update
        K_kf[:, :, t]       = U2star[:, :, t]*pinv(sqrtF_kf[:, :, t])
        a_kf[t+1, :]        = T_kf*a_kf[t, :] + K_kf[:, :, t]*v_kf[t, :] + c_kf
        sqrtP_kf[:, :, t+1] = Ustar[range1, range1]
    end

    return SquareRootFilter(a_kf, v_kf, sqrtP_kf, sqrtF_kf, K_kf)
end

"""
    sqrt_kalman_filter(ln_F::Matrix{Typ}, T_V::Vector{Typ}, X::VecOrMat, p::SSParams, delta_t::Int) where Typ

Square Root Kalman Filter with a vector of time to maturity and exogenous variables as input.
"""
function sqrt_kalman_filter(ln_F::Matrix{Typ}, T_V::Vector{Typ}, X::VecOrMat, p::SSParams, delta_t::Int) where Typ
    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    sqrt_f = sqrt_kalman_filter(ln_F, T, X, p, delta_t)

    return sqrt_f
end

"""
    SquareRootFilter{T}

Struct with the Square Root Kalman Filter results.
"""
mutable struct SquareRootFilter{T}
    a_kf::Matrix{T}
    v_kf::Matrix{T}
    sqrtP_kf::Array{T, 3}
    sqrtF_kf::Array{T, 3}
    K_kf::Array{T, 3}
end
