"""
    sqrt_kalman_filter(ln_F::Matrix{Typ}, T::Matrix{Typ}, p::SSParams, delta_t::Int) where Typ

Square Root Kalman Filter.
"""
function sqrt_kalman_filter(ln_F::Matrix{Typ}, T::Matrix{Typ}, p::SSParams, delta_t::Int) where Typ

    n, prods = size(ln_F)

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

    ensure_pos_sym!(W(p, delta_t))

    sqrt_Q = cholesky(W(p, delta_t)).L
    sqrt_H = cholesky(V(p)).L

    # Pre-allocating for performance
    zeros_pr = zeros(prods, 2)
    zeros_mp = zeros(2, prods)
    range1   = (prods + 1):(prods + 2)
    range2   = 1:prods
    sqrtH_zeros_pr  = [sqrt_H zeros_pr]
    zeros_mp_RsqrtQ = [zeros_mp sqrt_Q]

    # Square-root Kalman filter
    for t = 1:n
        Z_kf = F(T[t, :], p)
        T_kf = G(p, delta_t)
        d_kf = d(T[t, :], p)
        c_kf = c(p, delta_t)

        v_kf[t, :] = ln_F[t, :] - Z_kf * a_kf[t, :] - d_kf
        # Manipulation of auxiliary matrices
        U         = [Z_kf * sqrtP_kf[:, :, t] sqrtH_zeros_pr;
                     T_kf * sqrtP_kf[:, :, t] zeros_mp_RsqrtQ]
        G_kf      = qr(Matrix(U')).Q
        Ustar     = U*G_kf
        U2star[:, :, t] = Ustar[range1, range2]
        sqrtF_kf[:, :, t]  = Ustar[range2, range2]

        # Kalman gain and predictive state update
        K_kf[:, :, t]       = U2star[:, :, t]*inv(sqrtF_kf[:, :, t])
        a_kf[t+1, :]        = T_kf*a_kf[t, :] + K_kf[:, :, t]*v_kf[t, :] + c_kf
        sqrtP_kf[:, :, t+1] = Ustar[range1, range1]
    end

    F_kf = gram_in_time(sqrtF_kf)
    ensure_pos_sym!(F_kf)

    return v_kf, F_kf
end

function gram_in_time(mat::Array{T, 3}) where T
    gram_in_time = similar(mat)
    @inbounds @views for t = 1:size(gram_in_time, 3)
        gram_in_time[:, :, t] = gram(mat[:, :, t])
    end
    return gram_in_time
end

function gram(mat::AbstractArray{T}) where T
    return mat*mat'
end
