"""
    kalman_filter(ln_F::Matrix{Typ}, T::Matrix{Typ}, p::SSParams{Typ}, delta_t::Int) where Typ

Definition of the Kalman Filter recursion.
"""
function kalman_filter(ln_F::Matrix{Typ}, T::Matrix{Typ}, p::SSParams{Typ}, delta_t::Int) where Typ

    n, prods = size(ln_F)

    # Predictive state and its covariance matrix
    a_kf   = Matrix{Typ}(undef, n+1, 2)
    P_kf   = Array{Typ, 3}(undef, 2, 2, n+1)
    att_kf = Matrix{Typ}(undef, n, 2)
    Ptt_kf = Array{Typ, 3}(undef, 2, 2, n)

    # One-step forecast (inovation) error and its covariance matrix
    v_kf = Matrix{Typ}(undef, n, prods)
    F_kf = Array{Typ, 3}(undef, prods, prods, n)

    # Initialization
    a_kf[1, :]    = zeros(2, 1)
    P_kf[:, :, 1] = 1e1 .* Matrix(I, 2, 2)

    RQR = W(p, delta_t)
    @assert isposdef(RQR)
    H = V(p)
    @assert isposdef(H)

    # Kalman filter recursion equations
    for t = 1:n
        Z_kf = F(T[t, :], p)
        T_kf = G(p, delta_t)
        d_kf = d(T[t, :], p)
        c_kf = c(p, delta_t)
        ZP   = Z_kf * P_kf[:, :, t]
        v_kf[t, :]      = ln_F[t, :] - Z_kf * a_kf[t, :] - d_kf
        F_kf[:, :, t]   = ZP * Z_kf' + H
        ensure_pos_sym!(F_kf, t)
        @assert isposdef(F_kf[:, :, t])
        inv_F_kf        = pinv(F_kf[:, :, t])
        ensure_pos_sym!(inv_F_kf)
        att_kf[t, :]    = a_kf[t, :] + ZP' * inv_F_kf * v_kf[t, :]
        a_kf[t+1, :]    = T_kf * att_kf[t, :] + c_kf
        Ptt_kf[:, :, t] = P_kf[:, :, t] - ZP' * inv_F_kf * ZP
        ensure_pos_sym!(Ptt_kf, t)
        P_kf[:, :, t+1] = T_kf * Ptt_kf[:, :, t] * T_kf' + RQR
        ensure_pos_sym!(P_kf, t)
        @assert isposdef(P_kf[:, :, t])
    end

    return Filter(a_kf, P_kf, att_kf, Ptt_kf, v_kf, F_kf)
end

"""
    kalman_filter(ln_F::Matrix{Typ}, T_V::Vector{Typ}, p::SSParams{Typ}, delta_t::Int) where Typ

Definition of the Kalman Filter recursion.
"""
function kalman_filter(ln_F::Matrix{Typ}, T_V::Vector{Typ}, p::SSParams{Typ}, delta_t::Int) where Typ

    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    f = kalman_filter(ln_F, T, p, delta_t)

    return f
end

mutable struct Filter{T}
    a_kf::Matrix{T}
    P_kf::Array{T, 3}
    att_kf::Matrix{T}
    Ptt_kf::Array{T, 3}
    v_kf::Matrix{T}
    F_kf::Array{T, 3}
end
