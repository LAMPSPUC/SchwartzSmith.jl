"""
Definition of the Kalman Filter recursion.
"""

function kalman_filter(ln_F::Matrix{Typ}, T::Matrix, p::SSParams, s::Vector{Typ}) where Typ

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

    RQR = W(p)
    @assert isposdef(RQR)
    H = V(s)
    @assert isposdef(H)

    # Kalman filter recursion equations
    for t = 1:n
        Z_kf = F(T[t, :], p)
        T_kf = G(p)
        d_kf = d(T[t, :], p)
        c_kf = c(p)
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

    return v_kf, F_kf, att_kf
end
