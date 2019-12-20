"""
    sqrt_filtered_state(T::Matrix{Typ}, sqrt_f::SquareRootFilter{Typ}, p::SSParams{Typ}, delta_t::Int64) where Typ

Filtered states resulted from the square root Kalman Filter.
"""
function sqrt_filtered_state(T::Matrix{Typ}, sqrt_f::SquareRootFilter{Typ}, p::SSParams{Typ}, delta_t::Int64) where Typ
    X_t = Vector{Float64}(undef, 0)
    n_exp = 0

    # Dimensions data
    n, prods = size(T)
    m        = size(sqrt_f.a_kf, 2)

    # Square root filter data
    a_kf = sqrt_f.a_kf
    v_kf = sqrt_f.v_kf
    F_kf = gram_in_time(sqrt_f.sqrtF_kf)
    P_kf = gram_in_time(sqrt_f.sqrtP_kf)

    # Filtered state and its covariance
    att_kf = Matrix{Typ}(undef, n, m)
    Ptt_kf = Array{Typ, 3}(undef, m, m, n)

    for t in 1:n
        Z_kf = F(T[t, :], p, X_t)
        T_kf = G(p, n_exp, delta_t)
        d_kf = d(T[t, :], p)
        c_kf = c(p, n_exp, delta_t)

        PZF = P_kf[:, :, t] * Z_kf' * pinv(F_kf[:, :, t])
        att_kf[t, :]    = a_kf[t, :] + PZF * v_kf[t, :] + c_kf
        Ptt_kf[:, :, t] = P_kf[:, :, t] - PZF * Z_kf * P_kf[:, :, t]
        ensure_pos_sym!(Ptt_kf, t)
    end

    return FilteredState(att_kf, Ptt_kf)
end

"""
    sqrt_filtered_state(T::Matrix{Typ}, X::Matrix{Typ}, sqrt_f::SquareRootFilter{Typ}, p::SSParams{Typ}, delta_t::Int64) where Typ

Filtered state resulted from the square root Kalman Filter when exogenous variables are included.
"""
function sqrt_filtered_state(T::Matrix{Typ}, X::Matrix, sqrt_f::SquareRootFilter{Typ}, p::SSParams{Typ}, delta_t::Int64) where Typ
    X = X[:, :]
    n_exp = size(X, 2)

    # Dimensions data
    n, prods = size(T)
    m        = size(sqrt_f.a_kf, 2)

    # Square root filter data
    a_kf = sqrt_f.a_kf
    v_kf = sqrt_f.v_kf
    F_kf = gram_in_time(sqrt_f.sqrtF_kf)
    P_kf = gram_in_time(sqrt_f.sqrtP_kf)

    # Filtered state and its covariance
    att_kf = Matrix{Typ}(undef, n, m)
    Ptt_kf = Array{Typ, 3}(undef, m, m, n)

    for t in 1:n
        Z_kf = F(T[t, :], p, X[t, :])
        T_kf = G(p, n_exp, delta_t)
        d_kf = d(T[t, :], p)
        c_kf = c(p, n_exp, delta_t)

        PZF = P_kf[:, :, t] * Z_kf' * pinv(F_kf[:, :, t])
        att_kf[t, :]    = a_kf[t, :] + PZF * v_kf[t, :] + c_kf
        Ptt_kf[:, :, t] = P_kf[:, :, t] - PZF * Z_kf * P_kf[:, :, t]
        ensure_pos_sym!(Ptt_kf, t)
    end

    return FilteredState(att_kf, Ptt_kf)
end

"""
    FilteredState{T}

Struct with the filtered states.
"""
mutable struct FilteredState{T}
    att_kf::Matrix{T}
    Ptt_kf::Array{T, 3}
end

"""
    Filter{T}

Struct with square root Kalman Filter results and filtered states.
"""
mutable struct Filter{T}
    sqrt_f::SquareRootFilter{T}
    att_kf::Matrix{T}
    Ptt_kf::Array{T, 3}
end
