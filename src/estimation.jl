"""
    calc_likelihood(v_kf::Matrix{Typ}, F_kf::Array{Typ, 3}, n::Int, prods::Int) where Typ

Obtains the log likelihood value given a set of parameters.
"""
function calc_likelihood(v_kf::Matrix{Typ}, F_kf::Array{Typ, 3}, n::Int, prods::Int) where Typ
    log_likelihood = -n * prods * log(2*pi)/2
    @inbounds for t in 1:n
        log_likelihood -= 0.5 * (logdet(F_kf[:, :, t]) + v_kf[t, :]' * inv(F_kf[:, :, t]) * v_kf[t, :])
    end
    return -log_likelihood
end

"""
    compute_likelihood(ln_F::Matrix{Typ}, T::Matrix{Typ}, psi::Vector{Typ}) where Typ

Parameters definition and Kalman filter execution to calculate the log likelihood. Function that will be optmized to obtain
the best set of parameters.
"""
function compute_likelihood(ln_F::Matrix{Typ}, T::Matrix{Typ}, psi::Vector{Typ}, delta_t::Int) where Typ

    k = exp(psi[1])
    σ_χ = exp(psi[2])
    λ_χ = psi[3]
    μ_ξ = psi[4]
    σ_ξ = exp(psi[5])
    μ_ξ_star = psi[6]
    ρ_ξ_χ = -1 + 2/(1 + exp(-psi[7]))
    s = exp.(psi[8:end])

    p = SSParams(k, σ_χ, λ_χ, μ_ξ, σ_ξ, μ_ξ_star, ρ_ξ_χ, s)

    n, prods = size(ln_F)
    @assert length(s) == prods

    v_kf, F_kf, att_kf = kalman_filter(ln_F, T, p, delta_t)

    return calc_likelihood(v_kf, F_kf, n, prods)
end
