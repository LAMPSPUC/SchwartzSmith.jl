"""
    ensure_pos_sym!(M::AbstractArray{T}, t::Int; ϵ::T = T(1e-8)) where T
    ensure_pos_sym!(M::AbstractArray{T}; ϵ::T = T(1e-8)) where T

Force matrix to be positive definite.
"""
function ensure_pos_sym!(M::AbstractArray{T}, t::Int; ϵ::T = T(1e-8)) where T
    @inbounds for j in axes(M, 2), i in 1:j
        if i == j
            M[i, i, t] = (M[i, i, t] + M[i, i, t])/2 + ϵ
        else
            M[i, j, t] = (M[i, j, t] + M[j, i, t])/2
            M[j, i, t] = M[i, j, t]
        end
    end
    return
end

function ensure_pos_sym!(M::AbstractArray{T}; ϵ::T = T(1e-8)) where T
    @inbounds for j in axes(M, 2), i in 1:j
        if i == j
            M[i, i] = (M[i, i] + M[i, i])/2 + ϵ
        else
            M[i, j] = (M[i, j] + M[j, i])/2
            M[j, i] = M[i, j]
        end
    end
    return
end

"""
    calc_seed(ln_F::Matrix{Typ}, T::Matrix{Typ}, n_seed::Int64, delta_t_s::Int64) where Typ

Random seed calculation for a time to maturity matrix.
"""
function calc_seed(ln_F::Matrix{Typ}, T::Matrix{Typ}, n_seed::Int64, delta_t_s::Int64) where Typ

    # Test with different seeds
    seed_l = -0.2*rand(Typ, 7 + size(ln_F, 2))
    min_l = compute_likelihood(ln_F, T, seed_l, delta_t_s)
    for i = 1:n_seed
        seed = -0.2*rand(Typ, 7 + size(ln_F, 2))
        calc_l = compute_likelihood(ln_F, T, seed, delta_t_s)

        if calc_l < min_l
            min_l = calc_l
            seed_l = seed
        end
    end
    return seed = seed_l
end

"""
    calc_seed(ln_F::Matrix{Typ}, T::Vector{Typ}, n_seed::Int64, delta_t_s::Int64) where Typ

Random seed calculation for a time to maturity vector.
"""
function calc_seed(ln_F::Matrix{Typ}, T_V::Vector{Typ}, n_seed::Int64, delta_t_s::Int64) where Typ

    n, prods = size(ln_F)
    T = Matrix{Typ}(undef, n, prods)

    # Representation of the time to maturity matrix
    for i in 1:n, j in 1:prods
        T[i, j] = T_V[j]
    end

    # Test with different seeds
    seed_l = -0.2*rand(Typ, 7 + size(ln_F, 2))
    min_l = compute_likelihood(ln_F, T, seed_l, delta_t_s)
    for i = 1:n_seed
        seed = -0.2*rand(Typ, 7 + size(ln_F, 2))
        calc_l = compute_likelihood(ln_F, T, seed, delta_t_s)

        if calc_l < min_l
            min_l = calc_l
            seed_l = seed
        end
    end
    return seed = seed_l
end
