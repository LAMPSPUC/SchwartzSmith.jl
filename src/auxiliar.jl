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
    calc_seed(ln_F::Matrix{Typ}, n_seed::Int64) where Typ

Random seed calculation for a time to maturity matrix.
"""
function calc_seed(ln_F::Matrix{Typ}, n_seed::Int64) where Typ

    n, prods = size(ln_F)
    seeds = Matrix{Typ}(undef, 7 + prods, n_seed)

    return transpose!(seeds, -0.2*rand(Typ, n_seed, 7 + size(ln_F, 2)));
end

"""
    calc_D(s::Int, dates::Vector{Int64})

Calculates the dummy matrix for seasonality.
"""
function calc_D(s::Int, dates::Vector{Int64})
    n = length(dates)
    D = zeros(Float64, n, s)

    for i in 1:n
        D[i, dates[i]] = 1
    end
    return D
end
