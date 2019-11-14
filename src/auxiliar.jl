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

    n, prods = size(ln_F)

    println("------------------ Seed 1 ------------------")
    seed = -0.2*rand(Typ, 7 + size(ln_F, 2))
    optseed = optimize(psi -> compute_likelihood(ln_F, T, psi, delta_t_s), seed, LBFGS(), Optim.Options(f_tol = 1e-6, g_tol = 1e-6, show_trace = true))

    min_seed = seed;
    min_opt = optseed.minimum;
    for i = 1:(n_seed - 1)
        println("------------------ Seed ", i + 1, " ------------------")
        seed = -0.2*rand(Typ, 7 + size(ln_F, 2))
        optseed = optimize(psi -> compute_likelihood(ln_F, T, psi, delta_t_s), seed, LBFGS(), Optim.Options(f_tol = 1e-6, g_tol = 1e-6, show_trace = true))

        if optseed.minimum < min_opt
            min_seed = seed
        end
    end

    return min_seed
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

    println("------------------ Seed 1 ------------------")
    seed = -0.2*rand(Typ, 7 + size(ln_F, 2))
    optseed = optimize(psi -> compute_likelihood(ln_F, T, psi, delta_t_s), seed, LBFGS(), Optim.Options(f_tol = 1e-6, g_tol = 1e-6, show_trace = true))

    min_seed = seed;
    min_opt = optseed.minimum;
    for i = 1:(n_seed - 1)
        println("------------------ Seed ", i + 1, " ------------------")
        seed = -0.2*rand(Typ, 7 + size(ln_F, 2))
        optseed = optimize(psi -> compute_likelihood(ln_F, T, psi, delta_t_s), seed, LBFGS(), Optim.Options(f_tol = 1e-6, g_tol = 1e-6, show_trace = true))

        if optseed.minimum < min_opt
            min_seed = seed
        end
    end

    return min_seed
end
