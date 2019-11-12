"""
    simulate(p::SSParams, att_kf::Matrix{Float64}, T::Matrix{Float64}, N::Int, S::Int; delta_t::Int = 1)

Simulate S future scenarios up to N steps ahead.
"""
function simulate(p::SSParams, att_kf::Matrix{Float64}, T::Matrix{Float64}, N::Int, S::Int; delta_t::Int = 1, average_T::String = "true")
    n = size(att_kf, 1)
    prods = size(T, 2)

    # Calculation of average time to maturity
    if average_T == "true"
        T_M = Matrix{Float64}(undef, n, prods)

        for i in 1:n, j in 1:prods
            T_M[i, j] = mean(T[:, j])
        end
        T = T_M
    end

    # Covariance matrices
    Q = W(p, delta_t)
    H = V(p)

    # Distribution of the state space errors
    dist_ω = MvNormal(zeros(2), Q)
    dist_v = MvNormal(zeros(prods), H)

    y_sim = Array{Float64, 3}(undef, N, prods, S)
    x_sim = Array{Float64, 3}(undef, N, 2, S)

    for s in 1:S

        ω = rand(dist_ω, N)'
        v = rand(dist_v, N)'

        # Initial values
        x_sim[1, :, s] = c(p, delta_t) + G(p, delta_t) * att_kf[n, :] + ω[1, :]
        y_sim[1, :, s] = d(T[1, :], p) + F(T[1, :], p) * x_sim[1, :, s] + v[1, :]

        # Simulating future scenarios
        for t = 2:N
            x_sim[t, :, s] = c(p, delta_t) + G(p, delta_t) * x_sim[t-1, :, s] + ω[t, :]
            y_sim[t, :, s] = d(T[t, :], p) + F(T[t, :], p) * x_sim[t, :, s] + v[t, :]
        end
    end

    return x_sim, y_sim
end
