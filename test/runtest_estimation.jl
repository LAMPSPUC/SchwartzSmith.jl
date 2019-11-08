# include("C:\\Users\\mdietze\\Dropbox\\PUC\\Eneva\\Schwartz-Smith\\SchwartzSmith.jl\\src\\SchwartzSmith.jl")

push!(LOAD_PATH, "C:\\Users\\mdietze\\Dropbox\\PUC\\Eneva\\Schwartz-Smith\\SchwartzSmith.jl\\src")
using SchwartzSmith

using DelimitedFiles
using Statistics

path = "C:\\Users\\mdietze\\Dropbox\\PUC\\Eneva\\Schwartz-Smith"

function get_prices_T(path::String)
    in_data = readdlm(joinpath(path,"price_tm.csv"),';')
    in_data = in_data[3:end,1:end]
    n_t = size(in_data,1)
    n_cont = trunc(Int32,(size(in_data,2)-1)/2)

    prices = Array{Float64,2}(undef,n_t,n_cont)
    T = Array{Float64,2}(undef,n_t,n_cont)

    k = 2
    for i in 1:n_cont
        prices[:,i] = in_data[:, k]
        T[:,i] = in_data[:, k + 1]
        k += 2
    end

    return prices,T
end

prices, T = get_prices_T(path);
n, prods = 230, 2

ln_F = log.(prices[1:n, 1:prods])
T_M = Matrix{Float64}(undef, n, prods)

for i in 1:n, j in 1:prods
    T_M[i, j] = mean(T[:, j])
end

p, seed = schwartzsmith(ln_F, T_M)

seed = [ -0.06180695966037159
 -0.17609367052333552
 -0.17261783758232532
 -0.19012491757762395
 -0.1308693643483689
 -0.27550910116468547
 -0.07625347503184028
 -0.08939463265361693
 -0.021270250220516696]

y, x = estimated_prices_states(p, T_M, ln_F)

N = 100
S = 200
T_sim = T_M[1:N, :]

x_sim, y_sim = simulate(p, x, T_sim, N, S)

using Plots

product = 2
plot(ln_F[:, product], label = "")
plot!(y[:, product], label = "")
y_init = ones(n, S).*NaN

y_simul = vcat(y_init, y_sim[:, product, :])

plot!(y_simul[:, :], label = "")

# forecast

y_forec = mean(y_simul, dims = 2)
y_forec[231:330]

plot(prices[1:(n+N), product], label = "")
plot!(exp.(y_forec), label = "")

forec = mean(y_sim, dims = 3)

forec[:, 2, :]

y_forec, x_forec = forecast(y_sim, x_sim)

plot(x_sim[:,1,1])
plot!(x_sim[:,2,1])

spot = exp.(x_forec[:, 1, 1] + x_forec[:, 2, 1])

plot(spot)
plot!(exp.(y_forec[:,1,1]))
