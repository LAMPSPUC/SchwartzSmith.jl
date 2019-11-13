"""
    Package test with artificial data from monthly contracts.
"""
push!(LOAD_PATH, "C:\\Users\\mdietze\\Dropbox\\PUC\\Eneva\\Schwartz-Smith\\SchwartzSmith.jl\\src")
using SchwartzSmith

using DelimitedFiles
using Plots
using Statistics

# Loading the data
path = "C:\\Users\\mdietze\\Dropbox\\PUC\\Eneva\\Schwartz-Smith\\SchwartzSmith.jl\\test\\data.csv"

function get_prices_T(path::String)
    in_data = readdlm(path,',')
    in_data = in_data[3:end,1:end]
    n_t = size(in_data,1)
    n_cont = trunc(Int32,size(in_data,2)/2)

    prices = Array{Float64,2}(undef,n_t,n_cont)
    T = Array{Float64,2}(undef,n_t,n_cont)

    k = 1
    for i in 1:n_cont
        prices[:, i] = in_data[:, k]
        T[:, i] = in_data[:, k + 1]
        k += 2
    end

    return prices,T
end

prices, T_all = get_prices_T(path)

n, prods = size(prices)
ln_F = log.(prices[1:n, 1:prods])[1:800, :]
T = T_all[1:800, :]

T_A = [mean(T[:, 1]); mean(T[:,2])]

# Schwartz Smith model with random seed
p, seed, optseed = schwartzsmith(ln_F, T_A)
y, f = estimated_prices_states(p, T_A, ln_F)

p, seed, optseed = schwartzsmith(ln_F, T)
y, f = estimated_prices_states(p, T, ln_F)

# Schwartz Smith model with a specific seed
seed_t = [-0.12484220610793516
-0.13603638507367366
-0.1634576131026249
-0.05535350189974833
-0.1998703761043637
-0.1738276551834587
-0.156635465891093
-0.13360827922373616
-0.17940876802089156];

p, seed, optseed = schwartzsmith(ln_F, T; seed = seed_t)
y, f = estimated_prices_states(p, T, ln_F)

p, seed, optseed = schwartzsmith(ln_F, T_A; seed = seed_t)
y, f = estimated_prices_states(p, T_A, ln_F)

# Scenarios simulation
N = 100
S = 200

T_sim = T_all[801:900, :]

x_sim, y_sim = simulate(p, f.att_kf, T_A, N, S)

# N steps ahead forecasting
for_N = forecast(f, T_A, p, 100)
