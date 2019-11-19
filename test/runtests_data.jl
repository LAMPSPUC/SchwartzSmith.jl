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
ln_F = log.(prices[1:n, 1:prods])[1:500, :]
T = T_all[1:500, :]

T_A = [mean(T[:, 1]); mean(T[:,2])]

# Schwartz Smith model with random seed
p, seed, optseed = schwartzsmith(ln_F, T_A)
y, f = estimated_prices_states(p, T_A, ln_F)
seed_A = seed;

p, seed, optseed = schwartzsmith(ln_F, T)
y, f = estimated_prices_states(p, T, ln_F)
seed_T = seed;

# Schwartz Smith model with a specific seed
seed_A = [-0.15698310465032694
 -0.14497516893356796
 -0.10300713562481496
 -0.09296434815599874
 -0.13956616744266512
 -0.13056494535739907
 -0.04466052260412545
 -0.1278401688883932
 -0.13762021695260906]

seed_A = SchwartzSmith.calc_seed(ln_F, 5)

p, seed, optseed = schwartzsmith(ln_F, T_A)
y, f = estimated_prices_states(p, T_A, ln_F)

# Scenarios simulation
N = 100
S = 200

T_sim = T_all[801:900, :]

x_sim, y_sim = simulate(p, f.att_kf, T_A, N, S)

# N steps ahead forecasting
for_N = forecast(f, T_A, p, 100)
