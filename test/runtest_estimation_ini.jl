# include("C:\\Users\\mdietze\\Dropbox\\PUC\\Eneva\\Schwartz-Smith\\SchwartzSmith.jl\\src\\SchwartzSmith.jl")

push!(LOAD_PATH, "C:\\Users\\marin\\Dropbox\\PUC\\Eneva\\Schwartz-Smith\\SchwartzSmith.jl\\src")
using SchwartzSmith

using DelimitedFiles
using Statistics
using Plots

path = "C:\\Users\\marin\\Dropbox\\PUC\\Eneva\\Schwartz-Smith"

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

prices, T = get_prices_T(path)
n, prods = 200, 1

ln_F = log.(prices[1:n, 1:prods])
T_M = Matrix{Float64}(undef, n, prods)

for i in 1:n, j in 1:prods
    T_M[i, j] = mean(T[:, j])
end

seed = [ -0.10654784669735702
 -0.12807569123084248
 -0.19325351987735104
 -0.18393617771989865
 -0.049912804722217624
 -0.14371420739033378
 -0.05822995783198568
 -0.1308888882160153
 -0.0662292245283235
 -0.1382878328685147
 -0.01261296624712367;]

plot(exp(ln_F[:, 1]), label = "M + 1", width = 1, title = "Forward Prices", xlabel = "Time", ylabel = "\$")
plot!(ln_F[:, 2], label = "M + 2", width = 1)
plot!(ln_F[:, 3], label = "M + 3", width = 1)
plot!(ln_F[:, 4], label = "M + 4", width = 1)

plot(exp.(ln_F[:, 1]))

#---------------- Schwartz Smith parameters estimation -------------------#
p, seed = schwartzsmith(ln_F, T_M);

#---------------- Prices estimated by the model -------------------#
y, f = estimated_prices_states(p, T_M, ln_F)

plot(ln_F[:, 1], label = "", title = "M + 1", xlabel = "Time", ylabel = "\$", width = 1)
forward_prices_M1 = plot(y[:, 1], label = "", width = 1)

plot(ln_F[:, 2], label = "", title = "M + 2", xlabel = "Time", ylabel = "\$", width = 1)
forward_prices_M2 = plot!(y[:, 2], label = "", width = 1)

plot(ln_F[:, 3], label = "", title = "M + 3", xlabel = "Time", ylabel = "\$", width = 1)
forward_prices_M3 = plot!(y[:, 3], label = "", width = 1)

plot(ln_F[:, 4], label = "", title = "M + 4", xlabel = "Time", ylabel = "\$", width = 1)
forward_prices_M4 = plot!(y[:, 4], label = "", width = 1)

plot(forward_prices_M1, forward_prices_M2, forward_prices_M3, forward_prices_M4)

plot!((sum(f.att_kf, dims = 2)))

plot(prices[1:200, 1])
plot!(exp.(y[:, 1]))
plot!(exp.(sum(f.att_kf, dims = 2)))

histogram(f.v_kf[:])
#---------------- Scenarios simulation -------------------#
N = 100
S = 200

T_sim = T_M[1:N, :]

x_sim, y_sim = simulate(p, f.att_kf, T_sim, N, S)
y_init = ones(n, S).*NaN

y_simul_1 = vcat(y_init, y_sim[:, 1, :])
plot(ln_F[:, 1], label = "", title = "M + 1", xlabel = "Time", ylabel = "\$", width = 1)
simul_prices_M1 = plot!(y_simul_1, label = "", width = 1)

y_simul_2 = vcat(y_init, y_sim[:, 2, :])
plot(ln_F[:, 2], label = "", title = "M + 2", xlabel = "Time", ylabel = "\$", width = 1)
simul_prices_M2 = plot!(y_simul_2, label = "", width = 1)

y_simul_3 = vcat(y_init, y_sim[:, 3, :])
plot(ln_F[:, 3], label = "", title = "M + 3", xlabel = "Time", ylabel = "\$", width = 1)
simul_prices_M3 = plot!(y_simul_3, label = "", width = 1)

y_simul_4 = vcat(y_init, y_sim[:, 4, :])
plot(ln_F[:, 4], label = "", title = "M + 4", xlabel = "Time", ylabel = "\$", width = 1)
simul_prices_M4 = plot!(y_simul_4, label = "", width = 1)

#---------------- Forecast N steps ahead -------------------#
for_N = forecast(f, T_sim, p, 100)
ln_F = log.(prices[1:n+N, 1:prods])

y_init = ones(n).*NaN
y_forecast = vcat(y_init, for_N[:, 1])
plot(ln_F[1:n + N, 1], label = "", title = "M + 1", xlabel = "Time", ylabel = "\$")
plot!(y_forecast)

y_forecast = vcat(y_init, for_N[:, 2])
plot(ln_F[1:n + N, 2], label = "", title = "M + 2", xlabel = "Time", ylabel = "\$")
plot!(y_forecast)

y_forecast = vcat(y_init, for_N[:, 3])
plot(ln_F[1:n + N, 3], label = "", title = "M + 3", xlabel = "Time", ylabel = "\$")
plot!(y_forecast)

y_forecast = vcat(y_init, for_N[:, 4])
plot(ln_F[1:n + N, 4], label = "", title = "M + 4", xlabel = "Time", ylabel = "\$")
plot!(y_forecast)
