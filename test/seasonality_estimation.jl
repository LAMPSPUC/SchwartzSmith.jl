function calc_T(T::Matrix{Float64}, contracts::Vector{String})
    for i in 1:length(contracts)
        

    end


end
#push!(LOAD_PATH, "C:\\Users\\marin\\Dropbox\\PUC\\Eneva\\Modelos\\Schwartz-Smith\\SchwartzSmith.jl\\src")
include("C:\\Users\\mdietze\\Dropbox\\PUC\\Eneva\\Modelos\\Schwartz-Smith\\SchwartzSmith.jl\\src\\SchwartzSmith.jl");

#using SchwartzSmith
using DelimitedFiles
using Statistics
using Dates
using LinearAlgebra
using Plots

path = "C:\\Users\\mdietze\\Dropbox\\PUC\\Eneva\\Modelos\\Benchmarks\\Projeto-Eneva\\Caso_2_Sazonalidade";

# Data reading
function read_data(path::String)
    in_data = readdlm(joinpath(path, "price_tm_new.csv"), ';')
    in_data = in_data[3:end, 2:end]
    n       = size(in_data, 1)
    prods   = trunc(Int32, size(in_data, 2)/3)

    prices     = Array{Float64, 2}(undef, n, prods)
    T          = Array{Float64, 2}(undef, n, prods)
    from_dates = Array{SubString{String}, 2}(undef, n, prods)

    k = 1
    for i in 1:prods
        prices[:, i]     = in_data[:, k]
        from_dates[:, i] = in_data[:, k + 1]
        T[:, i]          = in_data[:, k + 2]
        k += 3
    end

    return prices, T, from_dates
end

prices, T, from_dates = read_data(path);

dates_upd = Vector{Date}(undef, length(from_dates));
month_dates = Vector{Int64}(undef, length(from_dates));

df = DateFormat("d/m/y");
dates_upd   = Date.(from_dates, df)
month_dates = Dates.month.(dates_upd)

n = 2000;
prods = [1; 2; 6]
month_dates = month_dates[1:n, prods];
ln_F = log.(prices[1:n, prods]);
T_M = mean(T, dims = 1);
T_M = T_M[prods];
s = 12;

D = SchwartzSmith.calc_D(s, month_dates)

T = T[1:n, prods];

seed1 = [
    log(0.942606507767713)
    log(0.9044844546007459)
    -0.025957400993965048
    -0.00010867186345375894
    log(0.03002685907798695)
    4.653430858280236e-5
    -log((2/(-0.0034182133208745524 + 1)) - 1)
    log(0.281513)
    log(0.00226815)
    log(0.453614)
];

seed2 = [
    log(1.7811786384466834)
    log(0.6309196433480714)
    0.01781788000769942
    -5.3365761550246524e-5
    log(0.03210569668765596)
    0.00046439966928806637
    -log((2/(-0.06176572278576198 + 1)) - 1)
    log(0.261334)
    log(0.00881946)
    log(0.270908)
];


p_1, seed, optseeds = SchwartzSmith.schwartzsmith(ln_F, T_M, month_dates, s);
p_2, seed, optseeds = SchwartzSmith.schwartzsmith(ln_F, T, month_dates, s; seeds = seed2);
p_3, seed, optseeds = SchwartzSmith.schwartzsmith(ln_F, T_M; seeds = seed1);
p_4, seed, optSeeds = SchwartzSmith.schwartzsmith(ln_F, T; seeds = seeds = seed2);

y_1, f_1, s_1 = SchwartzSmith.estimated_prices(p_1, T_M, ln_F, month_dates, s);
y_2, f_2, s_2 = SchwartzSmith.estimated_prices(p_2, T, ln_F, month_dates, s);
y_3, f_3, s_3 = SchwartzSmith.estimated_prices(p_3, T_M, ln_F);
y_4, f_4, s_4 = SchwartzSmith.estimated_prices(p_4, T, ln_F);

# ----------------------------------------------------------------------------#

D = SchwartzSmith.calc_D(s, month_dates)

X_1 = zeros(2000)
X_2 = zeros(2000)
for i in 1:2000
    X_1[i] = dot(s_1.alpha[i, 3:end], D[i, :])
    #X_2[i] = dot(s_2.alpha[i, 3:end], D[i, :])
end

plot(ln_F[1:n, 1])
plot!(y_1[:, 1])
plot!(sum(s_1.alpha[:, 1:2], dims = 2) .+ X_1)

plot(ln_F[1:n, 1])
plot!(y_2[:, 1])
plot!(sum(s_2.alpha[:, 1:2], dims = 2) .+ X_2)

plot(ln_F[1:n, 1])
plot!(y_3[:, 1])
plot!(sum(s_3.alpha[:, 1:2], dims = 2))

plot(ln_F[1:n, 1])
plot!(y_4[:, 1])
plot!(sum(s_4.alpha[:, 1:2], dims = 2))

# --------------------------------------------------------------------------- #

month_dates = Dates.month.(dates_upd)
month_dates = month_dates[2001:2500]
prices, T, dates = read_data(path);

forec_1 = SchwartzSmith.forecast(f_1, T_M, month_dates, 12, p_1, 500);
forec_2 = SchwartzSmith.forecast(f_2, T[1001:1500, 1:3], month_dates, 12, p_2, 500);
forec_3 = SchwartzSmith.forecast(f_3, T_M, p_3, 500);
forec_4 = SchwartzSmith.forecast(f_4, T[1001:1500, 1:3], p_4, 500);

y_init = ones(n).*NaN
y_forecast = vcat(y_init, forec_4[:, 1])
plot(prices[1:n + 500, 1])
plot!(exp.(y_forecast))

a = Matrix{Float64}(undef, 1000, 2 + 12)
a0 = f_1.att_kf[end, :]
a[1, :]    = SchwartzSmith.G(p_1, 12, 1) * a0 + SchwartzSmith.c(p_1, 12, 1)

# --------------------------------------------------------------------------- #

month_dates = Dates.month.(dates_upd);
month_dates = month_dates[1001:1500];
D = SchwartzSmith.calc_D(s, month_dates);
prices, T, dates = read_data(path);
T = T[1001:1500, 1:prods]

N = 500;
S = 200;
y_sim_1 = SchwartzSmith.simulate(p_1, f_1.att_kf, T_M, month_dates, s, N, S)[2]
y_sim_2 = SchwartzSmith.simulate(p_2, f_2.att_kf, T, month_dates, s, N, S)[2];
y_sim_3 = SchwartzSmith.simulate(p_3, f_3.att_kf, T_M, N, S)[2];
y_sim_4 = SchwartzSmith.simulate(p_4, f_4.att_kf, T, N, S)[2];

y_init = ones(n, S).*NaN
simul_1 = vcat(y_init, y_sim_2[:, 1, :])
plot(ln_F[:, 1])
plot!(simul_1, label = "")
