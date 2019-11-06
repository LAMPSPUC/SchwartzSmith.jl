include("C:\\Users\\mdietze\\Dropbox\\PUC\\Eneva\\Schwartz-Smith\\SchwartzSmith.jl\\src\\SchwartzSmith.jl")

using DataFrames
using DelimitedFiles

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
