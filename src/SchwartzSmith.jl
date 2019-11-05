module SchwartzSmith

using Optim

include("create_matrices.jl")
include("auxiliar.jl")

"""
The Schwartz-Smith model equations are:
Observation equation: y_t = d_t + F'_t*x_t + v_t
State equation: x_t = c + G*x_t-1 + ω_t

For each trading period t, we use a set of forward contracts with
differents times to maturity T_1,...T_n. The default value for Δt is 1.
"""


end # module
