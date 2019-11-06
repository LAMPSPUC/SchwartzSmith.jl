include("C:\\Users\\mdietze\\Dropbox\\PUC\\Eneva\\Schwartz-Smith\\SchwartzSmith.jl\\src\\SchwartzSmith.jl")

# Matrices
k = 1.49
sigma_chi = 0.286
lambda_chi = 0.157
mi_xi = -0.0125
sigma_xi = 0.145
mi_xi_star = 0.0115
rho_xi_chi = 0.3

T = [10;20]

s = [
    0.042;
    0.006;
]

p = SchwartzSmith.SSParams(k,sigma_chi,lambda_chi,mi_xi,sigma_xi,mi_xi_star,rho_xi_chi,s)

A_test = SchwartzSmith.A(20,p)
W_test = SchwartzSmith.W(p)
G_test = SchwartzSmith.G(p)
c_test = SchwartzSmith.c(p)
d_test = SchwartzSmith.d(T[1,:],p)
F_test = SchwartzSmith.F(T[1,:],p)
V_test = SchwartzSmith.V(p)

# Kalman filter
F    = [collect(1:1:5) collect(6:1:10)]
ln_F = log.(F)
T    = [collect(10:10:50) collect(10:10:50)]
v_kf_test, F_kf_test, att_kf_test = SchwartzSmith.kalman_filter(ln_F, T, p, s)
