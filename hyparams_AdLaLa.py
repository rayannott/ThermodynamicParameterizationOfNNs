from AdLaLa import AdLaLa_func

# Set the desired hyperparameters: 
# hs = learning rate, gams = friction, T1 = Temperature of first layer (AdLa part), T2 = temp of second layer (Langevin part)
hs    = [0.25, 0.25]
gams  = [ 0.1,  0.1]
T1s   = [1e-4, 1e-6]
T2s   = [1e-4, 1e-6]

# Make sure they have equal length
assert(len(hs)==len(gams))
assert(len(hs)==len(T1s))
assert(len(T1s)==len(T2s))

# Run over the different sets of hyperparameters
for h, gamma,T1,T2 in zip(hs,gams,T1s,T2s):
	print("h = ", h, ", gam =", gamma, ", T1 = ", T1, ", T2 = ", T2)
	AdLaLa_func(h,gamma,T1,T2)
