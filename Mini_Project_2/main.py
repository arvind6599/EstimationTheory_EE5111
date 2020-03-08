import numpy as np
import matplotlib.pyplot as plt
import sys
	
def newton_raphson(x,thresh,gamma):	
	theta_0 = np.median(x,0) 
	b = (x-theta_0)/gamma
	l_1 = 2*np.sum(b /(1 + b*b), 0)
	while(np.all(np.abs(l_1) > thresh)):
		l_2 = 2*sum((b*b - 1)/((1 + b*b)*(1 + b*b)))
		theta_0 = theta_0 - l_1/l_2
		b = (x - theta_0)/gamma
		l_1 = 2*sum(b /(1 + b*b))
	return theta_0

def main():
	Q_no = 1 # Question number in assignment	
	N_s = [1, 100, 1000, 10000] # Number of samples
	A = 1 	# DC Value to be estimated
	thresh = 0.000001
			
	iters = 10000 # Number of iterations run for finding the mean and variance of the estimator 

	A_estimated = []

	# Generating samples
	if Q_no == 1:
		for N in N_s:
			noise = np.random.normal(0,1,(N,iters)) 
			X = A + noise
			A_estimated.append(np.sum(X,0)/N)
	elif Q_no == 2:
		for N in N_s: 
			noise = np.random.laplace(0,1/(2**0.5),(N,iters)) 
			X = A + noise
			A_estimated.append(np.median(X,0))
	else:
		gamma = ((2*1.78)**0.5)
		for N in N_s:
			noise = np.random.standard_cauchy((N,iters))*gamma
			X = A + noise
			A_estimated.append(newton_raphson(X,thresh,gamma))

	# Finding mean and variance of estimator
	print("N\tE[A_est]\t\tVar(A_est)")
	for j in range(len(N_s)):
		N = N_s[j]
		A_est = A_estimated[j]
		E_A_estimated = np.mean(A_est)
		Var_A_estimated = np.var(A_est)
		print("{}\t{}\t{}".format(N, E_A_estimated, Var_A_estimated))

		# Plotting PDF (or CDF) of the estimator
		plt.figure('PDFs')
		
		n_bins = 30
		bins = np.linspace(-3*(N*Var_A_estimated)**0.5, 3*(N*Var_A_estimated)**0.5, n_bins)
		vals, bins_pos,patch = plt.hist((A_est-A)*(N**0.5), bins, density=True)
		# plt.cla()
		cdf_vals = np.zeros(vals.shape)
		cdf_vals[0] = vals[0]
		for i in range(1, len(vals)):
			cdf_vals[i] = vals[i] + cdf_vals[i-1]
		plt.plot(bins_pos[:-1]+(bins[1]-bins[0])/2, vals, label="N="+str(N))
		plt.figure('CDFs')
		plt.plot(bins_pos[:-1]+(bins[1]-bins[0])/2, cdf_vals/cdf_vals[-1],label="N="+str(N))
		
	plt.figure('PDFs')
	plt.legend(loc='upper right')
	plt.figure('CDFs')
	plt.legend(loc='upper right')
	plt.show()


if __name__== "__main__":
	main()