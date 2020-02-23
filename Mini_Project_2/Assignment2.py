import numpy as np
import matplotlib.pyplot as plt
	
def newton_rhaphson(x,thresh,gamma):	
	theta_0 = np.median(x,0) 
	b = (x-theta_0)/gamma
	l_1 = 2*np.sum(b /(1 + b*b),0)
	while(np.maximum(np.abs(l_1)) > thresh):
		l_2 = 2*sum((b*b - 1)/((1 + b*b)*(1 + b*b)))
		theta_0 = theta_0 - l_1/l_2
		b = (x - theta_0)/gamma
		l_1 = 2*sum(b /(1 + b*b))
	return theta_0

def main():
	Q_no = 1 # Question number in assignment	
	N = 1000 # Number of samples
	A = 1 	# DC Value to be estimated
	thresh = 0.0001
			
	iters = 100000 # Number of iterations run for finding the mean and variance of the estimator 

	# Generating samples
	if Q_no == 1:
		noise = np.random.normal(0,1,(N,iters)) 
		X = A + noise
		A_estimated = np.sum(X,0)/N  # shape = (iters,1) and contains the MLE estimated value of A for each iteration
	elif Q_no == 2:
		noise = np.random.laplace(0,1/(2**0.5),(N,iters)) 
		X = A + noise
		A_estimated = np.median(X,0)  # shape = (iters,1) and contains the MLE estimated value of A for each iteration
	else:
		gamma = ((2*1.78)**0.5)
		noise = np.random.standard_cauchy((N,iters))*((2*1.78)**0.5)*gamma
		X = A + noise
		A_estimated = newton_rhaphson(X,thresh,gamma)  # shape = (iters,1) and contains the MLE estimated value of A for each iteration

	# Finding mean and variance of estimator
	E_A_estimated = np.mean(A_estimated)
	Var_A_estimated = np.var(A_estimated)
	print("Expected of value of the MLE estimate is {} and its variance is {}".format(E_A_estimated,Var_A_estimated))

	# Plotting PDF (or CDF) of the estimator
	n_bins = 30
	bins = (np.arange(n_bins)/n_bins-0.5)*4*2
	vals, bins_pos,patch = plt.hist((A_estimated-A)*(N**0.5),bins,normed = True)
	plt.cla()
	plt.plot(bins_pos[:-1]+(bins[1]-bins[0])/2,vals)
	plt.show()

if__name__== "__main__":
	main()