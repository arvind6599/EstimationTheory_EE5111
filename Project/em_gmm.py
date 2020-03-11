import numpy as np 
import matplotlib.pyplot as plt 

thresh = 1e-10

N = 1000
mu = [-5, 5]	# means of the mixed Gaussians
alpha = 0.025				# mixing coefficient
sigma = [2.5, 2.5]		# std deviation

X = np.append(np.random.normal(mu[0], sigma[0], int(N*alpha)), np.random.normal(mu[1], sigma[1], N-int(N*alpha))) 
np.random.shuffle(X) # random samples drawn from the GMM
print('alpha: {}, mu: ({}, {}), sigma: ({}, {})'.format(alpha, mu[0], mu[1], sigma[0], sigma[1]))
plt.figure('True PDF')
n_bins = 30
bins = np.linspace(min(mu)-3*max(sigma), max(mu)+3*max(sigma), n_bins)
vals, bins_pos,patch = plt.hist(X, bins, density=True)
plt.plot(bins_pos[:-1]+(bins[1]-bins[0])/2, vals, label="N="+str(N))
plt.grid(True)
plt.show()

def P_gaussian(x, mu, sigma):
	return (1/(2*np.pi*sigma**2)**0.5)*np.exp(-((x-mu)**2)/2/sigma**2)

def likelihood(alpha, x, mu, sigma):
	return (alpha_est*P_gaussian(x, mu[0], sigma[0]) + (1-alpha_est)*P_gaussian(x, mu[1], sigma[1]))

# EM algorithm
# Initial estimates
alpha_est = 0.4
mu_est = [-10, 10.]
sigma_est = [1., 1.]
llh_1 = likelihood(alpha_est, X, mu_est, sigma_est)
h = alpha_est*P_gaussian(X, mu_est[0], sigma_est[0])/llh_0
steps = 0
error = np.array([1, 1, 1, 1, 1])

while np.any(error >= thresh):
	steps += 1
	print("Step {}".format(steps), end="\r")

	llh_0 = llh_1

	h_tot = np.sum(h)
	mu_est[0] = np.sum(h*X)/h_tot
	mu_est[1] = np.sum((1-h)*X)/(N-h_tot)
	sigma_est[0] = (np.sum(h*(X-mu_est[0])**2)/h_tot)**0.5
	sigma_est[1] = (np.sum((1-h)*(X-mu_est[1])**2)/(N - h_tot))**0.5
	alpha_est = h_tot/N

	llh_1 = likelihood(alpha_est, X, mu_est, sigma_est)

	h = alpha_est*P_gaussian(X, mu_est[0], sigma_est[0])/llh_1

	error = np.abs((llh_1-llh_0)/llh_1)


print('alpha_est: {}, mu_est: ({}, {}), sigma_est: ({}, {})'.format(alpha_est, mu_est[0], mu_est[1], sigma_est[0], sigma_est[1]))
print('Steps taken: {}'.format(steps))