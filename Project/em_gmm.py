import numpy as np 
import matplotlib.pyplot as plt 

thresh = 1e-6

N = 1000
mus = [3]		# means of the mixed Gaussians
alpha = 0.5				# mixing coefficient
sigma = [2.5, 2.5]		# std deviation

# X = np.append(np.random.normal(mu[0], sigma[0], int(N*alpha)), np.random.normal(mu[1], sigma[1], N-int(N*alpha))) 
# np.random.shuffle(X) # random samples drawn from the GMM


def P_gaussian(x, mu, sigma, beta):
	return ((1/(2*np.pi*sigma**2)**0.5)*np.exp(-((x-mu)**2)/2/sigma**2))**beta

def likelihood(alpha, x, mu, sigma, beta):
	return ((alpha_est**beta)*P_gaussian(x, mu[0], sigma[0], beta) + ((1-alpha_est)**beta)*P_gaussian(x, mu[1], sigma[1], beta))

def ds_error(alpha, mu, sigma, alpha_est, mu_est, sigma_est):
	# Symmetric KL divergence
	if abs(1 - alpha - alpha_est) < abs(alpha - alpha_est):
		mu_est = [mu_est[1], mu_est[0]]
		sigma_est = [sigma_est[1], sigma_est[0]]
	err = 0
	for i in range(len(mu)):
		err += 0.5*((sigma[i]/sigma_est[i])**2 + (sigma_est[i]/sigma[i])**2)
		err += 0.5*((mu_est[i] - mu[i])*(1/sigma[i]**2 + 1/sigma_est[i]**2)*(mu_est[i] - mu[i]))
		err -= 1 # dimension
	return err


alphas = np.linspace(0.1, 0.6, 3) # mixing coefficients

errors = []
for mu_iter in mus:
	print(mu_iter)
	mu = [-mu_iter, mu_iter]
	errorss = []
	for alpha in alphas:
		X = np.append(np.random.normal(mu[0], sigma[0], int(N*alpha)), np.random.normal(mu[1], sigma[1], N-int(N*alpha))) 
		np.random.shuffle(X) # random samples drawn from the GMM

		# print('alpha: {}, mu: ({}, {}), sigma: ({}, {})'.format(alpha, mu[0], mu[1], sigma[0], sigma[1]))
		# plt.figure('True PDF')
		# n_bins = 30
		# bins = np.linspace(min(mu)-3*max(sigma), max(mu)+3*max(sigma), n_bins)
		# vals, bins_pos,patch = plt.hist(X, bins, density=True)
		# plt.plot(bins_pos[:-1]+(bins[1]-bins[0])/2, vals, label="N="+str(N))
		# plt.grid(True)
		# plt.show()

		errors = []
		steps = 0
		alpha_est = 0.4
		betas = [0.75, 1.0, 1.2, 1.0]
		# EM algorithm
		# Initial estimates
		mu_est = [0, 1]
		sigma_est = [1., 1.]

		for beta in betas:
			llh_01 = likelihood(alpha_est, X, mu_est, sigma_est, 1)
			llh_1 = likelihood(alpha_est, X, mu_est, sigma_est, beta)
			h = (alpha_est**beta)*P_gaussian(X, mu_est[0], sigma_est[0], beta)/llh_1
			tolerance = np.array([1, 1, 1, 1, 1])
			while np.any(tolerance >= thresh) and steps <= 100000:
				steps += 1
				# print("Step {}".format(steps))
				llh_00 = llh_01
				llh_0 = llh_1

				h_tot = np.sum(h)
				mu_est[0] = np.sum(h*X)/h_tot
				mu_est[1] = np.sum((1-h)*X)/(N-h_tot)
				sigma_est[0] = (np.sum(h*(X-mu_est[0])**2)/h_tot)**0.5
				sigma_est[1] = (np.sum((1-h)*(X-mu_est[1])**2)/(N - h_tot))**0.5
				alpha_est = h_tot/N

				llh_1 = likelihood(alpha_est, X, mu_est, sigma_est, beta)
				llh_01 = likelihood(alpha_est, X, mu_est, sigma_est, 1)

				h = (alpha_est**beta)*P_gaussian(X, mu_est[0], sigma_est[0], beta)/llh_1
				tolerance = np.abs((llh_01-llh_00)/llh_00)
				errors.append(ds_error(alpha, mu, sigma, alpha_est, mu_est, sigma_est))
		errorss.append(errors)
		print('alpha_est: {}, mu_est: ({}, {}), sigma_est: ({}, {})'.format(alpha_est, mu_est[0], mu_est[1], sigma_est[0], sigma_est[1]))
	plt.figure()
	plt.title(r'Error vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	for i, alpha in enumerate(alphas):
		plt.plot(errorss[i], 'x-', label=r'$\alpha=$'+str(alpha))
	plt.grid(True)
	plt.legend(loc='upper right')
	plt.show()

		# stepss.append(steps);
	# stepsss.append(stepss)
	# print(stepss)

# plt.figure()
# plt.title(r'iterations vs. $\alpha$')
# for j, stepss in enumerate(stepsss):
# 	plt.semilogy(alphas, stepss, '^-', label=r'$\mu$='+str(mus[j]))
# 	# plt.semilogy(alphas, stepss)
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.show()

print('alpha_est: {}, mu_est: ({}, {}), sigma_est: ({}, {})'.format(alpha_est, mu_est[0], mu_est[1], sigma_est[0], sigma_est[1]))
print('Steps taken: {}'.format(steps))