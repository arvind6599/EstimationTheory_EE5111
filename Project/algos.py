import time
import numpy as np 

def P_gaussian(x, mu, sigma, beta):
	n, N = x.shape
	if n > 1:
		wsig, vsig = np.linalg.eig(sigma)
		sig_inv = vsig.T.dot(np.diag(1/wsig).dot(vsig)) # sigma inverse
	else:
		wsig = sigma
		sig_inv = 1/sigma
	
	x_mu = x - mu
	xx = sig_inv[:, 0].reshape((n, 1)) * x_mu[0] 
	for i in range(1,n):
		xx += sig_inv[:, i].reshape((n, 1)) * x_mu[i]
	exp_arg = -(x_mu * xx)/ 2

	return ((1/((2*np.pi)**n/2)*(abs(np.prod(wsig)))**0.5)*np.exp(exp_arg)**beta)

def likelihood(alphas, x, mus, sigmas, beta):
	ll = 0
	K = alphas.size
	for k in range(K):
		ll += (alphas[k]**beta)*P_gaussian(x, mus[k], sigmas[k], beta)
	return ll

def ds_error(alpha, mu, sigma, alpha_est, mu_est, sigma_est):
	################ NEEEEDSSS CHANGING #####################
	###### HAVEN'T DONE THISSSS #############################
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

class Solver:

	def __init__(self, mu, sigma, alpha):
		self.mu = mu
		self.sigma = sigma
		self.alpha = alpha
	
	def DAEM_GMM(self, X, thresh, mu_est=None, sigma_est=None, alpha_est=None, betas=[1.], K=2):
		"""
			Deterministic Annealing EM Algorithm for k n-dimensional Gaussians

			X.shape = n x N. Xi is n-dimensional. N data points 
		"""
		n, N = X.shape

		errors = []
		alpha_ests = []; mu_ests=[]; likelihoods = []
		beta_step = []
		steps = 0

		# Initial estimates
		if alpha_est is None:
			alpha_est = np.array([1./K for j in range(K)])
		if mu_est is None:
			mu_est = [X[:, int(np.random.random()*N)].reshape((n,1)) for j in range(K)]
		if sigma_est is None:
			sample_mean = np.sum(X, axis=0)/N
			X_mu = X - sample_mean
			cov = np.zeros((n,n))
			for i in range(n):
				cov[i] += np.sum(X_mu[i]*X_mu, axis=1)
			cov /= N
			sigma_est = [cov for j in range(K)]
 
		actual_likelihood = np.sum(np.log(likelihood(self.alpha, X, self.mu, self.sigma, 1))) # With actual parameters
	
		#errors.append(ds_error(self.alpha, self.mu, self.sigma, alpha_est, mu_est, sigma_est)) # error of first estimate

		for beta in betas:	
			print('Maximization for beta = {}'.format(beta))
			llh_01 = likelihood(alpha_est, X, mu_est, sigma_est, 1)
			llh_1 = likelihood(alpha_est, X, mu_est, sigma_est, beta)

			# define h[k, i] = probability that xi belongs to class k
			# h will be a (K, N, 1, 1)  dimensional array
			# This is the shape for easy multiplication with dataset X
			h = [(alpha_est[k]**beta)*P_gaussian(X, mu_est[k], sigma_est[k], beta)/llh_1 for k in range(K)]
			
			tolerance = np.ones(N)
			if beta == 1:
				thresh = 1e-10

			while np.any(tolerance >= thresh) and steps <= 100000:
				steps += 1
				print("Step {}".format(steps), end='\r')
				llh_00 = llh_01.copy()
				llh_0 = llh_1.copy()
				h_tot = np.sum(h)

				for k in range(K):
					h_tot_k = np.sum(h[k])
					mu_est[k] = np.sum(h[k]*X, axis=1).reshape((n, 1))/h_tot_k

					# Perturb the mu estimates so they split
					# if beta!=1:
					# 	mu_est[k] += np.random.normal(0, 1)

					X_mu = X - mu_est[k]
					h_X_mu = h[k]*X_mu
					for i in range(n):
						sigma_est[k][i] = np.sum(X_mu[i]*h_X_mu, axis=1)
					sigma_est[k] /= h_tot_k

					alpha_est[k] = h_tot_k/h_tot

				llh_1 = likelihood(alpha_est, X, mu_est, sigma_est, beta)
				llh_01 = likelihood(alpha_est, X, mu_est, sigma_est, 1)

				h = [(alpha_est[k]**beta)*P_gaussian(X, mu_est[k], sigma_est[k], beta)/llh_1 for k in range(K)]
				tolerance = np.abs((llh_01-llh_00)/llh_01)

				#errors.append(ds_error(self.alpha, self.mu, self.sigma, alpha_est, mu_est, sigma_est))
				likelihoods.append(np.sum(np.log(llh_01)))
				alpha_ests.append(alpha_est); mu_ests.append(np.array(mu_est))
			print("Steps {}".format(steps))
			beta_step.append((beta, steps-1))

			# print('Beta: {} --- alpha_est: {}, mu_est: {}, sigma_est: {}'.format(beta, alpha_ests, mu_est, sigma_est))
		return alpha_ests, mu_ests, sigma_est, errors, steps, beta_step, likelihoods, actual_likelihood

	def EM_GMM(self, X, thresh, mu_est=None, sigma_est=None, alpha_est=None):
		"""
			Regular Expectation Maximization
		"""
		return self.DAEM_GMM(X=X, thresh=thresh, mu_est=mu_est, sigma_est=sigma_est, alpha_est=alpha_est, betas=[1])
