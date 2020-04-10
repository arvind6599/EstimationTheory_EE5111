import numpy as np 

def P_gaussian(x, mu, sigma, beta):
	N, n, hx = x.shape
	if hx != 1:
		print('Wrong dimensions... X needs to be an N-length array of n x 1 vectors')
		return	
	if n > 1:
		wsig, vsig = np.linalg.eig(sigma)
		sig_inv = vsig.T.dot(np.diag(1/wsig).dot(vsig)) # sigma inverse
	else:
		wsig = np.array([sigma]); vsig = np.array([1])
		sig_inv = 1/sigma
		
	return np.array([(((1/((2*np.pi)**n/2)*abs(np.prod(wsig)))*np.exp(-((x[i]-mu).T.dot(sig_inv.dot(x[i]-mu)))/2))**beta) for i in range(N)])

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
	
	def DAEM_GMM(self, X, thresh, mu_est=None, sigma_est=None, alpha_est=None, betas=[0.8,1.], K=2):
		"""
			Deterministic Annealing EM Algorithm for k n-dimensional Gaussians

			X is an N-length array of Xis; Xi is an n-dimensional vector
		"""
		N, n, hx = X.shape
		if hx != 1:
			print('Wrong dimensions... X needs to be an N-length array of n x 1 vectors')
			return

		errors = []
		alpha_ests = []; mu_ests=[]; likelihoods = []
		beta_step = []
		steps = 0

		# Initial estimates
		if alpha_est is None:
			alpha_est = np.array([1./K for j in range(K)])
		if mu_est is None:
			mu_est = [X[int(np.random.random()*N)] for j in range(K)]
		if sigma_est is None:
			sample_mean = np.sum(X, axis=0)/N
			cov = (X[0]-sample_mean).dot((X[0]-sample_mean).T)
			for i in range(1, N):
				cov += (X[i]-sample_mean).dot((X[i]-sample_mean).T)
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
					mu_est[k] = np.sum(h[k]*X, axis=0)/h_tot_k

					# Perturb the mu estimates so they split
					if beta!=1:
						mu_est[k] += np.random.normal(0, 1)

					h_X_mu = h[k]*(X-mu_est[k])
					for i in range(N):
						sigma_est[k] += (h_X_mu[i]*(X[i]-mu_est[k]).T)
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
