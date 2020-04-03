import numpy as np 
import matplotlib.pyplot as plt 
from algos import Solver

N = 1000		# number of samples
K = 2			# number of mixed Gaussians
mus = [5]		# means of the mixed Gaussians
sigma = [2.5, 2.5]		# std deviation

def toss(alpha):
	x = np.random.random()
	if(x<=alpha):
		return 0
	else:
		return 1

alphas = [0.6] #np.linspace(0.6, 0.6, 1) # mixing coefficients

colors = ['red', 'pink', 'brown']


for mu_iter in mus:
	mu = [-mu_iter, mu_iter]
	errorss_daem = []
	errorss_em = []
	bs = []

	for alpha in alphas:
		solver = Solver(alpha=alpha, mu=mu, sigma=sigma)

		X = np.array([np.random.normal(mu[toss(alpha)], sigma[toss(alpha)]) for j in range(N)])
		print('Actual:')
		print('alpha: {}, mu: {}, sigma: {}'.format(alpha, mu, sigma))
		alpha_est_daem, mu_est_daem, sigma_est_daem, errors_daem, steps, beta_step = solver.DAEM_GMM(X=X, thresh=1e-6)
		errorss_daem.append(alpha_est_daem)
		bs.append(beta_step)
		print('DAEM')
		print('alpha_est: {}, mu_est: {}, sigma_est: {}'.format(alpha_est_daem[-1], mu_est_daem, sigma_est_daem))
		alpha_est_em, mu_est_em, sigma_est_em, errors_em, steps, __ = solver.EM_GMM(X=X, thresh=1e-10)
		errorss_em.append(alpha_est_em)
		print('EM')
		print('alpha_est: {}, mu_est: {}, sigma_est: {}'.format(alpha_est_em[-1], mu_est_em, sigma_est_em))
		print()

	plt.figure('DAEM')
	plt.title(r'DAEM, $\hat{\alpha}$ vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	#plt.title(r'DAEM Error vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	for i, alpha in enumerate(alphas):
		plt.plot(errorss_daem[i], #'x-', 
			label=r'$\alpha=$'+str(alpha))
		for _bs in bs[i]:
			plt.axvline(x=_bs[1], color=colors[i], ls=':', lw=1, label=str(_bs[0]))
	plt.grid(True)
	plt.legend(loc='upper right')

	plt.figure('EM')
	plt.title(r'EM, $\hat{\alpha}$ vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	# plt.title(r'EM Error vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')'))
	for i, alpha in enumerate(alphas):
		plt.plot(errorss_em[i], #'x-', 
			label=r'$\alpha=$'+str(alpha))
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