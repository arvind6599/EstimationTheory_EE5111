import numpy as np
import matplotlib.pyplot as plt

TOTAL_ITERATIONS = 100

# Parameters

N = 512		# dimensions of signal detected
L = 32 		# channel taps

k0 = 6		# sparsity of h vector
gb = 180	# guard bands on either side

n_sigma_simulation = [0.1**0.5, 0.01**0.5]	# noise during estimation

c_sigma = 0.5 ** 0.5 	# channel parameter
lambda_ = 0.2			# channel parameter

alpha = 0.1		# regularization

def print_channel(h, text):
	print(text)
	for i, hk in enumerate(h):
		print(i+1, ' : ', hk)
	print()

p = np.exp(-lambda_ * (np.arange(1,L+1) - 1)).reshape((L,1))
a = np.random.normal(0, c_sigma, (L,1))
b = np.random.normal(0, c_sigma, (L,1))
norm_p = np.sum(p**2)

# h0 is the true channel impulse response vector
h0 = ((a + 1j*b)*p)/norm_p	# h[k] = (a[k] + jb[k])p[k] / norm(p)

QPSK_SYMBOLS = [1+1j, -1+1j, 1-1j, -1-1j]

# building the F matrix
F = np.array([ [ np.exp((np.pi*2j*i*j)/N) for j in range(L)] for i in range(N)])

def plot_channel(hplot, label):
	plt.figure()
	plt.subplot(2,1,1)
	plt.title(label + ' Real and Imaginary')
	plt.grid(True)
	plt.stem([(hh[0]).real for hh in h0], linefmt='g-', markerfmt='go', label='Original', use_line_collection=True)
	plt.stem([(hh[0]).real for hh in hplot], linefmt='r:', markerfmt='ro', label=label, use_line_collection=True)
	plt.legend(loc='upper right')
	plt.subplot(2,1,2)
	plt.grid(True)
	plt.stem([(hh[0]).imag for hh in h0], linefmt='g-', markerfmt='go', label='Original', use_line_collection=True)
	plt.stem([(hh[0]).imag for hh in hplot], linefmt='r:', markerfmt='ro', label=label, use_line_collection=True)
	plt.legend(loc='upper right')

# Expectation values:
Eh1 = np.zeros((L,1), dtype=np.complex128)
Eh2 = np.zeros((L,1), dtype=np.complex128)
Eh3 = np.zeros((L,1), dtype=np.complex128)
Eh4 = np.zeros((L,1), dtype=np.complex128)
Eh5 = np.zeros((L,1), dtype=np.complex128)


for n_sigma in n_sigma_simulation:
	print('process noise sigma: ', n_sigma)
	for iteration in range(TOTAL_ITERATIONS):
		process_noise = np.random.normal(0, n_sigma, (N,1)) # process noise
		input_data = np.random.randint(len(QPSK_SYMBOLS), size=N) # to generate the bits

		# generate X matrix
		X = np.diag([QPSK_SYMBOLS[sym] for sym in input_data])

		XF = X.dot(F)
		y = XF.dot(h0) + process_noise
		XFH = XF.conjugate().T
		XFH_XF_inv = np.linalg.inv(XFH.dot(XF))

		######################## QUESTION - 1 Least Squares Estimate ####################################
		h1 = XFH_XF_inv.dot(XFH).dot(y)		# estimate 1
		Eh1 = (iteration)/(iteration + 1) * Eh1 + h1/(iteration + 1)

		################### QUESTION - 2 Least Squares Estimate with sparsity ###########################
		non_zero_indices = np.arange(L-k0, L)
		h2 = np.zeros((L,1), dtype=np.complex128); 
		for ii in non_zero_indices:
			h2[ii] = h1[ii]
		Eh2 = (iteration)/(iteration + 1) * Eh2 + h2/(iteration + 1)

		######################## QUESTION - 3 Least Squares Estimate Guard Band ##############################
		X1 = X[gb:-gb, gb:-gb]
		X1_F = X1.dot(F[gb:-gb,:])
		y1 = X1_F.dot(h0) + process_noise[gb:-gb]

		X1_FH = X1_F.conjugate().T
		X1_FH_X1_F_inv = np.linalg.inv(X1_FH.dot(X1_F))
		# h3_1a = X1_FH_X1_F_inv.dot(X1_FH).dot(y1)

		# h3_2a = np.zeros((L,1)); 
		# for ii in non_zero_indices:
		# 	h3_2a[ii] = h3_1a[ii]

		# print_channel(h3_1a, 'Q3. (1a) h_est')
		# print_channel(h3_2a, 'Q3. (2a) h_est')

		# Applying regularisation
		X1_FH_X1_F_inv_reg = np.linalg.inv(X1_FH.dot(X1_F) + alpha * np.eye(L))
		h3_b = X1_FH_X1_F_inv_reg.dot(X1_FH).dot(y1)
		Eh3 = (iteration)/(iteration + 1) * Eh3 + h3_b/(iteration + 1)
		


		################### QUESTION - 4 Least Squares Estimate with constraints ##########################
		# A(h_est) = b; b = [0 0 0]^T
		# selected entries are zero
		A = np.zeros((3, L), dtype=np.complex128)
		A[0][0] = 1; A[0][1] = -1; # h[1] = h[2]
		A[1][2] = 1; A[1][3] = -1; # h[3] = h[4]
		A[2][4] = 1; A[2][5] = -1; # h[5] = h[6]

		# Least squares with constraints
		XFH_XF_inv_AT = XFH_XF_inv.dot(A.T)
		A_XFH_XF_inv_AT_inv = np.linalg.inv(A.dot(XFH_XF_inv_AT))
		h4 = h1 - XFH_XF_inv_AT.dot(A_XFH_XF_inv_AT_inv).dot(A).dot(h1)
		Eh4 = (iteration)/(iteration + 1) * Eh4 + h4/(iteration + 1)
		
		
		################################### Q5 - LOCATING NON ZERO LOCATIONS #########################
		r=y
		S_omp=[]
		P = np.zeros((N,N)) # initially empty
		P_ortho = np.identity(N) - P

		for k in range(k0):
			t = np.argmax(abs(AH.dot(r)))
			S_omp.append(t)

			nc = A[:, t].reshape((N,1)) # new column
			ncH = nc.conjugate().T

			# A_S_omp = np.append(A_S_omp, A[:, t].reshape((N,1)), axis=1) # append column t to AS
			# A_S_ompH = A_S_omp.T.conjugate()

			# Moore - Penrose inverse 
			# Following method will not be vector optimized as we are going to perform 512 vector multiplications
			# ASD = np.linalg.inv(A_S_ompH.dot(A_S_omp)).dot(A_S_ompH)
			# P_k = A_S_omp.dot(ASD)

			# A_S_ompH_A_S_omp_inv = np.linalg.inv(A_S_ompH.dot(A_S_omp))
			# r = y - A_S_omp.dot(A_S_ompH_A_S_omp_inv.dot(A_S_ompH.dot(y))) # Requires only k vector multiplications

			### Using recursive projection matrix formula (requires no inverses)

			P_ortho_nc = P_ortho.dot(nc) 
			P = P + (P_ortho_nc.dot(ncH.dot(P_ortho))) / np.ravel(ncH.dot(P_ortho_nc))[0]
			P_ortho = np.identity(N) - P
			r = P_ortho.dot(y)

		h5 = np.zeros((L,1), dtype=np.complex128); 
		for ii in S_omp:
			h5[ii] = h1[ii]
		Eh5 = (iteration)/(iteration + 1) * Eh5 + h5/(iteration + 1)

		if iteration == TOTAL_ITERATIONS - 1:
			print('MSE(h1) = ', np.sum(abs(Eh1-h0)**2)/L)
			print_channel(Eh1, 'Eh1')
			print('MSE(h2) = ', np.sum(abs(Eh2-h0)**2)/L)
			print_channel(Eh2, 'Eh2')
			print('MSE(h3) = ', np.sum(abs(Eh3-h0)**2)/L)
			print_channel(Eh3, 'Eh3')
			print('MSE(h4) = ', np.sum(abs(Eh4-h0)**2)/L)
			print_channel(Eh4, 'Eh4')
			print('MSE(h5) = ', np.sum(abs(Eh5-h0)**2)/L)
			print_channel(Eh5, 'Eh5')
			plot_channel(h1, 'h1')
			plot_channel(h2, 'h2')
			plot_channel(h3_b, 'h3_b')
			plot_channel(h4, 'h4')
			plot_channel(h5, 'h5')
			print('//////////////////////////////////////////////////////////////////')
			plt.show()
