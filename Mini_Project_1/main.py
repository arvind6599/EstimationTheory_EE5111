import numpy as np
import matplotlib.pyplot as plt

TOTAL_ITERATIONS = 1

# Parameters

N = 512		# dimensions of signal detected
L = 32 		# channel taps

k0 = 6		# sparsity of h vector
gb = 180	# guard bands on either side

n_sigma_simulation = [0.1**0.5, 0.01**0.5]	# noise during estimation

c_sigma = 0.5 ** 0.5 	# channel parameter
lambda_ = 0.2			# channel parameter

alpha = 0.1				# regularization

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
print_channel(h0, 'h0')

QPSK_SYMBOLS = [1+1j, -1+1j, 1-1j, -1-1j]

# building the F matrix
F = np.array([ [ np.exp((np.pi*2j*i*j)/N) for j in range(L)] for i in range(N)])
FH = F.conjugate().T

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



############################## For QUESTION 4 ######################
A = np.zeros((L, L-3), dtype=np.complex128)
A[0][0] = 1; A[1][0] = 1; # null space basis vector 1
A[2][1] = 1; A[3][1] = 1; # null space basis vector 2
A[4][2] = 1; A[5][2] = 1; # null space basis vector 3
A[6:,3:] = np.eye(L-6)
P_A = A.dot(np.linalg.inv(A.T.dot(A))).dot(A.T) # Projection matrix into the required nullspace


######################### Beginning Iterations ############################
for n_sigma in n_sigma_simulation:
	print('process noise sigma: ', n_sigma)
	for iteration in range(TOTAL_ITERATIONS):
		process_noise = np.random.normal(0, n_sigma, (N,1)) # process noise
		input_data = np.random.randint(len(QPSK_SYMBOLS), size=N) # to generate the bits

		# generate X matrix
		X_elems = np.array([QPSK_SYMBOLS[sym] for sym in input_data]).reshape(N)
		X = np.diag(X_elems)

		XF = X.dot(F)
		y = XF.dot(h0) + process_noise
		XFH = XF.conjugate().T
		# XFH_XF_inv = np.linalg.inv(XFH.dot(XF))

		######################## QUESTION - 1 Least Squares Estimate ####################################
		X_inv_y = (y.flatten()/X_elems).reshape((N,1))
		h1 = FH.dot(X_inv_y)/N
		Eh1 = ((1.*iteration)/(iteration + 1.) * Eh1) + h1/(iteration + 1)

		################### QUESTION - 2 Least Squares Estimate with sparsity ###########################
		non_zero_indices = np.arange(L-k0, L)
		h2 = np.zeros((L,1), dtype=np.complex128); 
		h2[non_zero_indices] = h1[non_zero_indices]
		Eh2 = ((1.*iteration)/(iteration + 1.) * Eh2) + h2/(iteration + 1)

		######################## QUESTION - 3 Least Squares Estimate Guard Band ##############################
		X1 = X[gb:-gb, gb:-gb]
		X1_F = X1.dot(F[gb:-gb,:])
		y1 = X1_F.dot(h0) + process_noise[gb:-gb]

		X1_FH = X1_F.conjugate().T
		# Applying regularisation
		X1_FH_X1_F_inv_reg = np.linalg.inv(X1_FH.dot(X1_F) + alpha * np.eye(L))
		
		h3_b = X1_FH_X1_F_inv_reg.dot(X1_FH).dot(y1)
		Eh3 = ((1.*iteration)/(iteration + 1.) * Eh3) + h3_b/(iteration + 1)
		


		################### QUESTION - 4 Least Squares Estimate with constraints ##########################
		# A(h_est) = b; b = [0 0 0]^T
		# selected entries are zero
		# Least squares with constraints
		h4 = P_A.dot(h1)
		Eh4 = ((1.*iteration)/(iteration + 1.) * Eh4) + h4/(iteration + 1)
		
		
		################################### Q5 - LOCATING NON ZERO LOCATIONS #########################
		r = y
		S_omp = []
		P = np.zeros((N,N)) # initially empty
		P_ortho = np.identity(N) - P

		for k in range(k0):
			t = np.argmax(abs(XFH.dot(r)))
			S_omp.append(t)

			nc = XF[:, t].reshape((N,1)) # new column
			ncH = nc.conjugate().T

			### Using recursive projection matrix formula (requires no inverses)
			P_ortho_nc = P_ortho.dot(nc) 
			P = P + (P_ortho_nc.dot(ncH.dot(P_ortho))) / np.ravel(ncH.dot(P_ortho_nc))[0]
			P_ortho = np.identity(N) - P
			r = P_ortho.dot(y)

		h5 = np.zeros((L,1), dtype=np.complex128); 
		h5[S_omp] = h1[S_omp]
		Eh5 = ((1.*iteration)/(iteration + 1.) * Eh5) + h5/(iteration + 1)
		
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
