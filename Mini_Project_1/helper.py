import numpy as np

import matplotlib.pyplot as plt


# Parameters

N = 512		# dimensions of signal detected
L = 32 		# channel taps

k0 = 6	# sparsity of h vector
gb = 180	# guard bands on either side

# n_sigma_simulation = [0.1**0.5, 0.01**0.5]	# noise during estimation

n_sigma = 0.1
c_sigma = 0.5 ** 0.5 	# channel parameter
lambda_ = 0.2			# channel parameter

alpha = 0.1		# regularization

QPSK_SYMBOLS = [1+1j, -1+1j, 1-1j, -1-1j]

process_noise = np.random.normal(0, n_sigma, (N,1)) # process noise
input_data = np.random.randint(len(QPSK_SYMBOLS), size=N) # to generate the bits

p = np.exp(-lambda_ * (np.arange(1,L+1) - 1)).reshape((L,1))
a = np.random.normal(0, c_sigma, (L,1))
b = np.random.normal(0, c_sigma, (L,1))
norm_p = np.sum(p**2)

# h0 is the true channel impulse response vector
h0 = ((a + 1j*b)*p)/norm_p	# h[k] = (a[k] + jb[k])p[k] / norm(p)

# building the F matrix
F = np.array([ [ np.exp((np.pi*2j*i*j)/N) for j in range(L)] for i in range(N)])
X = np.diag([QPSK_SYMBOLS[sym] for sym in input_data])
XF = X.dot(F)
y = XF.dot(h0) + process_noise



################################### Q5 - LOCATING NON ZERO LOCATIONS #########################
A=XF
AH=A.conjugate().T
r=y
S_omp=[]
A_S_omp=np.zeros((N, 0), dtype=np.complex128)

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


h1 = np.linalg.inv(AH.dot(A)).dot(AH).dot(y)
h2 = np.zeros((L,1), dtype=np.complex128); 
for ii in S_omp:
	h2[ii] = h1[ii]

MSE = np.sum(abs(h2-h0)**2)/L
print('MSE: ', MSE)

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

plot_channel(h2, 'h2')
print(S_omp)
plt.show()