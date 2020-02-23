import numpy as np
import matplotlib.pyplot as plt

N = 512
L = 32

TOTAL_ITERATIONS = 10000

QPSK_SYMBOLS = [1+1j, -1+1j, 1-1j, -1-1j]

n_sigma = 0.1
c_sigma = 0.5 ** 0.5 	# channel parameter
lambda_ = 0.2			# channel parameter
p = np.exp(-lambda_ * (np.arange(1,L+1) - 1)).reshape((L,1))
a = np.random.normal(0, c_sigma, (L,1))
b = np.random.normal(0, c_sigma, (L,1))
norm_p = np.sum(p**2)

F_full = np.array([ [ np.exp((np.pi*2j*i*j)/N) for j in range(L)] for i in range(N)])

F = F_full[:, :32]

FH = F.conjugate().T

# h0 is the true channel impulse response vector
h0 = ((a + 1j*b)*p)/norm_p	# h[k] = (a[k] + jb[k])p[k] / norm(p)

Eh1 = np.zeros((L,1), dtype=np.complex128)
Eh2 = np.zeros((L,1), dtype=np.complex128)
Eh3 = np.zeros((6,1), dtype=np.complex128)

for iteration in range(TOTAL_ITERATIONS):
	process_noise = np.random.normal(0, n_sigma, (N,1)) # process noise
	input_data = np.random.randint(len(QPSK_SYMBOLS), size=N) # to generate the bits

	# generate X matrix
	X_diag = np.array([QPSK_SYMBOLS[sym] for sym in input_data]).reshape(N)
	X = np.diag(X_diag)
	X_inv = np.diag(1/X_diag)

	XF = X.dot(F)
	XFH = XF.conjugate().T
	XFH_XF_inv = np.linalg.inv(XFH.dot(XF))


	y = XF.dot(h0) + process_noise # flattening it
	X_inv_y = (y.flatten()/X_diag).reshape((N,1))


	h1 = XFH_XF_inv.dot(XFH.dot(y)) # this is the regular # LSE
	Eh1 = ((1.*iteration)/(iteration + 1.) * Eh1) + h1/(iteration + 1)
	# h2 = FH.dot(X_inv_y)/N	# no inverses here! 


	non_zero_indices = np.arange(6)
	h2 = np.zeros((L,1), dtype=np.complex128); 
	h2[non_zero_indices] = h1[non_zero_indices]
	Eh2 = ((1.*iteration)/(iteration + 1) * Eh2) + h2/(iteration + 1)

	F6 = F_full[:,:6]
	XF6 = X.dot(F6)
	XF6H = XF6.conjugate().T
	XF6H_XF6_inv = np.linalg.inv(XF6H.dot(XF6))
	h3 = XF6H_XF6_inv.dot(XF6H.dot(y))
	Eh3 = ((1.*iteration)/(iteration + 1) * Eh3) + h3/(iteration + 1)

print(h1.shape)
print('MSE(h1) = ', np.sum(abs(Eh1-h0)**2)/L)
print('MSE(h2) = ', np.sum(abs(Eh2-h0)**2)/L)
Eh3_ = np.zeros((L, 1), dtype=np.complex128)
Eh3_[:6,:] = Eh3
print('MSE(h3) = ', np.sum(abs(Eh3_-h0)**2)/L)
# print(h2.shape)

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
plot_channel(h3, 'h3')
# plot_channel(h2, 'h2')

plt.show()