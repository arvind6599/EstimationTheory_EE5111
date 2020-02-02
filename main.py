import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 512		# dimensions of signal detected
L = 32 		# channel taps

k0 = 6		# sparsity of h vector

n_sigma = 0.1			# noise during estimation
c_sigma = 0.5 ** 0.5 	# channel parameter
lambda_ = 0.2			# channel parameter

QPSK_SYMBOLS = [1+1j, -1+1j, 1-1j, -1-1j]

input_data = np.random.randint(len(QPSK_SYMBOLS), size=N) # to generate the bits

# generate X matrix
X = np.diag([QPSK_SYMBOLS[sym] for sym in input_data])

# print('Input data:')
# print(''.join(['00' if v==0 else '01' if v==1 else '10' if v==2 else '11' for v in input_data]))

p = np.exp(-lambda_ * (np.arange(1,L+1) - 1)).reshape((L,1))
a = np.random.normal(0, c_sigma, (L,1))
b = np.random.normal(0, c_sigma, (L,1))
norm_p = np.sum(p**2)

# h0 is the true channel impulse response vector
h0 = ((a + 1j*b)*p)/norm_p	# h[k] = (a[k] + jb[k])p[k] / norm(p)
print('h_actual')
for i, hk in enumerate(h0):
	print(i+1, ' : ', hk)
print()

# building the F matrix
F = np.array([ [ np.exp((np.pi*2j*i*j)/N) for j in range(L)] for i in range(N)])

XF = X.dot(F)
y = XF.dot(h0) + np.random.normal(0, n_sigma, (N,1)) # process noise
XFH = XF.conjugate().T
XFH_XF_inv = np.linalg.inv(XFH.dot(XF))

######################## QUESTION - 1 Least Squares Estimate ####################################
h1 = XFH_XF_inv.dot(XFH).dot(y)		# estimate 1
print('Q1. h_est')
for i, hk in enumerate(h1):
	print(i+1, ' : ', hk)
print()

################### QUESTION - 2 Least Squares Estimate with sparsity ###########################
# A(h_est) = b; b = [0 0 ... 0]^T
# selected entries are zero
A = np.zeros((L-k0, L))
zero_indices = np.arange(26) #np.random.choice(L, L-k0, replace=False)
for i, j in enumerate(zero_indices):
	A[i][j] = 1 

# Least squares with constraints
XFH_XF_inv_AT = XFH_XF_inv.dot(A.T)
A_XFH_XF_inv_AT_inv = np.linalg.inv(A.dot(XFH_XF_inv_AT))
h2 = h1 - XFH_XF_inv_AT.dot(A_XFH_XF_inv_AT_inv).dot(A).dot(h1)

print('Q2. h_est')
for i, hk in enumerate(h2):
	print(i+1, ' : ', hk)
print()

################### QUESTION - 4 Least Squares Estimate with constraints ##########################
# A(h_est) = b; b = [0 0 0]^T
# selected entries are zero
A = np.zeros((3, L))
A[0][0] = 1; A[0][1] = -1; # h[1] = h[2]
A[1][2] = 1; A[1][3] = -1; # h[3] = h[4]
A[2][4] = 1; A[2][5] = -1; # h[5] = h[6]

# Least squares with constraints
XFH_XF_inv_AT = XFH_XF_inv.dot(A.T)
A_XFH_XF_inv_AT_inv = np.linalg.inv(A.dot(XFH_XF_inv_AT))
h4 = h1 - XFH_XF_inv_AT.dot(A_XFH_XF_inv_AT_inv).dot(A).dot(h1)

print('Q4. h_est')
for i, hk in enumerate(h4):
	print(i+1, ' : ', hk)
print()

plt.figure()
plt.title('Real')
plt.grid(True)
plt.plot([hh[0].real for hh in h0], label='Original')
plt.plot([hh[0].real for hh in h1], label='Q1')
plt.plot([hh[0].real for hh in h2], label='Q2')
plt.plot([hh[0].real for hh in h4], label='Q4')
plt.legend(loc='upper right')

plt.figure()
plt.title('Imaginary')
plt.grid(True)
plt.plot([hh[0].imag for hh in h0], label='Original')
plt.plot([hh[0].imag for hh in h1], label='Q1')
plt.plot([hh[0].imag for hh in h2], label='Q2')
plt.plot([hh[0].imag for hh in h4], label='Q4')
plt.legend(loc='upper right')

plt.show()
