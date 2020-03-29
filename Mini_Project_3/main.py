import numpy as np
import random
import matplotlib.pyplot as plt

thresh = 1e-7


def signum(x):
	if(x>=0):
		return 0
	else:
		return 1

def generate_data(pi,p,q,m,n):
	coin_probability=[q,p]
	label=['A','B']
	# 1 in data means heads and 0 means tails
	data=np.zeros((n,m))
	s=0
	for i in range(n):

		#choosing which coin
		coin=signum(random.random()-pi)
		s+=coin
		p=coin_probability[coin]

		for j in range(m):
			data[i][j]=signum(random.random()-p)
		#print("probablity = ",np.sum(data[i])/m)
	print(" COIN A Probabilty ",s/n)
	return data


def em(data,pi,p0,q0,m,n):
	change=1
	iters=0; iter_vals=[]
	while(change>=thresh):
		iters+=1; iter_vals.append([p0, q0])
		heads=np.sum(data, axis=1)
		#likelihoods
		l1=(p0)**heads*(1-p0)**(m-heads)
		l2=(q0)**heads*(1-q0)**(m-heads)

		u=l1*pi/(l1*pi+l2*(1-pi))

		s1 = np.sum(u*heads)
		s2 = np.sum((1-u)*heads)

		#updates
		p1=s1/np.sum(u)/m
		q1=s2/(n-np.sum(u))/m

		change=max(abs(p1-p0),abs(q1-q0))

		p0=p1
		q0=q1
		print("Iteration: {} - p={}, q={}".format(iters,p0,q0), end="\r")
	print("Iterations: {} - p={}, q={}".format(iters,p0,q0))
	return np.array(iter_vals)

def em2(data,pi,p0,q0,m,n):
	change=1
	iters=0; iter_vals=[]
	while(change>=thresh):
		iters+=1; iter_vals.append([p0, q0, pi])
		heads=np.sum(data, axis=1)
		#likelihoods
		l1=(p0)**heads*(1-p0)**(m-heads)
		l2=(q0)**heads*(1-q0)**(m-heads)

		u=l1*pi/(l1*pi+l2*(1-pi))

		s1 = np.sum(u*heads)
		s2 = np.sum((1-u)*heads)

		#update rules with pi as well
		p1=s1/np.sum(u)/m
		q1=s2/(n-np.sum(u))/m
		pi1=sum(u)/n

		change=max(abs(p1-p0),abs(q1-q0),abs(pi1-pi))

		pi=pi1
		p0=p1
		q0=q1
		print("Iteration: {} - p={}, q={}, pi={}".format(iters,p0,q0,pi), end="\r")
	print("Iterations: {} - p={}, q={}, pi={}".format(iters,p0,q0,pi))
	return np.array(iter_vals)


# em with beta prior for class probabilities
def em3(data,pi,p0,q0,m,n,alpha,beta):
	change=1
	iters=0; iter_vals=[]
	while(change>=thresh):
		iters+=1; iter_vals.append([p0, q0, pi])
		heads=np.sum(data, axis=1)
		#likelihoods
		l1=(p0)**heads*(1-p0)**(m-heads)
		l2=(q0)**heads*(1-q0)**(m-heads)

		u=l1*pi/(l1*pi+l2*(1-pi))

		s1 = np.sum(u*heads)
		s2 = np.sum((1-u)*heads)

		#update rules with pi as well
		p1=s1/np.sum(u)/m
		q1=s2/(n-np.sum(u))/m
		pi1=(np.sum(u)+(alpha-1))/(n+alpha+beta-2)

		change=max(abs(p1-p0),abs(q1-q0),abs(pi1-pi))

		pi=pi1
		p0=p1
		q0=q1
		print("Iteration: {} - p={}, q={}, pi={}".format(iters,p0,q0,pi), end="\r")
	print("Iterations: {} - p={}, q={}, pi={}".format(iters,p0,q0,pi))
	return np.array(iter_vals)

ms = [1,10]
ns = [10,100,1000,10000]
data1={}; data2={}

for n in ns:
	for m in ms:
		data1[(n,m)]=generate_data(0.5,0.35,0.6,m,n)
		data2[(n,m)]=generate_data(0.25,0.35,0.6,m,n)

# Expt 1
plt.figure('Experiment 1').suptitle('Experiment 1', fontsize=16)
for n in ns:
	for m in ms:
		print("n={}, m={}".format(n,m))
		#a
		#initial estimates
		p0=0.45
		q0=0.5
		pi=0.5
		print("Experiment 1 a)")
		iter_vals = em(data1[(n,m)],pi,p0,q0,m,n)
		plt.subplot(3,2,1)
		plt.title(r'a) $\hat{p}$ vs. $k$')
		plt.plot(iter_vals[:,0], label='n={},m={}'.format(n,m))
		plt.grid(True)
		plt.subplot(3,2,2)
		plt.title(r'a) $\hat{q}$ vs. $k$')
		plt.plot(iter_vals[:,1], label='n={},m={}'.format(n,m))
		plt.grid(True)
		plt.legend(loc='upper right')
		#b
		#initial estimates
		p0=0.45
		q0=0.5
		pi=0.5
		print(" Experiment 1 b")
		iter_vals = em(data2[(n,m)],pi,p0,q0,m,n)
		plt.subplot(3,2,3)
		plt.title(r'b) $\hat{p}$ vs. $k$')
		plt.plot(iter_vals[:,0], label='n={},m={}'.format(n,m))
		plt.grid(True)
		plt.subplot(3,2,4)
		plt.title(r'b) $\hat{q}$ vs. $k$')
		plt.plot(iter_vals[:,1], label='n={},m={}'.format(n,m))
		plt.grid(True)
		#c
		#initial estimates
		p0=0.45
		q0=0.5
		pi=0.25
		print(" Experiment 1 c")
		iter_vals = em(data2[(n,m)],pi,p0,q0,m,n)
		plt.subplot(3,2,5)
		plt.title(r'c) $\hat{p}$ vs. $k$')
		plt.plot(iter_vals[:,0], label='n={},m={}'.format(n,m))
		plt.grid(True)
		plt.subplot(3,2,6)
		plt.title(r'c) $\hat{q}$ vs. $k$')
		plt.plot(iter_vals[:,1], label='n={},m={}'.format(n,m))
		plt.grid(True)
plt.subplots_adjust(hspace=0.35)
plt.show()

plt.figure('Experiment 2').suptitle('Experiment 2', fontsize=16)
for n in ns:
	for m in ms:
		print("n={}, m={}".format(n,m))
		#initial estimates
		print(" Experiment 2 a")
		p0=0.45
		q0=0.5
		pi=0.5
		iter_vals=em2(data2[(n,m)],pi,p0,q0,m,n)
		plt.subplot(2,3,1)
		plt.title(r'a) $\hat{p}$ vs. $k$')
		plt.plot(iter_vals[:,0], label='n={},m={}'.format(n,m))
		plt.grid(True)
		plt.subplot(2,3,2)
		plt.title(r'a) $\hat{q}$ vs. $k$')
		plt.plot(iter_vals[:,1], label='n={},m={}'.format(n,m))
		plt.grid(True)
		plt.subplot(2,3,3)
		plt.title(r'a) $\hat{\pi}$ vs. $k$')
		plt.plot(iter_vals[:,2], label='n={},m={}'.format(n,m))
		plt.grid(True)
		plt.legend(loc='upper right')
		print(" Experiment 2 b")
		iter_vals=em2(data2[(n,m)], pi, q0, p0, m, n)
		plt.subplot(2,3,4)
		plt.title(r'b) $\hat{p}$ vs. $k$')
		plt.plot(iter_vals[:,0], label='n={},m={}'.format(n,m))
		plt.grid(True)
		plt.subplot(2,3,5)
		plt.title(r'b) $\hat{q}$ vs. $k$')
		plt.plot(iter_vals[:,1], label='n={},m={}'.format(n,m))
		plt.grid(True)
		plt.subplot(2,3,6)
		plt.title(r'b) $\hat{\pi}$ vs. $k$')
		plt.plot(iter_vals[:,2], label='n={},m={}'.format(n,m))
		plt.grid(True)
plt.subplots_adjust(hspace=0.35)
plt.show()

plt.figure('Experiment 3').suptitle('Experiment 3', fontsize=16)
for n in ns:
	for m in ms:
		print("n={}, m={}".format(n,m))
		print("Experiment 3")
		p0=0.45
		q0=0.5
		pi=0.32
		alpha = [1,1,2,3]; beta = [1,3,6,9]
		for i in range(len(alpha)):
			iter_vals=em3(data2[(n,m)], pi, p0, q0, m, n, alpha[i], beta[i])
			plt.subplot(4,3,3*i+1)
			plt.title(str(i+1)+r') $\hat{p}$ vs. $k$')
			plt.plot(iter_vals[:, 0], label='n={},m={}'.format(n,m))
			plt.grid(True)
			plt.subplot(4,3,3*i+2)
			plt.title(str(i+1)+r') $\hat{p}$ vs. $k$')
			plt.plot(iter_vals[:, 1], label='n={},m={}'.format(n,m))
			plt.grid(True)
			plt.subplot(4,3,3*i+3)
			plt.title(str(i+1)+r') $\hat{p}$ vs. $k$')
			plt.plot(iter_vals[:, 2], label='n={},m={}'.format(n,m))
			plt.grid(True)
plt.subplots_adjust(hspace=0.35)
plt.show()
