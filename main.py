import numpy as np
import random
import math
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

		#print('COIN ',label[coin],p)
		for j in range(m):
			data[i][j]=signum(random.random()-p)
		#print("probablity = ",np.sum(data[i])/m)
	print(" COIN A Probabilty ",s/n)
	return data


def em(data,pi,p0,q0,m,n):
	thesh=1e-6

	change=1
	while(change>=thesh):

		s1=0
		s2=0
		u=[]
		for i in range(n):
			
			heads=np.sum(data[i])

			#likelihoods
			l1=(p0)**heads*(1-p0)**(m-heads)
			l2=(q0)**heads*(1-q0)**(m-heads)

			u_i=l1*pi/(l1*pi+l2*(1-pi))
			u.append(u_i)

			s1+=u_i*heads
			s2+=(1-u_i)*heads

		#updates
		p1=s1/np.sum(u)/m
		q1=s2/(n-np.sum(u))/m

		change=max(abs(p1-p0),abs(q1-q0))

		p0=p1
		q0=q1
		print("probablities {}".format((p0,q0)), end="\r")

	print("probablities are ",p0,q0)

def em2(data,pi,p0,q0,m,n):
	thesh=1e-8

	change=1
	while(change>=thesh):

		s1=0
		s2=0
		u=[]
		for i in range(n):
			
			heads=np.sum(data[i])

			#likelihoods
			l1=(p0)**heads*(1-p0)**(m-heads)
			l2=(q0)**heads*(1-q0)**(m-heads)

			u_i=l1*pi/(l1*pi+l2*(1-pi))
			u.append(u_i)

			s1+=u_i*heads
			s2+=(1-u_i)*heads

		#update rules with pi as well
		p1=s1/np.sum(u)/m
		q1=s2/(n-np.sum(u))/m
		pi1=sum(u)/n

		change=change=max(abs(p1-p0),abs(q1-q0),abs(pi1-pi))

		pi=pi1
		p0=p1
		q0=q1
		print("probablities {}".format((p0,q0)), end="\r")

	print("estimates are ",pi,p0,q0)

# em with beta prior for class probabilities

def em3(data,pi,p0,q0,m,n,aplha,beta):
	thesh=1e-8

	change=1
	while(change>=thesh):

		s1=0
		s2=0
		u=[]
		for i in range(n):
			
			heads=np.sum(data[i])

			#likelihoods
			l1=(p0)**heads*(1-p0)**(m-heads)
			l2=(q0)**heads*(1-q0)**(m-heads)

			u_i=l1*pi/(l1*pi+l2*(1-pi))
			u.append(u_i)

			s1+=u_i*heads
			s2+=(1-u_i)*heads

		#update rules with pi as well
		p1=s1/np.sum(u)/m
		q1=s2/(n-np.sum(u))/m
		pi1=(sum(u)+n*(alpha-1))/(n*(alpha+beta-1))

		change=change=max(abs(p1-p0),abs(q1-q0),abs(pi1-pi))

		pi=pi1
		p0=p1
		q0=q1
		print("probablities {}".format((p0,q0)), end="\r")

	print("estimates are ",pi,p0,q0)


'''
def monte(p,n):
	k=np.zeros(n)
	for i in range(n):
		k[i]=signum(random.random()-p)
	print("probablity = ",np.sum(k)/n)
'''
#m=[1,10]
#n=[10,100,1000,10000]

#Experiment 1

#a

m=10
n=10000

data1=generate_data(0.5,0.35,0.6,m,n)
#initial estimates
p0=0.45
q0=0.5
pi=0.5
em(data1,pi,p0,q0,m,n)

#b
data2=generate_data(0.25,0.35,0.6,m,n)

#initial estimates
p0=0.45
q0=0.5
pi=0.5
em(data2,pi,p0,q0,m,n)

#c

#initial estimates
p0=0.45
q0=0.5
pi=0.25
em(data2,pi,p0,q0,m,n)

#Experiment 2

#initial estimates
p0=0.45
q0=0.5
pi=0.5
em2(data2,pi,p0,q0,m,n)

em2(data2,pi,q0,p0,m,n)

# Experiment 3

p0=0.45
q0=0.5
pi=0.32
alpha=2
beta=4

em3(data2,pi,p0,q0,m,n,alpha,beta)
