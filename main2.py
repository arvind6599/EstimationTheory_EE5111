#MINI - PROJECT 2

import numpy as np
import matplotlib.pyplot as plt

cg=1.78
n=[1,10,100,1000,10000]
a=1
scale_laplace=1/np.power(2,0.5)			
scale_cauchy=np.power(2*cg,0.5)		# gamma for cauchy distribution
a1=[]									# A estimate for normal noise
a2=[]									# A estimate for laplace noise
a3=[]									# A estimate for cauchy noise
v1=[]									# variance of the estimator (normal) 
v2=[]									# variance of the estimator (laplace)


def newton_rhapson(x,tol):
	
	theta_0=np.median(x)

	b=(x-theta_0)/scale_cauchy
	l_1=2*sum(b /(1 + b*b))

	while(np.abs(l_1) > tol):

		#print(theta_0)

		l_2=2*sum((b*b - 1)/((1 + b*b)*(1 + b*b)))

		theta_0 = theta_0 - l_1/l_2

		b=(x - theta_0)/scale_cauchy

		l_1=2*sum(b /(1 + b*b))

	return theta_0


for i in n:

	x1=a*np.ones(i).reshape((i,1))+np.random.normal(0,1,(i,1))
	x2=a*np.ones(i).reshape((i,1))+np.random.laplace(0,scale_laplace,(i,1))
	x3=a*np.ones(i).reshape((i,1))+scale_cauchy*np.random.standard_cauchy((i,1))



	Ae_1=np.sum(x1)/i
	Ae_2=np.median(x2,axis=0)

	'''
	y1=x1-Ae_1*np.ones(i).reshape((i,1))
	y2=x2-Ae_2*np.ones(i).reshape((i,1))
	'''

	#variance assuming unbiased estimate
	y1=x1-a
	y2=x2-a

	v1.append(sum(np.square(y1))/i)
	v2.append(sum(np.square(y2))/i)

	# ESTIMATES
	a1.append(Ae_1)
	a2.append(Ae_2)
	a3.append(newton_rhapson(x3,0.0001))





print(a1)
print(a2)
print(a3)
print(v1)
print(v2)

