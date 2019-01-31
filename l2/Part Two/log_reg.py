import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

def load_data(file_name="credit.txt"):
	X_1,X_2,X_3,X_4,X_5,Y = [],[],[],[],[],[]
	file = open(file_name)

	for line in file:
		d = line.split(',')
		X_1.append(float(d[0]))
		X_2.append(float(d[1]))
		X_3.append(X_1[-1]*X_2[-1])
		X_4.append(X_1[-1]**2)
		X_5.append(X_2[-1]**2)
		Y.append(float(d[2]))

	for i in range(len(Y)):
		if Y[i]==0:
			plt.scatter(X_1[i],X_2[i],color="blue")
		else:
			plt.scatter(X_1[i],X_2[i],color="red")

	# plt.show()
	return np.array(X_1),np.array(X_2),np.array(X_3),np.array(X_4),np.array(X_5),np.array(Y)	

def find_error(h,Y):
	error = 0
	for i in range(len(Y)):
		error += (Y[i]-h[i])**2

	return error


def calculate_h(X,weights):
	n,N = X.shape
	h = X.transpose().dot(weights)
	h = np.array([sigmoid(x) for x in h]).reshape(N,1)
	return h

def newton_raphson(X,Y):
	n,N = X.shape
	weights = np.array([float(random.randint(-100,100))/100 for x in range(n)]).reshape(n,1)
	error = sys.maxsize
	h = calculate_h(X,weights)
	i=0

	print "initial weights = ",weights,"\n"

	while i<10000 and error>0.000001: 
		H = np.zeros(n**2).reshape(n,n)
		grad_l = np.zeros(n).reshape(n,1)
		h = calculate_h(X,weights)

		for j in range(n):
			for k in range(j,n):
				for l in range(N):
					H[j,k] += (-1)*X[k,l]*X[j,l]*h[l]*(1-h[l])
					if k==n-1:
						grad_l[j] += X[j,l]*(Y[l]-h[l])
				
				H[k,j] = H[j,k]

		H_inv = np.linalg.pinv(H)
		P = H_inv.dot(grad_l)
		weights -= P
		error = find_error(h,Y)
		
		print "Iteration = ",i,"; error = ",error,"; weights = ",weights.reshape(1,6)
		# ,"; H =\n",H,";\nH_inv =\n",H_inv,";\nGradient =",grad_l,";\nP =\n",P
		i+=1

def gradient_descent(X,Y,alpha = 0.001):
	n,N = X.shape
	weights = np.array([float(random.randint(-100,100))/100 for x in range(n)]).reshape(n,1)
	error = sys.maxsize
	i = 0

	while i<10000 and error>0.00001:	
		h = calculate_h(X,weights)
		for j in range(n):
			Sum = 0
			for k in range(N):
				Sum += (Y[k]-h[k])*X[j,k]

			weights[j] += alpha*Sum
		error = find_error(h,Y)
		print "Iteration = ",i,"; error = ",error,"; weights = ",weights	
		i += 1

	return error,weights


if __name__ == '__main__':
	file = open("result.txt","a")


	X_1,X_2,X_3,X_4,X_5,Y = load_data()
	X = np.zeros(6*len(Y)).reshape(6,len(Y))
	X[0:] = np.array([1 for x in range(len(Y))])
	X[1:] = np.array(X_1)
	X[2:] = np.array(X_2)
	X[3:] = np.array(X_3)
	X[4:] = np.array(X_4)
	X[5:] = np.array(X_5)

	X = X[:,0:10]
	Y = Y[:10]
	newton_raphson(X,Y)
	# Alpha = [0.00001,0.0001,0.0003,0.0007,0.001,0.002,0.005,0.007,0.01,0.02,0.05,0.1,0.2,0.5,0.7,0.9]
	# for a in Alpha:
	# 	error,weights = gradient_descent(X,Y,a)
	# 	file.write("Alpha:"+str(a)+"  "+"Error:"+str(error)+"  "+"Weights"+str(weights)+"\n")

	file.close()
	

