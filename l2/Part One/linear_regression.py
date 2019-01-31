'''
1 Change with initial weights
'''

import math
import sys
import random

import matplotlib.pyplot as plt

def create_data(file_name = "linregdata"):
	data = []
	file = open(file_name)
	for line in file:
		if line[-1]=='\n':
			d = line[:-1].split(',')
		else:
			d = line.split(',')

		if d[0]=='M':
			d[0] = 1
		elif d[0]=='F':
			d[0] = 2
		elif d[0]=='I':
			d[0] = 3

		for i in range(1,len(d)):
			d[i] = float(d[i])
		
		data.append(d)	
		
	return data

def standardize(data,mean=[],std=[]):
	if std==[] or mean==[]:
		N = len(data)
		mean = [0 for x in range(7)]
		std = [1 for x in range(7)]
	
		for i in range(1,8):
			mean[i-1] = sum([d[i] for d in data])/N

		for i in range(1,8):
			std[i-1] = math.sqrt(sum([(d[i]-mean[i-1])**2 for d in data])/N)
	
	for d in data:
		for i in range(1,8):
			d[i] = (d[i]-mean[i-1])/std[i-1]

	return data,mean,std


def mylinridgeregeval(X,weights):
	h = 0
	for x,w in zip(X,weights[1:]):
		h += x*w
	h += weights[0]

	return h


def mylinridgereg(X,Y,Lambda):
	weights = [0 for x in range(9)]
	last_weights = [1 for x in range(9)]
	error = sys.maxsize

	k = 0
	while abs(error)>0.000001 and k<10000 and set(weights)!=set(last_weights):
		last_weights = weights[:]
		for i in range(9):
			for x,y in zip(X,Y):
				h = mylinridgeregeval(x,weights)
				error = y-h	
				if i == 0:
					weights[i] += Lambda*error
				else:
					weights[i] += Lambda*error*x[i-1]
		print "Lambda = ",Lambda,"iteration = ",k,"error = ",error,"weights = ",weights
		k += 1

	return weights


def partition(data,frac=0.2):
	i = 0
	data_test,data_train = [],[]
	L = len(data)
	N = frac*len(data)

	while i<N:
		r = random.randint(0,L-1)
		while data[r] in data_test:
			r= random.randint(0,L-1)
		data_test.append(data[r])
		i+=1

	for d in data:
		if d not in data_test:
			data_train.append(d)

	return data_train, data_test


def meansquarederr(Y,H):
	error = 0
	for y,h in zip(Y,H):
		error += round(y-h,2)**2
		if(error)>100:
			error = 100000
			return error
		
	return error/len(Y)
	

if __name__ == '__main__':
	# data = create_data()
	# #data = [['F',0.61,0.495,0.21,1.548,0.724,0.3525,0.3925,10],['F',0.66,0.515,0.17,1.337,0.615,0.3125,0.3575,10],['F',1,0.515,0.17,1.337,0.615,0.3125,0.3575,10]]
	# data = standardize(data)
	# data_train, data_test = partition(data)
	# train_X = [d[:-1] for d in data_train]
	# train_Y = [d[-1] for d in data_train]
	# test_X = [d[:-1] for d in data_test]
	# test_Y = [d[-1] for d in data_test]
	
	# weights = []
	# # Lambda = [0.001,0.002,0.005,0.007,0.01,0.02,0.05,0.1,0.2,0.5]
	# Lambda = [0.001]
	# for l in Lambda:
	# 	weights.append(mylinridgereg(train_X,train_Y,l))
	
	# h = []
	# error = []
	# for theta in weights:
	# 	h = []
	# 	for x in test_X:
	# 		h.append(mylinridgeregeval(x,theta))
	# 	error.append(meansquarederr(test_Y,h))

	# print "\nMean Squared Errors:\n"
	# for i in range(len(Lambda)):
	# 	print i,":",Lambda[i],"=",error[i],"\n"

	data = create_data()

	fig_test = plt.figure(1)
	ax_test = fig_test.add_subplot(1,1,1)
	ax_test.set_xlabel("Lambda")
	ax_test.set_ylabel("MSE")
	fig_train = plt.figure(2)
	ax_train = fig_train.add_subplot(1,1,1)
	ax_train.set_xlabel("Lambda")
	ax_train.set_ylabel("MSE")

	file = open("results.txt","w")
	file.write("MSE_Train:MSE_Test\n\n")
	Lambda = [0.001,0.002,0.005,0.007,0.01,0.02,0.05,0.1,0.2,0.5,0.7,0.9]
	for i in range(1,10):
		frac = (i*9)/100.0
		data_train, data_test = partition(data,frac)
		data_train, mean, std = standardize(data_train)
		data_test, m , s= standardize(data_test,mean,std)

		train_X = [d[:-1] for d in data_train]
		train_Y = [d[-1] for d in data_train]
		test_X = [d[:-1] for d in data_test]
		test_Y = [d[-1] for d in data_test]

		file.write("i = "+str(i)+"\n")
		Error_Train, Error_Test = [], []
		for l in Lambda:
			theta = mylinridgereg(train_X,train_Y,l)
			h = []
			for x in train_X:
				h.append(mylinridgeregeval(x,theta))
			Error_Train.append(meansquarederr(train_Y,h))

			file.write(str(Error_Train[-1])+":")

			h = []
			for x in test_X:
				h.append(mylinridgeregeval(x,theta))
			Error_Test.append(meansquarederr(test_Y,h))

			file.write(str(Error_Test[-1])+"\n")

		file.write("\n")
		ax_train.scatter(Lambda,Error_Train)
		ax_test.scatter(Lambda,Error_Test)
	
	plt.show()




	




