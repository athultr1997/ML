from PIL import Image
import numpy as np

def sigmoid(X):
	Y = np.zeros(len(X)).reshape(len(X),1)
	
	for i in range(len(X)):
		if X[i]>=0:
			z = np.exp(-X[i])
			Y[i] = 1/(1.0+z)
		else:
			z = np.exp(X[i])
			Y[i] = z/(1.0+z)
	
	return Y


def rgb_to_gs_convertor(im_rgb):
	row,col,depth = im_rgb.shape
	im_gs = np.zeros(row*col).reshape(row,col)
	for i in range(row):
		for j in range(col):
			im_gs[i,j] = 0.2989*im_rgb[i,j,0] + 0.5870*im_rgb[i,j,1] + 0.1140*im_rgb[i,j,2]
			
	return im_gs


def open_image(i):
	im = np.asarray(Image.open("steering/img_"+str(i)+".jpg"))
	im = rgb_to_gs_convertor(im)
	row,col = im.shape
	im = im.reshape(row*col,1)
	return im


def get_actual_output():
	file = open("data.txt","r")
	output = []

	for line in file:
		d = line[:-1].split("\t")
		output.append(float(d[-1]))

	return output


def train_ANN(net_arch=[1024,512,64,1],alpha=0.001,max_epoch=1000,N=24000):
	file = open("result.txt","a")
	L = len(net_arch)
	W,MSE = [],[]
	train_real_output = get_actual_output()
	
	for i in range(L-1):
		W.append(np.random.randint(-100,101,size=(net_arch[i+1],net_arch[i])))

	for i in range(L-1):
		W[i] = W[i].dot(0.0001)

	epoch = 0
	while epoch<max_epoch:
		mse = 0
		for i in range(N):
			Y,Error = [],[]
			Y.append(open_image(i))
			for l in range(L-1):
				Y.append(sigmoid(W[l].dot(Y[l])))
			
			e = np.zeros(len(Y[-1])).reshape(len(Y[-1]),1)
			for k in range(len(Y[-1])):
				e[k] = Y[-1][k]*(1-Y[-1][k])*(train_real_output[i]-Y[-1][k])
			Error.insert(0,e)	

			for l in range(L-2,0,-1):
				e = np.zeros(len(Y[l])).reshape(len(Y[l]),1)
				for k in range(len(Y[l])):
					Sum = 0
					for m in range(len(Error[0])):
						Sum += W[l][m,k]*Error[0][m]
					e[k] = Y[l][k]*(1-Y[l][k])*Sum
				Error.insert(0,e)

			for l in range(L-1):
				for k in range(len(Error[l])):
					for j in range(len(Y[l])):
						W[l][k,j] += alpha*Error[l][k]*Y[l][j]

			print "epoch=",epoch,";sample=",i,";error=",Error[-1][0]
			mse += Error[-1][0]**2
			i+=1

		MSE.append(mse)
		epoch += 1

	file.write(str(MSE))
	file.close()


if __name__ == '__main__':
	train_ANN()

	
