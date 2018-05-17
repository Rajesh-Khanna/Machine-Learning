import numpy as np
import matplotlib.pyplot as plt
import math
#data
class LogisticRegression:
	def Jteta(self,j):
		errSum = 0
		for i in range(len(self.x)):
			errSum+=(1/(1+math.exp(-np.vdot(self.x[i],self.coefficients)))-self.y[i])*self.x[i][j]
		errSum = errSum/len(self.x)
		return errSum

	def __init__(self,alpha):
		self.alpha = alpha
		self.errMax = 0.00001

	def fit(self,data):
		self.data = data
		self.x = np.array(data)
		self.mi = np.amin(self.x)
		self.mx = np.amax(self.x)
		z = np.zeros((len(data),1))+1
		self.x = np.append(self.x,z,axis = 1)
		self.y = Y
		self.coefficients = np.zeros((len(self.x[0]),1))
		temp = self.coefficients
		cnt = 0
		count = np.zeros((len(self.coefficients),1))
		plt.scatter(self.data[:,0],self.data[:,1],c = self.y,s = 20)
		# print(self.mx)
		# print(self.valu([self.mx,1]))
		
	def backprop(Y):
		while True:
			for i in range(len(self.coefficients)):
				if count[i]!= 1 :
					# print(temp[i])
					temp[i] = 1*(self.coefficients[i]-self.alpha*self.Jteta(i))
			# 		if(abs(temp[i]-self.coefficients[i])<self.errMax):
			# 			count[i] = 1
			# if np.sum(count) == len(self.coefficients):
			# 	break
			self.coefficients = temp*1
			cnt+=1
			# print(temp)
			# c = input(" ")
			# plt.draw()
			# plt.pause(0.0001)
			if(cnt == 1000):
				break
		# print(cnt)

	def expected(self,x):
		if(1/(1+math.exp(-np.vdot(x,self.coefficients)))>0.5):
			return 1
		return 0

	def predict(self,X):
		z = np.zeros((len(X),1))+1
		X = np.append(X,z,axis = 1)
		guess = []
		for c in X:
			guess.append(self.expected(c))
		guess = np.array(color)
		return guess

	def draw(self,X):
		z = np.zeros((len(X),1))+1
		X = np.append(X,z,axis = 1)
		print(self.coefficients)
		color= []
		for c in X:
			color.append(self.expected(c))
		color = np.array(color)
		plt.scatter(X[:,0],X[:,1],c = color,s = 5)
		# self.predict();
		# plt.plot([self.mi,self.mx], [self.valu([self.mi,1]),self.valu([self.mx,1])])
		plt.show()
			
'''			
LR = LinearRegression(0.1)
x = np.random.rand(200,2)
y = []
for i in x: 
		if(i[0]+i[1]-1)>0:
			y.append(1)
		else:
			y.append(0)
y= np.array(y)
LR.fit(x,y)
# LR.predict()
# LR.predict(X)
# print(LR.predict([0.5,0.2,1]))
X = np.random.rand(100,2)
LR.draw(x)
'''