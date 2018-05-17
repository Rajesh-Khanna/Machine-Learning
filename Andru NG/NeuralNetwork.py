import numpy as np
from logisticRegression import LogisticRegression

class NeuralNet:
	def __init__(self,layers,Neurons):
		self.layers = []
		self.alpha = 0.1
		self.nuofLayers = layers
		self.Neuronscount = Neurons
		if (layers!=len(Neurons)):
			print("Number of neurons in each layer is not specified")
			exit()
		for i in range(layers):
			self.layers.append([])
			for j in range(Neurons[i]):
				self.layers[i].append(LogisticRegression(self.alpha))
		self.layers = np.array(self.layers)

	def fit(self,data):
		self.data = data
		self.layers[0,:].fit(data)