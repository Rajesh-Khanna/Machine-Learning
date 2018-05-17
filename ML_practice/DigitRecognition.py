from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


model = Sequential()
model.add(Dense(16,input_dim = 784))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.compile(optimizer = 'rmsprop',
	loss = 'categorical_crossentropy',
	metric = ['accuracy'])


mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
y_train = mnist.train.labels

X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
y_test = mnist.test.labels


model.fit(X_train,y_train,epochs = 10,batch_size=100)
score = model.evaluate(X_test, y_test, batch_size=100)
print(score)
del mnist