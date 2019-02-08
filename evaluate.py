#!/usr/bin/python

import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Build the model
model = Sequential()
model.add(Dense(2, activation='linear', input_dim=2))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.load_weights('weights.h5')

# Configure a model for mean-squared error regression.
model.compile(loss='mse', optimizer='adam')

# Evaluate
dataset = np.loadtxt('moon_data_test.csv', delimiter=',')
data = dataset[:,0:2]
labels = dataset[:,2]

print(model.evaluate(data, labels))
print(model.metrics_names)

# Plot decision boundary
for y in np.linspace(300,-300,100):
	for x in np.linspace(-300,300,100):
		if ((model.predict(np.array([[x/100, y/100]]))) > 0.5):
			print('1', end='')
		else:
			print('0', end='')
	print()

# Plot probability map
for y in np.linspace(300,-300,55):
	for x in np.linspace(-300,300,55):
		p=model.predict(np.array([[x/100, y/100]]))[0][0]
		if(p<0):
			p=0
		if(p>1):
			p=1
		plt.plot(x/100, y/100, 's', color=(1-p,0,p));
plt.show()

# Deployment -- Here's how you would use the trained neural network in an application. 
# Replace the [1, 1] with any x, y that you would like to predict.
round(model.predict(np.array([[1, 1]]))[0][0])
