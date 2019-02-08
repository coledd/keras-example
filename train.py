#!/usr/bin/python

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from keras import initializers

# Load the training data, format: x, y, label
dataset = np.loadtxt('moon_data.csv', delimiter=',')
data = dataset[:,0:2]
labels = dataset[:,2]

# Split the data into training and test data
data_train, data_test, labels_train, labels_test = train_test_split(data, labels)

# Build the model
model = Sequential()
model.add(Dense(2, activation='linear'))
model.add(Dropout(0.001))
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.001))
model.add(Dense(1, activation='sigmoid'))

# Configure a model for mean-squared error regression.
model.compile(loss='mse', optimizer='adam')

checkpointer = keras.callbacks.ModelCheckpoint(filepath="weights.h5", verbose=1, save_best_only=True)

# Train the model
history = model.fit(data_train, labels_train, epochs=5000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[checkpointer])
model.summary()
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# Evaluate
dataset = np.loadtxt('moon_data_test.csv', delimiter=',')
data = dataset[:,0:2]
labels = dataset[:,2]
print(model.evaluate(data, labels))
print(model.metrics_names)

# Predict
print(model.predict(data), labels)
#print(labels)

#for i in range(len(labels)):
#	print(model.predict(data)[i][0], labels[i])

#model.predict(np.array([[1.2,2.3]]))

