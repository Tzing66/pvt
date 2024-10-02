import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to local dataset
train_path = 'C:/Users/tanma/Downloads/archive/fashion-mnist_train.csv'
test_path = 'C:/Users/tanma/Downloads/archive/fashion-mnist_test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


# All pixel values from the second column onwards
# The first column is the label
xtrain = train_data.iloc[:, 1:].values  
ytrain = train_data.iloc[:, 0].values   

xtest = test_data.iloc[:, 1:].values    
ytest = test_data.iloc[:, 0].values  

print(xtrain.shape)
print(xtest.shape)

#Reshaping into num_of_sample and 28x28
xtrain = xtrain.reshape(-1, 28, 28)  
xtest = xtest.reshape(-1, 28, 28)

#just to SEE a random test image
imgIndex = 9
image = xtrain[imgIndex]
print("Image Label :",ytrain[imgIndex])
plt.imshow(image) 
#got an error for "Invalid shape" on the first attempt - resolved using reshape


# Here we are defining a SEQUENTIAL NN model using Keras (api from tensorflow). 

#Sequential -> a linear stack of layers, where each layer has exactly one input tensor and one output tensor. Layers are added sequentially, one after the other.
#Flatten layer -> reshapes a multi-dimensional input into a one-dimensional array. This layer flattens each 28x28 image into a single vector of 784 values (28 * 28 = 784) [this is basically data prep for the next dense layer]

#Dense layer -> (fully connected layer) 300 neurons. 
# ReLu -> common activation function for hidden layers. It outputs max(0, x), meaning it allows positive values to pass through while setting negative values to zero. Helps introduce non-linearity into the model, allowing it to learn more complex patterns. 
#parameters (weigths + biases) for relu -> Each neuron in this layer has 784 inputs (from prev layer). Weights: 784 input connections × 300 neurons = 235,200, Each neuron has one bias, so 300 biases. Total Parameters: 235,200 + 300 = 235,500.
#Second ReLu -> same as last layer and takes 300 inputs (from prev lauyer), total params -> 300 input connections × 100 neurons = 30,000 + 100 biases -> 30,100.

#Output Layer -> The number of neurons here is 10 because this model is built to classify 10 categories (for example, digits 0 to 9 or fashion items with 10 classes). 
# "softmax" outputs a probability distribution across the 10 classes. Each neuron outputs a value between 0 and 1, and the sum of all the outputs will be 1. The class with the highest probability is the predicted class.
#params -> 1010 (100 from prev layer x 10 neurons + 10 biases)


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
print(model.summary())


xvalid, xtrain = xtrain[:5000], xtrain[5000:]
yvalid, ytrain = ytrain[:5000], ytrain[5000:]

#normalising the values so they are between 0 and 1
xtrain = xtrain / 255.0
xtest = xtest / 255.0

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(xtrain, ytrain, epochs=30, validation_data=(xvalid, yvalid))

new = xtest[:5]
predictions = model.predict(new)
print(predictions)

classes = np.argmax(predictions, axis=1)
print(classes)
