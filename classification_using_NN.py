import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to your local dataset (https://www.kaggle.com/datasets/zalando-research/fashionmnist)
train_path = 'C:/Users/tanma/Downloads/archive/fashion-mnist_train.csv'
test_path = 'C:/Users/tanma/Downloads/archive/fashion-mnist_test.csv'

# Load the CSV files
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
# imgIndex = 9
# image = xtrain[imgIndex]
# print("Image Label :",ytrain[imgIndex])
# plt.imshow(image) 
#got an error for "Invalid shape" on the first attempt - resolved using reshape

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
