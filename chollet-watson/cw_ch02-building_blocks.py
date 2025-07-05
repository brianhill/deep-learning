from keras.datasets import mnist
import keras
from keras import layers, activations
# import matplotlib.pyplot as plt
import numpy as np

# Grus Chapter 19 is a better place to learn the material
# in Chollet & Watson Chapter 2.

# PDF pages refer to MEAP e3v4. To purchase,
# see https://www.manning.com/books/deep-learning-with-python-third-edition
# Given a PDF page, subtract 5 to get the printed page
# number, or add 5 to get back to the PDF page number.

# Chapter 2 is printed pp. 21-80, inclusive (or PDF pp. 26-85).

# PDF p. 27 - Training a neural network for digit recognition

# Creates a local cache at ~/.keras/datasets/mnist.npz
(original_train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

train_images = original_train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# model.fit(train_images, train_labels, epochs=5, batch_size=128)
#
# test_digits = test_images[0:10]
# predictions = model.predict(test_digits)
# print(predictions[0])
# print(predictions[0].argmax)
# print(predictions[0][7])
# print(test_labels[0])

# PDF p. 35 - Display a training image

# digit = original_train_images[4]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# PDF p. 41 - Tensor operations

array_3x2 = np.array([[1, 2],
                      [3, -4],
                      [5, 6]])

array_2x4 = np.array([[7, 8, 9, 10],
                      [11, 12, 13, 14]])

array_3x4 = np.matmul(array_3x2, array_2x4)
print(array_3x4)

relu_3x4 = activations.relu(array_3x4)
print(relu_3x4)

# PDF p. 42 - Element-wise operations

# PDF p. 46 - Tensor product

# PDF p. 50 - Tensor reshaping

# PDF p. 51 - Geometric interpretation of tensor operations

# PDF p. 56 - A geometric interpretation of deep learning

# PDF p. 57 - The engine of neural networks: gradient-based optimization

# PDF p. 63 - Stochastic gradient descent

# PDF p. 68 - Chaining derivatives - the backpropagation algorithm

# PDF p. 76 - Looking back at our first example

# PDF p. 78 - Reimplementing our first example from scratch
