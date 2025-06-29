from keras.datasets import mnist
import keras
from keras import layers, activations
# import matplotlib.pyplot as plt
import numpy as np

# PDF pages refer to MEAP e3v4
# Subtract 5 to get the printed page numer.
# Or add 5 to get back to the PDF page number.
# Chapter 2 is printed pp. 21-80, inclusive or
# PDF pp. 26-85, inclusive.

# PDF p. 28

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

# PDF p. 36

# digit = original_train_images[4]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# PDF p. 41

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
