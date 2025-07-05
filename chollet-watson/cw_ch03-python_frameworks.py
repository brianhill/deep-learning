# PDF p. 86 - History

# 2009
#    Theano
# 2015
#    Keras
#    TensorFlow (Google)
# 2016
#     PyTorch (Meta)
# 2018
#     JAX (Google)

# On Mac, TensorFlow needs tensorflow-metal installed for performance

# Example of TensorFlow on Mac
# import tensorflow as tf
# # Check if GPU is available
# if tf.config.list_physical_devices('GPU'):
#     print("GPU is available via tensorflow-metal")
# else:
#     print("GPU not available, falling back to CPU")
#
# # Define a tensor and move it to GPU
# x = tf.constant([1.0, 2.0, 3.0])
# with tf.device('/GPU:0'):
#     y = x * 2  # Perform operation on GPU
#     print(y.numpy())  # Output: [2. 4. 6.]

# On Mac, PyTorch already has mps (mps stands for Metal Performance Shaders)

# Example of PyTorch on Mac
# import torch
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     x = torch.tensor([1.0, 2.0, 3.0]).to(device)
#     y = x * 2  # Runs on GPU via MPS
#     print(y)

# PDF p. 90 - Introduction to TensorFlow

# THIS MESS IS STARTING TO MAKE ME RECONSIDER USING KERAS.

# WHY NOT JUST USE PYTORCH DIRECTLY?!?
