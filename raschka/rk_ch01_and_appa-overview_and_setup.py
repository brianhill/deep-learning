
import torch

# Raschka p. 256

print(torch.__version__)                  # prints 2.7.1
print(torch.cuda.is_available())          # prints False
print(torch.backends.mps.is_available())  # prints True

# Raschka p. 259

tensor0d = torch.tensor(1)

tensor1d = torch.tensor([1, 2, 3])

tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])

tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(tensor2d)

print(tensor3d.dtype)  # Prints torch.int64

float1d = torch.tensor([1.0, 2.0, 3.0])

print(float1d.dtype)   # Prints torch.float32

# Raschka p. 260

print(tensor2d.shape)                # shape is [2, 3]

# If the data is not contiguous, reshape() makes a copy!!
print(tensor2d.reshape(3, 2).shape)  # shape is [3, 2]

# Regardless of contiguousness, the original is unaffected.
print(tensor2d.shape)                # shape is [2, 3]

# The following is probably faster, but requires the data
# to be contiguous and fails it if isn't:
print(tensor2d.view(3, 2).shape)     # shape is [3, 2]

# Raschka p. 261

t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])

t_expected = torch.tensor([[22, 28], [49, 64]])

print(t1.matmul(t2))
print(t1 @ t2)
print(t_expected)

# All three of the above print

# tensor([[22, 28],
#         [49, 64]])
