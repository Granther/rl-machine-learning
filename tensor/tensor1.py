import torch
import numpy as np

# Tensors can be created from scratch

data = [[2,3], [4,5]]

t_data = torch.tensor(data)

# From numpy

np_array = np.array(data)

t_np = torch.from_numpy(np_array)

# Creating a new tensor from another tensor retains the shape and datatype of the parent

t_new_tensor = torch.ones_like(t_data)

TENSOR = torch.tensor([[[1, 2, 3], # shape is (1, 3, 3) 
                        [3, 6, 9], # One vector, of a 3 x 3 matrix
                        [5, 6, 7]]])

print(TENSOR.ndim)

rand_tensor = torch.rand(300, 300)

print(rand_tensor)

rand_image_size_tensor = torch.rand(224, 224, 3) # h, w, color channels

print(rand_image_size_tensor)

zeros = torch.zeros(3, 3)

print(zeros)

arange = torch.arange(0, 10) # gives a vector in the range

print(arange)

torch.arange(start=0, end=1000, step=70) #Steps up by 70 instead of 1

ten_zeroes = torch.zeros_like(arange)

float_32_tensor = torch.tensor([3, 4, 5], dtype=float)

print(float_32_tensor)
