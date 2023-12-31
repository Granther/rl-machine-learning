import torch
import os
import torch.nn as tnf
#import torch.nn.functional as tnf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from PIL import Image
import torchvision.transforms as tvt


'''
input_tensor = np.random.rand(1, 5, 5)  # Example 1x5x5 tensor
kernel = np.random.rand(3, 3, 1)       # Example 3x3x1 kernel

# Perform convolution
output_feature = np.zeros((3, 3, 1)) 

for i in range(3):
    for j in range(3):
        # Extract the relevant part of the input tensor
        input_section = input_tensor[0, i:i+3, j:j+3]

        # Perform element-wise multiplication and sum
        convolved_value = np.sum(input_section * kernel[:, :, 0])
        
        # Assign to output feature map
        output_feature[i, j, 0] = convolved_value

print(input_tensor)

print(kernel)

print(output_feature)

print(np.convolve([1, 2, 3], [3, 4, 5]))

kernel = torch.tensor([[[1, 1, 1],
                        [1, 2, 1],
                        [1, 1, 1]]])

image = Image.open("/home/grant/Downloads/kirby.jpg")

transform = tvt.ToTensor()

image_tensor = transform(image)

print(image_tensor.shape)

print(image_tensor)
'''
# [1, 2, 3] conv [3, 4, 5]

# 1 * 3 =                       3
# (1 * 4) + (2 * 3) =           10
# (1 * 5) + (2 * 4) + (3 * 3) = 22
# (2 * 5) + (3 * 4) =           22
# 3 * 5 =                       15

# output = [3 ,10, 22, 22, 15]

TENSOR = torch.tensor([[[1, 2, 3], # shape is (1, 3, 3) 
                        [3, 6, 9], # One vector, of a 3 x 3 matrix
                        [5, 6, 7]]])

'''
class SimpleCNN(tnf.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define the layers of the CNN
        self.conv1 = tnf.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.relu = tnf.ReLU()
        self.pool = tnf.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = tnf.Conv2d(16, 32, 3, 1, 1)
        # Add more layers as needed

    def forward(self, x):
        # Define the forward pass
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # Continue through your network
        return x
    

cnn = SimpleCNN()

print(cnn.forward(TENSOR))
'''

tokenizer = Tokenizer(num_words=None)

model = Sequential()

# Embedding layer
model.add(
    Embedding(input_dim=num_words,
              input_length = training_length,
              output_dim=100,
              weights=[embedding_matrix],
              trainable=False,
              mask_zero=True))