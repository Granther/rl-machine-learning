from rnn2 import Net 

import torch
import torchvision
import tkinter
from torchvision import transforms, datasets

import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend
import matplotlib.pyplot as plt

import torch.nn as nn # more OOP
import torch.nn.functional as F # more just functions

import torch.optim as optim

train = datasets.MNIST("", train=True, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))
# Te MNIST is a dataset of hand drawn numbers

test = datasets.MNIST("", train=False , download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))
# Convert the data to tensors (cause they are not tensors by default)

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    # How many at a time we want to pass to our model
    # Shuffling helps because the dataset may be organized 

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

plt.imshow(X[0].view(28,28))
plt.show()