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

# Give it a batch of 10, go through and remove arbritary generalizations

class Net(nn.Module):
    def __init__(self):
        super().__init__() # nn.Module is the parent of Net, thus we must call its super class's init
        self.fc1 = nn.Linear(28*28, 64) # fc = fully connected
        self.fc2 = nn.Linear(64, 64) 
        self.fc3 = nn.Linear(64, 64) 
        self.fc4 = nn.Linear(64, 10) # output layer (10 classes 0-9)

    def forward(self, x): # x is the input, wether from the previous layer or user input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1) #  


X = torch.rand((28,28))
net = Net() 
#net.forward(X)

X = X.view(-1, 28*28) # by changing the shape to -1, we are saying, the shape that -1 represents could be anything. Basically, we can inserte 100 in

output = net(X)

print(output.shape)

optimizer = optim.Adam(net.parameters(), lr=0.001) # learning rate can be adjusted for overfitment
    # too low = wont learn
    # too fast = will overfit to the samples
    # decaying learning rate, starts with large steps and gets smaller as it gets closer to optimal loss

# epoch = a full pass through our entire dataset

EPOCHS = 3

for EPOCHS in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad() # the gradient must be zeroed each time data is passed through so it doesnt keep appending
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

        print("Accur: ", round(correct/total, 3))

plt.imshow(X[0].view(28,28))
plt.show()

print(torch.argmax(net(X[0].view(-1,784))[0]))


'''
for data in trainset:
    print(data)
    break

x, y = data[0][0], [1][0]

print(data[0][0].shape)

print(data[0][0].view([28, 28]).shape)

plt.imshow(data[0][1].view([28, 28]))
plt.show()

print(y)
'''