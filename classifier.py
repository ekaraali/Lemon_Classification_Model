import torch
import torch.nn as nn

#Define a CNN network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5)
        
        # fully connected layers
        self.fc1 = nn.Linear(256*27*27, 256)
        self.fc2 = nn.Linear(256, 1)

        #normalization layers
        self.conv2_batchnorm = nn.BatchNorm2d(128)
        self.conv3_batchnorm = nn.BatchNorm2d(256)
        self.fc1_batchnorm = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2_batchnorm(self.conv2(x))))
        x = self.pool(self.relu(self.conv3_batchnorm(self.conv3(x))))
        x = x.view(-1, 256*27*27) #flatten
        x = self.relu(self.fc1_batchnorm(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        
        return x