# Code is copied with necessary refactoring from https://github.com/jayroxis/Cophy-PGNN 

import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

# Define the ConvNet architecture
class ConvNet(nn.Module):
    def __init__(self, dataset_name):
        super(ConvNet, self).__init__()

        self.img_size = 4
        self.ch_size = 3

        self.conv1 = nn.Conv2d(self.ch_size, 8, kernel_size=3, padding=1)  # 1 input channel, 8 output channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # 8 input channels, 16 output channels
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 8 input channels, 16 output channels
        self.fc1 = nn.Linear(32 * self.img_size * self.img_size, 120)   # Fully connected layer with 120 neurons
        self.fc2 = nn.Linear(120, 10)           # Output layer with 10 neurons (10 classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 32 * self.img_size * self.img_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):
    def __init__(self, dataset_name):
        super(MLP, self).__init__()

        self.img_size = 32
        self.ch_size = 3

        self.fc1 = nn.Linear(self.img_size*self.img_size*self.ch_size, 16)  # First fully connected layer
        self.fc4 = nn.Linear(16, 10)      # Output layer

    def forward(self, x):
        x = x.view(-1, self.img_size*self.img_size*self.ch_size)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
        
def get_DNN(dataset_name, model_name, device):
    if model_name == "cnn":
        return ConvNet(dataset_name).to(device)
    
    return MLP(dataset_name).to(device)