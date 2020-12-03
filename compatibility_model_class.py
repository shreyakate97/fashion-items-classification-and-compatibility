import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class compat_model(nn.Module):
    def __init__(self):
        super(compat_model, self).__init__()
        
        # first: CONV => RELU => CONV => RELU => POOL set
        #block 1
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        
        #block 2
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        
        #block 3
        self.conv3_1 = nn.Conv2d(64, 64, 3, padding = 1)
        
        # fully connected (single) to RELU
        
        self.fc1 = nn.Linear(64*24*24, 512)
        self.dropoutfc_1 = nn.Dropout2d(0.50)
        
        self.fc2 = nn.Linear(512,1)
        
        
    def forward_once(self, x):    
        out = x
        out = F.relu(self.conv1_1(x))
        out = self.pool1(out)
        
        out = F.relu(self.conv2_1(out))
        out = self.pool2(out)
        
        out = F.relu(self.conv3_1(out))

        # flatten
        out = out.view(-1, 64 * 24 * 24)
        out = self.dropoutfc_1(out)
        out = F.relu(self.fc1(out))
        #out = self.fc2(out)
        
        # softmax classifier
        
        return out
    
    def forward(self, input1, input2):
        output1 = torch.sigmoid(self.forward_once(input1))
        output2 = torch.sigmoid(self.forward_once(input2))

        x = torch.abs(output1 - output2)
        x = self.fc2(x)
        return x
    
model = compat_model()