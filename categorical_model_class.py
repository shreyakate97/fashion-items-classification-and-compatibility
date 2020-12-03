#!/usr/bin/env python
# coding: utf-8

# In[1]:



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class own_model(nn.Module):
    def __init__(self):
        super(own_model, self).__init__()
        
        # first: CONV => RELU => CONV => RELU => POOL set
        #block 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.norm1_1 = nn.BatchNorm2d(64)
        
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.norm1_2 = nn.BatchNorm2d(64)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        #block 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding = 1)
        self.norm2_1 = nn.BatchNorm2d(128)
        
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding = 1)
        self.norm2_2 = nn.BatchNorm2d(128)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        #block 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding = 1)
        self.norm3_1 = nn.BatchNorm2d(256)
        
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding = 1)
        self.norm3_2 = nn.BatchNorm2d(256)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        #block 4
        self.conv4_1 = nn.Conv2d(256, 256, 3, padding = 1)
        self.norm4_1 = nn.BatchNorm2d(256)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, dilation=2)
        self.dropout4 = nn.Dropout2d(0.25)
        
        # fully connected (single) to RELU
        
        self.fc1 = nn.Linear(256*13*13, 256)
        self.normfc_1 = nn.BatchNorm1d(256)
        self.dropoutfc_1 = nn.Dropout2d(0.50)
        self.fc2 = nn.Linear(256, 153)
        
        
    def forward(self, x):    
        out = x
        out = F.relu(self.norm1_1(self.conv1_1(x)))
        out = F.relu(self.norm1_2(self.conv1_2(out)))
        out = self.pool1(out)
        out = self.dropout1(out)
        
        out = F.relu(self.norm2_1(self.conv2_1(out)))
        out = F.relu(self.norm2_2(self.conv2_2(out)))
        out = self.pool2(out)
        out = self.dropout2(out)
        
        out = F.relu(self.norm3_1(self.conv3_1(out)))
        out = F.relu(self.norm3_2(self.conv3_2(out)))
        out = self.pool3(out)
        out = self.dropout3(out)
        
        out = F.relu(self.norm4_1(self.conv4_1(out)))
        out = self.pool4(out)
        out = self.dropout4(out)
        
        # flatten
        out = out.view(-1, 256 * 13 * 13)
        
        out = F.relu(self.normfc_1(self.fc1(out)))
        out = self.dropoutfc_1(out)
        out = self.fc2(out)
        
        # softmax classifier
        
        return out

model = own_model()


# In[2]:


from torchsummary import summary

summary(model,input_size=(3,224,224))

