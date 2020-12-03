#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp
import numpy as np

from utils import Config
from model import model
from data import get_dataloader_compat

def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):
    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    acc_list = []
    loss_list = []
    test_acc_list= []
    test_loss_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for input1, input2, labels in tqdm(dataloaders[phase], position=0, leave=True):
                input1 = input1.to(device)
                input2 = input2.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(input1, input2)
                    outputs = torch.reshape(outputs, (outputs.shape[0],))
                    outputs = outputs.type(torch.DoubleTensor)
                    labels = labels.type(torch.DoubleTensor)

                    pred = []
                    for i in outputs:
                      if i>0.5:
                        pred.append(0)
                      else:
                        pred.append(1)
                    
                    pred = torch.FloatTensor(pred)

                    loss = criterion(outputs,labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * input1.size(0)
                running_corrects += torch.sum(pred==labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            if phase=='train':
              acc_list.append(epoch_acc)
              loss_list.append(epoch_loss)
            elif phase=='test':
              test_acc_list.append(epoch_acc)
              test_loss_list.append(epoch_loss)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase=='test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth'))
        print('Model saved at: {}'.format(osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth')))

    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))

    np.savetxt('acc_list.txt',acc_list)
    np.savetxt('test_acc_list.txt',test_acc_list)
    np.savetxt('loss_list.txt',loss_list)
    np.savetxt('test_loss_list.txt',test_loss_list)

if __name__=='__main__':

    dataloaders, dataset_size = get_dataloader_compat(debug=False, batch_size=Config['batch_size'], num_workers=Config['num_workers'])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config['learning_rate'])
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

    train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size)

