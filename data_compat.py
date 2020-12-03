#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

from utils import Config

class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()

    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms

    def create_dataset(self):
        # map id to category
        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        id_to_category = {}
        for k, v in tqdm(meta_json.items()):
            id_to_category[k] = v['category_id']

        # create X, y pairs
        files = open(file,r)
        X = []; y = []
        for x in files:
            if x[:-4] in id_to_category:
                X.append(x)
                y.append(int(id_to_category[x[:-4]]))

        y = LabelEncoder().fit_transform(y)
        print('len of X: {}, # of categories: {}'.format(len(X), max(y) + 1))

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, max(y) + 1

    def create_compatability(self):
        X_compatTrain=[]
        Y_compatTrain=[]
        X_compatValid=[]
        Y_compatValid=[]
        train_comapatability=open(osp.join(self.root_dir, Config['train_compatability']), 'r')
        valid_comapatability=open(osp.join(self.root_dir, Config['valid_compatability']), 'r')
        for pair in train_comapatability:
            pair=pair.strip('\n')
            pair_list=pair.split()
            X_compatTrain.append((pair_list[1]+'.jpg',pair_list[2]+'.jpg'))
            Y_compatTrain.append(int(pair_list[0]))

        for pair in valid_comapatability:
            pair=pair.strip('\n')
            pair_list=pair.split()
            X_compatValid.append((pair_list[1]+'.jpg',pair_list[2]+'.jpg'))
            Y_compatValid.append(int(pair_list[0]))
            
        return X_compatTrain, X_compatValid, Y_compatTrain, Y_compatValid
       
# For category classification
class polyvore_train(Dataset):
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        file1_path = osp.join(self.image_dir, self.X_train[item][0])
        file2_path = osp.join(self.image_dir, self.X_train[item][1])
        X1 = self.transform(Image.open(file1_path))
        X2 = self.transform(Image.open(file2_path))
        y = self.y_train[item]
        return X1, X2, y

class polyvore_test(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return len(self.X_test)

    def __getitem__(self, item):
        file1_path = osp.join(self.image_dir, self.X_test[item][0])
        file2_path = osp.join(self.image_dir, self.X_test[item][1])
        
        X1 = self.transform(Image.open(file1_path))
        X2 = self.transform(Image.open(file2_path))
        y = self.y_test[item]

        # If you want 2 seperate images then reomve torch.cat and put return X1, X2, y
        return X1, X2, y

def get_dataloader(debug, batch_size, num_workers):
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, classes = dataset.create_dataset()

    if debug==True:
        train_set = polyvore_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = polyvore_test(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    else:
        train_set = polyvore_train(X_train, y_train, transforms['train'])
        test_set = polyvore_test(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    datasets = {'train': train_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'test']}
    return dataloaders, classes, dataset_size, X_test


########################################################################
# For Pairwise Compatibility Classification

def get_dataloader_compat(debug, batch_size, num_workers):
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test = dataset.create_compatability()

    if debug==True:
        train_set = polyvore_train(X_train[:200000], y_train[:200000], transform=transforms['train'])
        test_set = polyvore_test(X_test[:200000], y_test[:200000], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    else:
        train_set = polyvore_train(X_train, y_train, transforms['train'])
        test_set = polyvore_test(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    datasets = {'train': train_set, 'test': test_set}

    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'test']}
    return dataloaders, dataset_size

