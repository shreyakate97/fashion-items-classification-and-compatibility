#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import os.path as osp
import argparse

Config ={}

Config['root_path'] = '/home/ec2-user/hw4/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''

Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 4
Config['batch_size'] = 64

Config['learning_rate'] = 0.05
Config['num_workers'] = 1

Config['train_compatability']='pairwise_compatibility_train.txt'
Config['valid_compatability']='pairwise_compatibility_valid.txt'

