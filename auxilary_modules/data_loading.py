import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

#import matplotlib.pyplot as plt

# DL-Kit
## for simplicity, add local and kestrel paths
sys.path.append('/projects/ecrpstats/dl-kit') #TODO adjust path
sys.path.append('/Users/hgoldwyn/Research/projects/SR_CNN/dl-kit')

# %%
## General params
hr_data_size = 56
lr_data_size = 8

# %%

def import_data(
    synth_data_dir='', 
    train_test_split=856, 
    batch_size=32
    ):

    if not os.path.exists(synth_data_dir):
        ## try local data dir
        local_data_dir = "/Users/hgoldwyn/Research/projects/SR_CNN/sr_cnn/synthetic_data/data/"
        kestrel_data_dir = "/projects/ecrpstats/sr_cnn/synthetic_data/data/"
        if os.path.exists(local_data_dir):
            synth_data_dir = local_data_dir
        ## try kestrel
        elif os.path.exists(kestrel_data_dir):
            synth_data_dir = kestrel_data_dir
        else: 
            return ValueError("can't find data path")
    else: 
        ## path is arg is fine
        pass
    
    x_HR = np.loadtxt(synth_data_dir + "synthetic_gaussian_data_HR.txt")
    x_LR = np.loadtxt(synth_data_dir + "synthetic_gaussian_data_LR.txt")
    # x_HR = np.loadtxt(synth_data_dir + "synthetic_gaussian_data_HR.txt")
    # x_LR = np.loadtxt(synth_data_dir + "synthetic_gaussian_data_LR.txt")

    ## rescale data, this was emperical
    x_HR = x_HR / 9 + 0.5
    x_LR = x_LR / 9 + 0.5

    train_test_split = batch_size*(train_test_split//batch_size)
    xtrainHR = x_HR[:train_test_split]
    xtestHR =  x_HR[train_test_split:]
    xtrainLR = x_LR[:train_test_split]
    xtestLR =  x_LR[train_test_split:]

    (
        xtrainHR,
        xtestHR,
        xtrainLR,
        xtestLR,
        ) = (
            xtrainHR.astype(np.float32).reshape((-1, 1, hr_data_size, hr_data_size)),
            xtestHR.astype(np.float32).reshape((-1, 1, hr_data_size, hr_data_size)),
            xtrainLR.astype(np.float32).reshape((-1, 1, lr_data_size, lr_data_size)),
            xtestLR.astype(np.float32).reshape((-1, 1, lr_data_size, lr_data_size)),
            )

    print('########################################')
    print('Reshaped data')
    print('- xtrainHR', xtrainHR.shape)
    print('- xtestHR ', xtestHR .shape)
    print('- xtrainLR', xtrainLR.shape)
    print('- xtestLR ', xtestLR .shape)

    return xtrainHR, xtestHR, xtrainLR, xtestLR

# %%
def create_dataloader(x, y, batch_size, labels=None, train_kwargs={'shuffle':False, 'drop_last':False}):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if labels is None:
        dataset = TensorDataset(x, y)
    else: 
        dataset = TensorDataset(x, y, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, **train_kwargs)
    return dataloader