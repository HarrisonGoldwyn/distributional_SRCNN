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

def import_data(
    region,
    subregion,
    train_fraction=.7, 
    hr_data_size=64,
    lr_data_size=8,
    order='(subregion, time)'
    ):

    try:
        data_hr = np.load("/Users/hgoldwyn/Research/projects/SR_CNN/paper_repo/data/subregions_wind_u_64x64.npy")
        data_lr = np.load("/Users/hgoldwyn/Research/projects/SR_CNN/paper_repo/data/subregions_wind_u_8x8_downscaled64x64.npy")
    except:
        pass
    try:
        data_hr = np.load("/projects/ecrpstats/distributional_SRCNN/data/subregions_wind_u_64x64.npy")
        data_lr = np.load("/projects/ecrpstats/distributional_SRCNN/data/subregions_wind_u_8x8_downscaled64x64.npy")
    except:
        pass

    if type(subregion) == int:
        data_hr = data_hr[region, subregion]
        data_lr = data_lr[region, subregion]
    
    elif subregion == 'all':    
        if order == '(subregion, time)':
            ## To order by timeshape, each 4 images being the 4 subregions
            data_hr = data_hr[region]
            data_lr = data_lr[region]
            # Transpose so axis 0 (4) comes after axis 1 (214)
            data_hr = np.transpose(data_hr, (1, 0, 2, 3))  # shape (214, 4, 64, 64)
            data_lr = np.transpose(data_lr, (1, 0, 2, 3)) 
            # Now reshape to (214*4, 64, 64)
            data_hr = data_hr.reshape((-1, 64, 64))  # shape (856, 64, 64)
            data_lr = data_lr.reshape((-1, 8, 8))
        elif order == '(time, subregion)':
            data_hr = data_hr.reshape((5, 4*214, hr_data_size, hr_data_size))
            data_hr = data_hr[region]
            data_lr = data_lr.reshape((5, 4*214, lr_data_size, lr_data_size))
            data_lr = data_lr[region]
        else: 
            raise ValueError("Unimplimented 'order'")
        
    else:
        raise ValueError("subregion invalid")

    # %%
    train_set_size = int(data_hr.shape[0] * train_fraction)
    test_set_size = data_hr.shape[0] - train_set_size

    # %%
    ## Normalize data
    raw_std = data_hr.std()
    data_hr = data_hr / raw_std
    rescaled_mean = data_hr.mean()
    data_hr = data_hr - rescaled_mean
    data_lr = data_lr / raw_std - rescaled_mean

    # %%
    ## Get train and test sets
    xtrainHR = data_hr[:train_set_size].astype(np.float32)[:, None, :, :] 
    xtestHR = data_hr[train_set_size:train_set_size+test_set_size].astype(np.float32)[:, None, :, :] 
    xtrainLR = data_lr[:train_set_size].astype(np.float32)[:, None, :, :] 
    xtestLR = data_lr[train_set_size:train_set_size+test_set_size].astype(np.float32)[:, None, :, :]         

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