# %%
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

# # def import_data(
# #     synth_data_dir='', 
# #     train_test_split=856, 
# #     batch_size=32,
# #     ):

# #     if not os.path.exists(synth_data_dir):
# #         ## try local data dir
# #         local_data_dir = "/Users/hgoldwyn/Research/projects/SR_CNN/sr_cnn/synthetic_data/data/"
# #         kestrel_data_dir = "/projects/ecrpstats/sr_cnn/synthetic_data/data/"
# #         if os.path.exists(local_data_dir):
# #             synth_data_dir = local_data_dir
# #         ## try kestrel
# #         elif os.path.exists(kestrel_data_dir):
# #             synth_data_dir = kestrel_data_dir
# #         else: 
# #             return ValueError("can't find data path")
# #     else: 
# #         ## path is arg is fine
# #         pass
    
# #     x_HR = np.loadtxt(synth_data_dir + "synthetic_gaussian_data_HR.txt")
# #     x_LR = np.loadtxt(synth_data_dir + "synthetic_gaussian_data_LR.txt")
# #     # x_HR = np.loadtxt(synth_data_dir + "synthetic_gaussian_data_HR.txt")
# #     # x_LR = np.loadtxt(synth_data_dir + "synthetic_gaussian_data_LR.txt")

# #     ## rescale data, this was emperical
# #     x_HR = x_HR / 9 + 0.5
# #     x_LR = x_LR / 9 + 0.5

# #     train_test_split = batch_size*(train_test_split//batch_size)
# #     xtrainHR = x_HR[:train_test_split]
# #     xtestHR =  x_HR[train_test_split:]
# #     xtrainLR = x_LR[:train_test_split]
# #     xtestLR =  x_LR[train_test_split:]

# #     (
# #         xtrainHR,
# #         xtestHR,
# #         xtrainLR,
# #         xtestLR,
# #         ) = (
# #             xtrainHR.astype(np.float32).reshape((-1, 1, hr_data_size, hr_data_size)),
# #             xtestHR.astype(np.float32).reshape((-1, 1, hr_data_size, hr_data_size)),
# #             xtrainLR.astype(np.float32).reshape((-1, 1, lr_data_size, lr_data_size)),
# #             xtestLR.astype(np.float32).reshape((-1, 1, lr_data_size, lr_data_size)),
# #             )

# #     print('########################################')
# #     print('Reshaped data')
# #     print('- xtrainHR', xtrainHR.shape)
# #     print('- xtestHR ', xtestHR .shape)
# #     print('- xtrainLR', xtrainLR.shape)
# #     print('- xtestLR ', xtestLR .shape)

# #     return xtrainHR, xtestHR, xtrainLR, xtestLR

# # %%
# def create_dataloader(x, y, batch_size, train_kwargs={'shuffle':False, 'drop_last':False}):
#     x = torch.from_numpy(x)
#     y = torch.from_numpy(y)
#     dataset = TensorDataset(x, y)
#     dataloader = DataLoader(dataset, batch_size=batch_size, **train_kwargs)
#     return dataloader

# %% [markdown]
# Model

# %%
class ConvNet(nn.Module):

    def __init__(
            self, 
            pre_d2s_filters=None,
            post_d2s_filters=None,
            up_fac=int(hr_data_size/lr_data_size)
            ):
        
        super(ConvNet, self).__init__()

        ## Default hidden layer filter numbers
        if pre_d2s_filters is None:
            pre_d2s_filters = [
                16, 
                16, 
                16, 
                16, 
                up_fac**2
                ]
        self.pre_d2s_filters = pre_d2s_filters
        ## Handle post d2s layers
        self.up_fac = up_fac
        if post_d2s_filters is None:
            # post_d2s_filters = [
            #     32, 
            #     16, 
            #     1
            #     ]
            post_d2s_filters = []
        self.post_d2s_filters = post_d2s_filters
        
        ## Create hidden layers
        in_size = 1
        self.hidden_layers = nn.ModuleList()
        for filter_num in pre_d2s_filters:
            layer = nn.Conv2d(in_size, filter_num, 3, padding='same') 
            self.hidden_layers.append(layer)
            in_size = filter_num
        ## Post D2S
        in_size = 1
        for filter_num in post_d2s_filters:
            layer = nn.Conv2d(in_size, filter_num, 3, padding='same') 
            self.hidden_layers.append(layer)
            in_size = filter_num

        self.conv_m = nn.Conv2d(1, 1, 3, padding='same')
        self.conv_s = nn.Conv2d(1, 1, 3, padding='same')

        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        
        h = x
        for i, layer in enumerate(self.hidden_layers):
            h = F.relu(layer(h))
            ## Add D2S after final "pre" layer
            if i == len(self.pre_d2s_filters)-1:
                ## Add d2s transformation
                h = F.pixel_shuffle(h, self.up_fac)
        ## Output layers
        mean = self.conv_m(h)
        stdd = F.elu(self.conv_s(h)) + 1
        ## flatten outputs
        mean = torch.flatten(mean, 1)
        stdd = torch.flatten(stdd, 1)
        # stdd.register_hook(
        #     lambda grad: print("Gradients of stdd:", grad))  

        
        return mean, stdd
    
    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize hidden layers
        for layer in self.hidden_layers:
            nn.init.kaiming_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.1) ## What should this be? keeping Johann's 
        ## Initialize output layers
        nn.init.xavier_uniform_(self.conv_m.weight)
        if self.conv_m.bias is not None:
            nn.init.constant_(self.conv_m.bias, 0.5)
        nn.init.kaiming_uniform_(self.conv_s.weight)
        if self.conv_s.bias is not None:
            nn.init.constant_(self.conv_s.bias, 0.2)


class EmpStddCovNet(nn.Module):

    def __init__(
            self, 
            pre_d2s_filters=None,
            post_d2s_filters=None,
            up_fac=int(hr_data_size/lr_data_size)
            ):
        
        super(ConvNet, self).__init__()

        ## Default hidden layer filter numbers
        if pre_d2s_filters is None:
            pre_d2s_filters = [
                16, 
                16, 
                16, 
                16, 
                up_fac**2
                ]
        self.pre_d2s_filters = pre_d2s_filters
        ## Handle post d2s layers
        self.up_fac = up_fac
        if post_d2s_filters is None:
            # post_d2s_filters = [
            #     32, 
            #     16, 
            #     1
            #     ]
            post_d2s_filters = []
        self.post_d2s_filters = post_d2s_filters
        
        ## Create hidden layers
        in_size = 1
        self.hidden_layers = nn.ModuleList()
        for filter_num in pre_d2s_filters:
            layer = nn.Conv2d(in_size, filter_num, 3, padding='same') 
            self.hidden_layers.append(layer)
            in_size = filter_num
        ## Post D2S
        in_size = 1
        for filter_num in post_d2s_filters:
            layer = nn.Conv2d(in_size, filter_num, 3, padding='same') 
            self.hidden_layers.append(layer)
            in_size = filter_num

        self.conv_m = nn.Conv2d(1, 1, 3, padding='same')
        # self.conv_s = nn.Conv2d(1, 1, 3, padding='same')

        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        
        h = x
        for i, layer in enumerate(self.hidden_layers):
            h = F.relu(layer(h))
            ## Add D2S after final "pre" layer
            if i == len(self.pre_d2s_filters)-1:
                ## Add d2s transformation
                h = F.pixel_shuffle(h, self.up_fac)
        ## Output layers
        mean = self.conv_m(h)
        # stdd = F.elu(self.conv_s(h)) + 1
        ## flatten outputs
        mean = torch.flatten(mean, 1)
        # stdd = torch.flatten(stdd, 1)
        
        return mean
    
    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize hidden layers
        for layer in self.hidden_layers:
            nn.init.kaiming_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.1) ## What should this be? keeping Johann's 
        ## Initialize output layers
        nn.init.xavier_uniform_(self.conv_m.weight)
        if self.conv_m.bias is not None:
            nn.init.constant_(self.conv_m.bias, 0.0)
        # nn.init.kaiming_uniform_(self.conv_s.weight)
        # if self.conv_s.bias is not None:
        #     nn.init.constant_(self.conv_s.bias, 0.0)
