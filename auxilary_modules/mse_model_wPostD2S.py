# %%
import os
import logging, timeit, sys
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
from dlkit.log.log_util import (logging_set_up, logging_get_logger)
from dlkit.loss.gaussian import gaussian_loss_fn
# from dlkit.nets.mlp import MLPModel
from dlkit.opt.train import train_epochs

# %%
## General params
hr_data_size = 56
lr_data_size = 8

class MeanNet(nn.Module):

    def __init__(
            self, 
            pre_d2s_filters=None,
            post_d2s_filters=None,
            up_fac=int(hr_data_size/lr_data_size)
            ):
        
        super(MeanNet, self).__init__()

        ## Default hidden layer filter numbers
        if pre_d2s_filters is None:
            pre_d2s_filters = [
                16, 
                16, 
                16, 
                16, 
                up_fac**2
                ]
            ## Copying what I had in TF MSE
            # pre_d2s_filters = [
            #     16, 
            #     32, 
            #     32, 
            #     32, 
            #     up_fac**2
            #     ]
        self.pre_d2s_filters = pre_d2s_filters
        ## Handle post d2s layers
        self.up_fac = up_fac
        if post_d2s_filters is None:
            post_d2s_filters = [
                32, 
                16, 
                1
                ]
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

        # self.conv_m = nn.Conv2d(1, 1, 3, padding='same')
        # self.conv_s = nn.Conv2d(1, 1, 3, padding='same')

        # initialize parameters
        self.init_parameters()

    def forward(self, x, d2s_layer=5):
        
        h = x
        for i, layer in enumerate(self.hidden_layers):
            h = F.relu(layer(h))
            ## Add D2S after final "pre" layer
            if i == len(self.pre_d2s_filters)-1:
                ## Add d2s transformation
                h = F.pixel_shuffle(h, self.up_fac)

        ## For MSE, stop with D2S layer
        mean = h
        # mean = self.conv_m(h)
        # stdd = F.elu(self.conv_s(h)) + 1
        return mean#, stdd
    
    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize hidden layers
        for layer in self.hidden_layers:
            nn.init.kaiming_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.1) ## What should this be? keeping Johann's 
