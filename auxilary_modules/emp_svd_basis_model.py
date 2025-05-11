# %%
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

#import matplotlib.pyplot as plt

# DL-Kit
## for simplicity, add local and kestrel paths
sys.path.append('/projects/ecrpstats/dl-kit') #TODO adjust path
sys.path.append('/Users/hgoldwyn/Research/projects/SR_CNN/dl-kit')

## General params
hr_data_size = 56
lr_data_size = 8

class SingBaseNet(nn.Module):

    def __init__(
            self, 
            pre_d2s_filters=None,
            post_d2s_filters=None,
            up_fac=int(hr_data_size/lr_data_size),
            hr_img_pixels=hr_data_size**2,
            ):
        
        super(SingBaseNet, self).__init__()

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

        self.hr_img_pixels = hr_img_pixels
        
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

        ## Predicting singular values1
        self.sing_1 = nn.Linear(
            in_features=hr_img_pixels, 
            out_features=1, 
            bias=True
            )
        self.sing_2 = nn.Linear(
            in_features=hr_img_pixels, 
            out_features=1, 
            bias=True
            )
        self.sing_3 = nn.Linear(
            in_features=hr_img_pixels, 
            out_features=1, 
            bias=True
            )

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

        h = h.flatten(-2, -1)
        sing_1 = F.elu(self.sing_1(h)) + 1
        sing_2 = F.elu(self.sing_2(h)) + 1
        sing_3 = F.elu(self.sing_3(h)) + 1
        ## flatten outputs
        mean = torch.flatten(mean, 1)
        sing_vals = torch.concat((sing_1, sing_2, sing_3), -2)  # Collapse into column vectors
        # sing.register_hook(
        #     lambda grad: print("Gradients of sing:", grad))  
        
        return mean, sing_vals
    
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
        
        nn.init.kaiming_uniform_(self.sing_1.weight)
        nn.init.kaiming_uniform_(self.sing_2.weight)
        nn.init.kaiming_uniform_(self.sing_3.weight)
        if self.sing_1.bias is not None:
            nn.init.constant_(self.sing_1.bias, 900)
        if self.sing_2.bias is not None:
            nn.init.constant_(self.sing_2.bias, 100)
        if self.sing_3.bias is not None:
            nn.init.constant_(self.sing_3.bias, 5)
