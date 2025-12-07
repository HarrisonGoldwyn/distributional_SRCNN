import os
import sys
import numpy as np
import torch
# from torch.utils.data import TensorDataset, DataLoader

# DL-Kit
sys.path.append('/Users/hgoldwyn/Research/projects/SR_CNN/dl-kit') #TODO adjust path
sys.path.append('/projects/ecrpstats/dl-kit') #TODO adjust path
from dlkit.log.log_util import (logging_set_up, logging_get_logger)
# from dlkit.nets.mlp import MLPModel
from dlkit.opt.train import train_epochs
from dlkit.nets.conv2d import Conv2dUpscaleModelInterpolate

## Load model module
sys.path.append('/Users/hgoldwyn/Research/projects/SR_CNN/paper_repo/auxilary_modules') #TODO adjust path
sys.path.append('/projects/ecrpstats/distributional_SRCNN/auxilary_modules')
import data_loading

import matplotlib.pyplot as plt

train_or_load = 'train'
# %%
hr_data_size = 64
lr_data_size = 8
# %%
epochs = 300
batch_size = 32
##
region = 0
subregion = 'all'

log_file_base_path = os.path.basename(__file__)[:-3]
print(f"Running {log_file_base_path}")

save_path = f'{log_file_base_path}.pt'

## ~~~~~~~~~~~## import data
xtrainHR, xtestHR, xtrainLR, xtestLR = data_loading.import_data(
    region,
    subregion,
    train_fraction=.7,
    order='(time, subregion)'
    )
# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
    )
print(f"Using device: {device}")

# %% [markdown]
# Load data

# %%
# try:
#     test_mse_error_fields = np.load("/Users/hgoldwyn/Research/projects/SR_CNN/sr_cnn/real_data/torch/scripts/interp_upscale/HR_64x64/mse/standard_normed_data/error_fields/mse_5l_i123_c32s_padR_schLrG0p95_reg0_errFields.npy")
#     train_mse_error_fields = np.load("/Users/hgoldwyn/Research/projects/SR_CNN/sr_cnn/real_data/torch/scripts/interp_upscale/HR_64x64/mse/standard_normed_data/error_fields/mse_5l_i123_c32s_padR_schLrG0p95_reg0_TrainErrFields.npy")
# except:
#     pass
try:
    test_mse_error_fields = np.load("/projects/ecrpstats/distributional_SRCNN/time_ordered_train0p7/stage_1/mse_5l_i123_c32s_padR_schLrG0p95_reg0_errFields.npy")
    train_mse_error_fields = np.load("/projects/ecrpstats/distributional_SRCNN/time_ordered_train0p7/stage_1/mse_5l_i123_c32s_padR_schLrG0p95_reg0_TrainErrFields.npy")
except:
    pass


# %%
mse_error_fields = np.concatenate((train_mse_error_fields, test_mse_error_fields), axis=0)


# %% [markdown]
# Define basis functions

# %%
N = 64
def compl_dft_basis(x, y, k_x, k_y):
    return np.exp(1j * 2*np.pi * (k_x*x + k_y*y)/N)

# %%
x = np.arange(0, N)

xg = np.tile(x, N)            # generates x-coordinates for the entire grid N*K 
yg = np.repeat(x, N, axis=0)

mat_xg = xg.reshape((64, 64))
mat_yg = yg.reshape((64, 64))

# %%
## Create basis functions to max period
# max_T = 20
# min_k = int(N/max_T)
min_k = 0
print(f"min k : {min_k}")
# max_k = (N)//2
max_k = (N)//2 + 1
print(f"max k : {max_k}")

basis_function_k_idx = []
basis_functions = []
for _kx in range(min_k, max_k):
    for _ky in range(min_k, max_k):
        basis_functions.append(
            compl_dft_basis(
                mat_xg, 
                mat_yg, 
                _kx, 
                _ky
                )
            )
        basis_function_k_idx.append((_kx, _ky))
        
basis_functions = np.asarray(basis_functions).reshape((-1, 64**2)).T
basis_functions.shape

# %%
## Generate complex tensor
basis_functions_tensor = torch.tensor(basis_functions, dtype=torch.complex64).to(device)

##
## Load parameters for global prior
try:
    global_fit_params = np.load("/projects/ecrpstats/distributional_SRCNN/time_ordered_train0p7/stage_2/anal_sln_global_params.npy")
except:
    pass

global_fit_params = torch.tensor(global_fit_params, dtype=torch.float32).to(device)

## Load emperical standard deviation for parameter prior
global_fit_param_stdd = np.load("/projects/ecrpstats/distributional_SRCNN/time_ordered_train0p7/stage_2/img_spec_params_std.npy")
global_fit_param_stdd = torch.tensor(global_fit_param_stdd).to(device)

# %%
def gaussian_loss_basis_cov(
        y_mean, y_data, 
        log_parmas,
        ):
    
    ## Define global prior
    prior = (global_fit_params - torch.exp(log_parmas))**2 / (global_fit_param_stdd / 3.5)**2
    
    ## Change type of real parameters to complex for consistent typing with complex basis functions
    log_parmas = torch.complex(log_parmas, torch.zeros_like(log_parmas))
    
    ## Invert covariance in terms of log(params)
    cov_inv = (
        torch.matmul(
            basis_functions_tensor/N, 
            torch.matmul(
                torch.diag_embed(
                    1.0
                    /
                    ## Exponentiate log(params)
                    (torch.exp(log_parmas))
                    ),
                torch.conj(basis_functions_tensor.T)/N
                )
            )
    )

    # log_part = torch.sum(torch.log(log_parmas))
    d = N**2
    log_part = torch.sum(
        log_parmas 
        # * (d+1)
        )

    ## Construct exponential argument    
    y_mean = torch.reshape(y_mean, [-1, hr_data_size**2])
    y_data = torch.reshape(y_data, [-1, hr_data_size**2])
    # calculate covariance-weighted inner-product
    y_diff = y_data - y_mean
    
    yt_Cinv_y = torch.matmul(
        torch.unsqueeze(y_diff, 1), 
        torch.matmul(
            cov_inv,
            torch.unsqueeze(y_diff, 2)
            )
        )
    
    # calculate Gaussian loss
    exp_part = yt_Cinv_y.squeeze_([1, 2])
    
    return torch.real(torch.sum(exp_part) + log_part) + torch.sum(prior)


# %%
mse_error_fields.shape

# %%
err_fields_tensor = torch.tensor(mse_error_fields, dtype=torch.complex64)
err_fields_tensor = err_fields_tensor.to(device)

zeros_tensor = torch.zeros((1, hr_data_size, hr_data_size), dtype=torch.complex64)
zeros_tensor = zeros_tensor.to(device)

def _gaussian_loss_basis_cov_params(
    params, img_idx
    ):
    """ Fit wrapper"""
    
    result = gaussian_loss_basis_cov(
        log_parmas=params,
        y_mean=err_fields_tensor[img_idx], 
        y_data=zeros_tensor,  
        )
    return result

# %% [markdown]
# Minimizer

# %%
def minimize(function, initial_parameters, epochs, lr=0.1):
    list_params = []
    params = initial_parameters.to(device)
    params.requires_grad_()
    optimizer = torch.optim.Adam([params], lr=lr)
    losses = []

    for i in range(epochs):
        optimizer.zero_grad()
        loss = function(params)
        loss.backward()
        optimizer.step()
        these_params = params.detach().clone()
        list_params.append(these_params.to("cpu")) #here
        losses.append(loss)
        print(f"epoch {i} loss: {loss}")
        print(f"~~~~~~~~~~~~~~~~~~~~~~")
        
    return params, list_params, losses

starting_point = torch.tensor(
    [.0]*basis_functions.shape[1],
    dtype=torch.float32
    ).to(device)

# %%
fit_params = []

for img_idx in range(err_fields_tensor.shape[0]):

    def this_loss(params):
        return _gaussian_loss_basis_cov_params(params, img_idx)
        
    minimized_params, list_of_params, loss = minimize(
        this_loss, starting_point, epochs=150,
        lr=0.1
        )
    
    fit_params.append(torch.exp(minimized_params).detach().to("cpu").numpy())

# %%
fit_params = np.asarray(fit_params)

# %%
np.save(f'{log_file_base_path}_param_fits.npy', fit_params)

