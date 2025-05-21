# %%
# %%
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
import plotting

import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime

train_or_load = 'train'
# plot_epoch = '300' 
plot_epoch = None


# %%
hr_data_size = 64
lr_data_size = 8
# %%
epochs = 300
batch_size = 32
##
# num_sing_mode = 50
##
region = 0
subregion = 'all'

try:
    log_file_base_path = os.path.basename(__file__)[:-3]
except:
    log_file_base_path = 'dev'
print(f"Running {log_file_base_path}")

save_path = f'{log_file_base_path}.pt'

## ~~~~~~~~~~~
## import data
xtrainHR, xtestHR, xtrainLR, xtestLR = data_loading.import_data(
    region,
    subregion,
    train_fraction=.75,
    order='(subregion, time)' 
    )      

# set device
device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
    )
print(f"Using device: {device}")

# set up logging
if train_or_load == 'train':
    logging_set_up(log_file_base_path)
    logger = logging_get_logger()
else:
    logger = None


## create data loader
## Need to add 'labels' arg to keep track of covariance parameters
dataloader_train = data_loading.create_dataloader(
    xtrainLR, xtrainHR, batch_size,
    # labels=torch.arange(xtrainLR.shape[0])
    )


# create model
input_size  = int(np.prod(xtrainLR.shape[1:]))
output_size = int(np.prod(xtrainHR.shape[1:]))
#
net = Conv2dUpscaleModelInterpolate(
    input_channels=1,
    hidden_conv_layers_channels=[32, 32, 32, 32, 32, 1],
    hidden_conv_layers_kernels=[3, 3, 3, 3, 3, 3],
    # hidden_conv_layers_activation=nn.ReLU(),
    # hidden_conv_layers_kwargs={},
    upscale_layer_indices=[1, 2, 3],
    output_activation=None,
    use_dropout=False,
    scale_factor=[2, 2, 2],
    padding_mode='replicate',
)
net.to(device)

# create optimizer and loss function
optimizer = torch.optim.Adam(
    net.parameters(), 
    lr=0.01,
    )

# Custom learning rate scheduler
class ExponentialDecayWithFloor(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay_rate, floor, last_epoch=-1):
        self.decay_rate = decay_rate
        self.floor = floor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Compute new learning rate, ensuring it doesn't go below the floor
        return [
            max(base_lr * self.decay_rate ** (self.last_epoch + 1), self.floor)
            for base_lr in self.base_lrs
        ]
    
# Initialize the scheduler
scheduler = ExponentialDecayWithFloor(optimizer, decay_rate=0.95, floor=0.0001)


## Load fit parameters
fit_params = np.load("/projects/ecrpstats/distributional_SRCNN/sr_ordered_train0p75/stage_2/anal_sln_global_params.npy")
fit_params = torch.tensor(fit_params).to(device)

## Define basis functions
def compl_dft_basis(x, y, k_x, k_y):
    return np.exp(1j * 2*np.pi * (k_x*x + k_y*y)/N)
## Generate basis function domain
N = 64
x = np.arange(0, N)
xg = np.tile(x, N)            # generates x-coordinates for the entire grid N*K 
yg = np.repeat(x, N, axis=0)
mat_xg = xg.reshape((N, N))
mat_yg = yg.reshape((N, N))
## Compute the functions
min_k = 0
max_k = (N)//2 + 1
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
basis_functions = np.asarray(basis_functions).reshape((-1, N**2)).T

## Divide by N so they are sum normalized
basis_functions = basis_functions / N

## Generate complex tensor
basis_functions_tensor = torch.tensor(basis_functions, dtype=torch.complex64).to(device)
basis_functions_tensor = basis_functions_tensor.unsqueeze(0)


def gaussian_loss_emp_cov(
        y_mean, y_data, 
        # labels,
        pixel_count=hr_data_size**2,
        ):
    
    y_mean = torch.reshape(y_mean, [-1, pixel_count])
    y_data = torch.reshape(y_data, [-1, pixel_count])
    # calculate covariance-weighted inner-product
    y_diff = y_data - y_mean

    ## Get covaraince params
    _params = fit_params
    
    ## Calculate determinant term per batch    
    log_part = torch.sum(torch.log(_params), axis=-1)

    ## Convert to complex
    _params = torch.complex(_params, torch.zeros_like(_params))

    ## Compute the inverse covariance
    cov_inv = (
        torch.matmul(
            basis_functions_tensor, 
            torch.matmul(
                torch.diag_embed(
                    1.0
                    /
                    ## Exponentiate log(params)
                    (_params)
                    ),
                torch.conj(torch.transpose(basis_functions_tensor, 1, 2))
                )
            )
        )
    ## Take real part of inverse covariance
    cov_inv = torch.real(cov_inv)

    # print(f"y_diff.shape: {y_diff.shape}")

    yt_Cinv_y = torch.matmul(
        torch.unsqueeze(y_diff, 1), 
        torch.matmul(
            cov_inv,
            torch.unsqueeze(y_diff, 2)
            )
        )
    
    # calculate Gaussian loss
    exp_part = yt_Cinv_y.squeeze_([1, 2])

    # print(f"exp_part.shape: {exp_part.shape}")
    # print(f"log_part.shape: {log_part.shape}")

    return torch.sum(
        exp_part 
        + log_part
        )

def _gaussian_loss_emp_cov(y_pred, y_data):
    result = gaussian_loss_emp_cov(
        y_pred, y_data,
        pixel_count=hr_data_size**2
        )
    return result

loss_fn = _gaussian_loss_emp_cov

## Need define mse loss for model load
def mse_loss(outputs, targets):
    y_mean = torch.flatten(outputs, 1)  ## Returns shape (1, 56**2)
    y_data = torch.flatten(targets, 1)
    
    y_diff = y_data - y_mean
    yt_y = torch.unsqueeze(y_diff, 1) @ torch.unsqueeze(y_diff, 2)
    return torch.mean(yt_y)

## ~~~~~~~~~~~
if train_or_load == 'train':
    ## Load previous model to train off of learned weights
    desired_file = "/projects/ecrpstats/distributional_SRCNN/sr_ordered_train0p75/stage_1/mse_5l_i123_c32s_padR_schLrG0p95_reg0/2025-05-21_t134912/499_l6.948e+01.pt"
    loaded = torch.load(desired_file)
    net.load_state_dict(loaded['model_state_dict'])
    
    ## Train model
    train_log = train_epochs(
        epochs, net, dataloader_train, optimizer, loss_fn,
        device=device, logger=logger, 
        checkpoint_dir=log_file_base_path,
        checkpoint_epochs=100,
        lr_scheduler=scheduler,
        )
elif train_or_load == 'load':
    ## Load model
    # Get last model checkpoint
    runs_dir = f"./{log_file_base_path[:-5]}"
    list_of_run_dirs = os.listdir(runs_dir)
    last_model_dir = max([f'./{runs_dir}/{run_dir}' for run_dir in list_of_run_dirs], key=os.path.getctime)
    list_of_model_checks = os.listdir(last_model_dir)

    if plot_epoch is not None:
        # Filter files that contain '100_' in their filename
        filtered_model_checks = [f'{last_model_dir}/{model_check}' for model_check in list_of_model_checks if f'{plot_epoch}_' in model_check]

        # Get the first file from the filtered list (sorted by creation time, optional)
        if filtered_model_checks:
            desired_file = min(filtered_model_checks, key=os.path.getctime)
            print(f"Found file: {desired_file}")
        else:
            print(f"No file found containing '{plot_epoch}_'")
    else:
        desired_file = max([f'{last_model_dir}/{model_check}' for model_check in list_of_model_checks], key=os.path.getctime)

    loaded = torch.load(desired_file)
    net.load_state_dict(loaded['model_state_dict'])




# %% [markdown]
# Plot results
net.eval()

# %%
dataloader_test = data_loading.create_dataloader(xtestLR, xtestHR, batch_size)

# %%
def prep_predictions_for_plot(dataloader_test):
    with torch.no_grad():
        
        batched_inputs, batched_pred_mean, batched_pred_stdd, batched_targets = [], [], [], []

        for batch_idx, data in enumerate(dataloader_test):
            inputs, targets = data
            batched_inputs.append(inputs) 
            batched_targets.append(targets)

            ## Return mean and std estimates
            inputs = inputs.to(device)
            (
                pred_mean#, 
                # pred_stdd 
                ) = net(inputs)
            batched_pred_mean.append(pred_mean)
            # batched_pred_stdd.append(pred_stdd)

    inputs = torch.cat(batched_inputs).reshape((-1, lr_data_size, lr_data_size))
    pred_mean = torch.cat(batched_pred_mean).reshape((-1, hr_data_size, hr_data_size))
    # pred_stdd = torch.cat(batched_pred_stdd).reshape((-1, hr_data_size, hr_data_size))
    targets = torch.cat(batched_targets).reshape((-1, hr_data_size, hr_data_size))

    return (
        inputs, 
        pred_mean, 
        # pred_stdd, 
        targets
        )


# %%
(
    inputs, 
    pred_mean, 
    # pred_stdd, 
    targets
    ) = prep_predictions_for_plot(dataloader_test)
## Move back to cpu
inputs = inputs.cpu()
pred_mean = pred_mean.cpu()
targets = targets.cpu()

## Save error fields for analysis
error_fields = pred_mean - targets
np.save(f'{log_file_base_path}_errFields', error_fields)

# %%
plotter = plotting.plot_w_interpolator(hr_data_size=64)

# %%
plotter.look_at_prediction_vs_interp(0, pred_mean, inputs, targets, match_colorbars=True)
plt.savefig(f'{log_file_base_path}_img0.pdf', bbox_inches='tight')

# %%
plotter.look_at_prediction_vs_interp(1, pred_mean, inputs, targets, match_colorbars=True)
plt.savefig(f'{log_file_base_path}_img1.pdf', bbox_inches='tight')
# %%
plotter.look_at_prediction_vs_interp(2, pred_mean, inputs, targets, match_colorbars=True)
plt.savefig(f'{log_file_base_path}_img2.pdf', bbox_inches='tight')
# %%
plotter.look_at_prediction_vs_interp(3, pred_mean, inputs, targets, match_colorbars=True)
plt.savefig(f'{log_file_base_path}_img3.pdf', bbox_inches='tight')
# %% 
plotter.look_at_prediction_vs_interp(4, pred_mean, inputs, targets, match_colorbars=True)
plt.savefig(f'{log_file_base_path}_img4.pdf', bbox_inches='tight')

# %% [markdown]
# Quantify MAPE across images

# %%
pred_mapes = []
inte_mapes = []

for i in range(pred_mean.shape[0]):
    pred_mapes.append(
        plotting.mean_absolute_percentage_error(hr_data=targets[i], pred=pred_mean[i])
        )
    
    _interp = plotter.gen_interp(inputs[i].numpy())

    inte_mapes.append(
        plotting.mean_absolute_percentage_error(hr_data=targets[i], pred=_interp)
        )

# %%
pred_mapes = np.asarray(pred_mapes)
inte_mapes = np.asarray(inte_mapes)

## Mape plots
plt.figure(dpi=600, figsize=(2,2))
plt.hist(pred_mapes-inte_mapes, bins=50)
plt.vlines(0, *plt.gca().get_ylim(), color='k', zorder=1, ls='--')
plt.ylabel('number of images')
plt.xlabel(r"Pred MAPE $-$ Interp MAPE")
plt.savefig(f'{log_file_base_path}_mape.pdf', bbox_inches='tight')

## cut out outliers to 2 Stds
plt.figure(dpi=600, figsize=(2,2))
dat = pred_mapes-inte_mapes
mean = np.mean(dat)
stdd = np.std(dat)
dat = dat[dat < mean + 2*stdd]
dat = dat[dat > mean - 2*stdd]
plt.hist(dat, bins=50)
plt.vlines(0, *plt.gca().get_ylim(), color='k', zorder=1, ls='--')
plt.ylabel('number of images')
plt.xlabel(r"Pred MAPE $-$ Interp MAPE")
plt.savefig(f'{log_file_base_path}_mapeTo2std.pdf', bbox_inches='tight')

## cut out outliers to 1 Std
plt.figure(dpi=600, figsize=(2,2))
dat = dat[dat < mean + 1*stdd]
dat = dat[dat > mean - 1*stdd]
plt.hist(dat, bins=50)
plt.vlines(0, *plt.gca().get_ylim(), color='k', zorder=1, ls='--')
plt.ylabel('number of images')
plt.xlabel(r"Pred MAPE $-$ Interp MAPE")
plt.savefig(f'{log_file_base_path}_mapeTo1std.pdf', bbox_inches='tight')

## Calculate grad Mape
def grad_mag(img):
    grad = np.gradient(img)
    return (grad[0]**2 + grad[1]**2)**0.5

pred_gm_mapes = []
inte_gm_mapes = []

for i in range(pred_mean.shape[0]):
    target_grad_mag = grad_mag(targets[i])
    pred_grad_mag = grad_mag(pred_mean[i])
    pred_gm_mapes.append(
        plotting.mean_absolute_percentage_error(hr_data=target_grad_mag, pred=pred_grad_mag)
        )
    
    _interp = plotter.gen_interp(inputs[i].numpy()).reshape((hr_data_size, hr_data_size))
    interp_grad_mag = grad_mag(_interp)

    inte_gm_mapes.append(
        plotting.mean_absolute_percentage_error(hr_data=target_grad_mag, pred=interp_grad_mag)
        )

# %%
pred_gm_mapes = np.asarray(pred_gm_mapes)
inte_gm_mapes = np.asarray(inte_gm_mapes)

## Save mape and grad mape to file
np.save(f'{log_file_base_path}_mape', pred_mapes)
np.save(f'{log_file_base_path}_grad_mape', pred_gm_mapes)


## Plot training curve
import re

# Initialize an empty list to store the extracted values
epochs = []
loss_means = []
loss_std = []

test_log = f"{log_file_base_path}_info.log"
# Open and read the file line by line
with open(test_log, 'r') as file:
    for line in file:
        # Use regex to search for "loss_mean" followed by a number
        match = re.search(r"epoch\s+([\d]+)", line)
        if match:
            # Append the extracted number to the list
            epochs.append(float(match.group(1)))
        match = re.search(r"loss_mean\s+([\d.e+-]+)", line)
        if match:
            # Append the extracted number to the list
            loss_means.append(float(match.group(1)))
        match = re.search(r"std\s+([\d.e+-]+)", line)
        if match:
            # Append the extracted number to the list
            loss_std.append(float(match.group(1)))

# Convert the list to a numpy array
epochs = np.array(epochs)
loss_means = np.array(loss_means)
loss_std = np.array(loss_std)

plt.figure(figsize=(2,2), dpi=600)
plt.plot(epochs, loss_means)
ylim = plt.gca().get_ylim()
# plt.yscale('log')
plt.fill_between(epochs, loss_means-loss_std, loss_means+loss_std, alpha=.25)
plt.ylim(*ylim)
plt.ylabel('MSE loss')
plt.xlabel('epoch')
plt.savefig(f'{log_file_base_path}_training_curve.pdf', bbox_inches='tight')


## Save training curve
np.save(f'{log_file_base_path}_training_curve', np.array([epochs, loss_means, loss_std]))

